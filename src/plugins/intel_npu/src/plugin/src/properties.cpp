// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin
#include "properties.hpp"

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"

namespace {

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

inline bool isSpecialBothProperty(const std::string& key) {
    return key == ov::hint::performance_mode.name() || key == ov::enable_profiling.name() ||
           key == ov::log::level.name();
}

void filterPropertiesByCompilerSupport(intel_npu::FilteredConfig& config,
                                       const intel_npu::ICompilerAdapter* compiler,
                                       const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                                       const intel_npu::Logger& logger) {
    using namespace intel_npu;
    bool legacy = false;
    std::optional<std::vector<std::string>> compilerSupportList{};
    uint32_t compilerVersion = 0;

    OPENVINO_ASSERT(compiler != nullptr, "Compiler must be present to filter properties by compiler support");

    compilerVersion = compiler->get_version();
    compilerSupportList = compiler->get_supported_options();

    if (!compilerSupportList.has_value()) {
        logger.info("No compiler support options list received! Fallback to version-based option registration");
        legacy = true;
    }

    // Logs
    logger.debug("Compiler version: %u", compilerVersion);
    logger.debug("Legacy registration: %s", legacy ? "true" : "false");
    if (!legacy) {
        const auto& supportedOptions = compilerSupportList.value();
        logger.debug("Compiler supported options list (%zu): ", supportedOptions.size());
        for (const auto& str : supportedOptions) {
            logger.debug("    %s ", str.c_str());
        }
    }

    // Parse enables
    config.walkEnables([&](const std::string& key) {
        bool isEnabled = false;
        auto opt = config.getOpt(key);
        // Special case for some both configs. Don't need compiler for these Both properties.
        // Runtime (plugin-only) options are always enabled
        if (opt.mode() != OptionMode::RunTime && !isSpecialBothProperty(key)) {
            if (legacy) {
                // Compiler or common option in Legacy mode? Checking its supported version
                if (compilerVersion >= opt.compilerSupportVersion()) {
                    isEnabled = true;
                }
            } else {
                // We have compiler, we are not in legacy mode = we have a valid list of supported options
                // Searching in the list
                const auto& supportedOptions = compilerSupportList.value();
                auto it = std::find(supportedOptions.begin(), supportedOptions.end(), key);
                if (it != supportedOptions.end()) {
                    isEnabled = true;
                } else {
                    // Not found in the supported options list.
                    if (compiler != nullptr) {
                        // Checking if it is a private option?
                        isEnabled = compiler->is_option_supported(key);
                    } else {
                        // Not in the list and not a private option = disabling
                        isEnabled = false;
                    }
                }
            }
            if (!isEnabled) {
                logger.debug("Config option %s not supported! Requirements not met.", key.c_str());
            } else {
                logger.debug("Enabled config option %s", key.c_str());
            }
            // update enable flag
            config.enable(key, isEnabled);
        }
    });

    // Special cases
    // NPU_TURBO which might not be supported by compiler, but driver will still use it
    // if it exists in config = driver supports it
    // if compiler->is_option_suported is false = compiler doesn't support it and gets marked disabled by default logic
    // however, if driver supports it, we still need it (and will skip giving it to compiler) = force-enable
    if (backend && backend->isCommandQueueExtSupported()) {
        config.enable(ov::intel_npu::turbo.name(), true);
    }
}

void disableCompilerProperties(intel_npu::FilteredConfig& config, const ov::SoPtr<intel_npu::IEngineBackend>& backend) {
    using namespace intel_npu;
    // Parse enables
    config.walkEnables([&](const std::string& key) {
        auto opt = config.getOpt(key);

        // Special case for some both configs. Don't need compiler for these Both properties.
        // Runtime (plugin-only) options are always enabled
        if (opt.mode() != OptionMode::RunTime && !isSpecialBothProperty(key)) {  // Compiler and common options
            // Disable all compiler options
            config.enable(key, false);
        }
    });

    // Special cases
    // NPU_TURBO might be supported by the driver
    if (backend && backend->isCommandQueueExtSupported()) {
        config.enable(ov::intel_npu::turbo.name(), true);
    }
}

}  // namespace

namespace intel_npu {

// Local helper function for appending platform name to the config
static Config add_platform_to_the_config(Config config, const std::string_view platform) {
    config.update({{ov::intel_npu::platform.name(), std::string(platform)}});
    return config;
}

// Local helper function for retrieving the device name
static auto get_specified_device_name(const Config config) {
    if (config.has<DEVICE_ID>()) {
        return config.get<DEVICE_ID>();
    }
    return std::string();
}

// Heuristically obtained number. Varies depending on the values of PLATFORM and PERFORMANCE_HINT
// Note: this is the value provided by the plugin, application should query and consider it, but may supply its own
// preference for number of parallel requests via dedicated configuration
static int64_t getOptimalNumberOfInferRequestsInParallel(const Config& config) {
    const std::string platform = config.get<PLATFORM>();

    if (platform == ov::intel_npu::Platform::NPU3720) {
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 4;
        } else {
            return 1;
        }
    } else {
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 8;
        } else {
            return 1;
        }
    }
}

Properties::Properties(const PropertiesType pType,
                       const FilteredConfig& config,
                       const std::shared_ptr<Metrics>& metrics,
                       const ov::SoPtr<IEngineBackend>& backend)
    : _pType(pType),
      _config(config),
      _metrics(metrics),
      _backend(backend),
      _logger("Properties", _config.get<LOG_LEVEL>()) {
    registerProperties();
}

Properties::Properties(const Properties& other)
    : Properties([&other]() {
          std::lock_guard<std::mutex> lock(other._mutex);
          return CopyState{other._pType,
                           other._config,
                           other._metrics,
                           other._backend,
                           other._logger,
                           other._currentlyUsedCompiler,
                           other._currentlyUsedPlatform,
                           other._compilerConfigsFilteredByCompiler,
                           other._compatibilityCheckFiltered,
                           other._properties,
                           other._supportedProperties};
      }()) {}

Properties::Properties(CopyState&& state)
    : _pType(state.pType),
      _config(std::move(state.config)),
      _metrics(std::move(state.metrics)),
      _backend(std::move(state.backend)),
      _logger(std::move(state.logger)),
      _currentlyUsedCompiler(state.currentlyUsedCompiler),
      _currentlyUsedPlatform(std::move(state.currentlyUsedPlatform)),
      _compilerConfigsFilteredByCompiler(state.compilerConfigsFilteredByCompiler),
      _compatibilityCheckFiltered(state.compatibilityCheckFiltered),
      _properties(std::move(state.properties)),
      _supportedProperties(std::move(state.supportedProperties)) {}

void Properties::registerProperties() {
    // Reset
    _properties.clear();
    _supportedProperties.clear();

    switch (_pType) {
    case PropertiesType::PLUGIN:
        registerPluginProperties();
        break;
    case PropertiesType::COMPILED_MODEL:
        registerCompiledModelProperties();
        break;
    default:
        OPENVINO_THROW("Invalid plugin configuration!");
        break;
    }

    // 2.3. Common metrics (exposed same way by both Plugin and CompiledModel)
    registerMetric(ov::supported_properties, true, [this](const Config&) {
        return _supportedProperties;
    });

    // 3. Populate supported properties list
    // ========
    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, std::get<1>(property.second)));
        }
    }
}

void Properties::registerPluginProperties() {
    // 1. Configs
    // ========
    // 1.1 simple configs which only return value
    tryRegisterSimpleProperty<PERF_COUNT>(ov::enable_profiling);
    tryRegisterSimpleProperty<PERFORMANCE_HINT>(ov::hint::performance_mode);
    tryRegisterSimpleProperty<EXECUTION_MODE_HINT>(ov::hint::execution_mode);
    tryRegisterSimpleProperty<PERFORMANCE_HINT_NUM_REQUESTS>(ov::hint::num_requests);
    tryRegisterSimpleProperty<COMPILATION_NUM_THREADS>(ov::compilation_num_threads);
    tryRegisterSimpleProperty<INFERENCE_PRECISION_HINT>(ov::hint::inference_precision);
    tryRegisterSimpleProperty<LOG_LEVEL>(ov::log::level);
    tryRegisterSimpleProperty<CACHE_DIR>(ov::cache_dir);
    tryRegisterSimpleProperty<CACHE_MODE>(ov::cache_mode);
    tryRegisterSimpleProperty<COMPILED_BLOB>(ov::hint::compiled_blob);
    tryRegisterSimpleProperty<DEVICE_ID>(ov::device::id);
    tryRegisterSimpleProperty<NUM_STREAMS>(ov::num_streams);
    tryRegisterSimpleProperty<WEIGHTS_PATH>(ov::weights_path);
    tryRegisterSimpleProperty<EXCLUSIVE_ASYNC_REQUESTS>(ov::internal::exclusive_async_requests);
    tryRegisterSimpleProperty<COMPILATION_MODE_PARAMS>(ov::intel_npu::compilation_mode_params);
    tryRegisterSimpleProperty<DMA_ENGINES>(ov::intel_npu::dma_engines);
    tryRegisterSimpleProperty<TILES>(ov::intel_npu::tiles);
    tryRegisterSimpleProperty<COMPILATION_MODE>(ov::intel_npu::compilation_mode);
    tryRegisterSimpleProperty<COMPILER_TYPE>(ov::intel_npu::compiler_type);
    tryRegisterSimpleProperty<PLATFORM>(ov::intel_npu::platform);
    tryRegisterSimpleProperty<CREATE_EXECUTOR>(ov::intel_npu::create_executor);
    tryRegisterSimpleProperty<DYNAMIC_SHAPE_TO_STATIC>(ov::intel_npu::dynamic_shape_to_static);
    tryRegisterSimpleProperty<PROFILING_TYPE>(ov::intel_npu::profiling_type);
    tryRegisterSimpleProperty<BACKEND_COMPILATION_PARAMS>(ov::intel_npu::backend_compilation_params);
    tryRegisterSimpleProperty<BATCH_MODE>(ov::intel_npu::batch_mode);
    tryRegisterSimpleProperty<TURBO>(ov::intel_npu::turbo);
    tryRegisterSimpleProperty<MODEL_PRIORITY>(ov::hint::model_priority);
    tryRegisterSimpleProperty<BYPASS_UMD_CACHING>(ov::intel_npu::bypass_umd_caching);
    tryRegisterSimpleProperty<DEFER_WEIGHTS_LOAD>(ov::intel_npu::defer_weights_load);
    tryRegisterSimpleProperty<COMPILER_DYNAMIC_QUANTIZATION>(ov::intel_npu::compiler_dynamic_quantization);
    tryRegisterSimpleProperty<QDQ_OPTIMIZATION>(ov::intel_npu::qdq_optimization);
    tryRegisterSimpleProperty<QDQ_OPTIMIZATION_AGGRESSIVE>(ov::intel_npu::qdq_optimization_aggressive);
    tryRegisterSimpleProperty<DISABLE_VERSION_CHECK>(ov::intel_npu::disable_version_check);
    tryRegisterSimpleProperty<EXPORT_RAW_BLOB>(ov::intel_npu::export_raw_blob);
    tryRegisterSimpleProperty<IMPORT_RAW_BLOB>(ov::intel_npu::import_raw_blob);
    tryRegisterSimpleProperty<BATCH_COMPILER_MODE_SETTINGS>(ov::intel_npu::batch_compiler_mode_settings);
    tryRegisterSimpleProperty<ENABLE_CPU_PINNING>(ov::hint::enable_cpu_pinning);
    tryRegisterSimpleProperty<WORKLOAD_TYPE>(ov::workload_type);
    tryRegisterSimpleProperty<ENABLE_WEIGHTLESS>(ov::enable_weightless);
    tryRegisterSimpleProperty<SEPARATE_WEIGHTS_VERSION>(ov::intel_npu::separate_weights_version);
    tryRegisterSimpleProperty<MODEL_SERIALIZER_VERSION>(ov::intel_npu::model_serializer_version);
    tryRegisterSimpleProperty<ENABLE_STRIDES_FOR>(ov::intel_npu::enable_strides_for);
    tryRegisterSimpleProperty<DISABLE_IDLE_MEMORY_PRUNING>(ov::intel_npu::disable_idle_memory_prunning);
    tryRegisterSimpleProperty<SHARED_COMMON_QUEUE>(ov::intel_npu::shared_common_queue);

    tryRegisterCustomFuncProperty(ov::intel_npu::stepping, [this](const Config& config) {
        if (!config.has<STEPPING>()) {
            try {
                const auto specifiedDeviceName = get_specified_device_name(config);
                return static_cast<int64_t>(_metrics->GetSteppingNumber(specifiedDeviceName));
            } catch (...) {
                _logger.warning("Metrics GetSteppingNumber failed to get value from device.");
            }
        }
        return config.get<STEPPING>();
    });
    tryRegisterCustomFuncProperty(ov::intel_npu::max_tiles, [this](const Config& config) {
        if (!config.has<MAX_TILES>()) {
            try {
                const auto specifiedDeviceName = get_specified_device_name(config);
                return static_cast<int64_t>(_metrics->GetMaxTiles(specifiedDeviceName));
            } catch (...) {
                _logger.warning("Metrics GetMaxTiles failed to get value from device.");
            }
        }
        return config.get<MAX_TILES>();
    });

    tryRegisterVarpubProperty<RUN_INFERENCES_SEQUENTIALLY>(ov::intel_npu::run_inferences_sequentially, [this] {
        if (_backend && _backend->getInitStructs()) {
            if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                return true;
            }
        }
        return false;
    }());
    tryRegisterCustomProperty(ov::compatibility_check,
                              true,
                              ov::PropertyMutability::RO,
                              [](const Config& /* unusedConfig */) {
                                  // This property is implemented in the plugin directly
                                  // This implementation here serves only to publish it in supported_properties
                                  return false;
                              });

    tryRegisterCustomProperty(ov::cache_encryption_callbacks,
                              true,
                              ov::PropertyMutability::WO,
                              [](const Config& /* unusedConfig */) {
                                  return ov::EncryptionCallbacks{nullptr, nullptr};
                              });

    forceRegisterCustomProperty(ov::hint::model,
                                true,
                                ov::PropertyMutability::RO,
                                [](const Config& /* unusedConfig */) {
                                    return std::shared_ptr<const ov::Model>(nullptr);
                                });

    // NPUW properties are requested by OV Core during caching and have no effect on the NPU plugin. But we still need
    // to enable those for OV Core to query.
    for_each_exposed_npuw_option([this](auto tag) {
        using Opt = typename decltype(tag)::type;
        tryRegisterNpuwOptionProperty<Opt>();
    });

    // 2. Metrics (static device and enviroment properties)
    // ========
    if (_metrics != nullptr) {
        registerMetric(ov::available_devices, true, [this](const Config&) {
            return _metrics->GetAvailableDevicesNames();
        });
        registerMetric(ov::device::capabilities, true, [this](const Config&) {
            return _metrics->GetOptimizationCapabilities();
        });
        registerMetric(ov::optimal_number_of_infer_requests, true, [this](const Config& config) {
            return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
                config,
                utils::getCompilationPlatform(
                    config.get<PLATFORM>(),
                    _backend == nullptr ? config.get<DEVICE_ID>()
                                        : _backend->getDevice(config.get<DEVICE_ID>())->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames()))));
        });
        registerMetric(ov::range_for_async_infer_requests, true, [this](const Config&) {
            return _metrics->GetRangeForAsyncInferRequest();
        });
        registerMetric(ov::range_for_streams, true, [this](const Config&) {
            return _metrics->GetRangeForStreams();
        });
        registerMetric(ov::device::pci_info, true, [this](const Config& config) {
            return _metrics->GetPciInfo(get_specified_device_name(config));
        });
        registerMetric(ov::device::gops, true, [this](const Config& config) {
            return _metrics->GetGops(get_specified_device_name(config));
        });
        registerMetric(ov::device::type, true, [this](const Config& config) {
            return _metrics->GetDeviceType(get_specified_device_name(config));
        });
        registerMetric(ov::internal::supported_properties,
                       false,
                       [this](const Config&) -> const std::vector<ov::PropertyName>& {
                           return _internalSupportedProperties;
                       });
        registerMetric(ov::internal::cache_header_alignment, false, [](const Config&) {
            return utils::STANDARD_PAGE_SIZE;
        });
        registerMetric(ov::intel_npu::device_alloc_mem_size, true, [this](const Config& config) {
            return _metrics->GetDeviceAllocMemSize(get_specified_device_name(config));
        });
        registerMetric(ov::intel_npu::device_total_mem_size, true, [this](const Config& config) {
            return _metrics->GetDeviceTotalMemSize(get_specified_device_name(config));
        });
        registerMetric(ov::intel_npu::driver_version, true, [this](const Config&) {
            return _metrics->GetDriverVersion();
        });
        registerMetric(ov::intel_npu::backend_name, false, [this](const Config&) {
            return _metrics->GetBackendName();
        });
        registerMetric(ov::device::architecture,
                       !_metrics->GetAvailableDevicesNames().empty(),
                       [this](const Config& config) {
                           const auto specifiedDeviceName = get_specified_device_name(config);
                           return _metrics->GetDeviceArchitecture(specifiedDeviceName);
                       });
        registerMetric(ov::device::full_name,
                       !_metrics->GetAvailableDevicesNames().empty(),
                       [this](const Config& config) {
                           const auto specifiedDeviceName = get_specified_device_name(config);
                           return _metrics->GetFullDeviceName(specifiedDeviceName);
                       });
        registerMetric(ov::device::luid,
                       _backend == nullptr ? false : _backend->isLUIDExtSupported(),
                       [this](const Config& config) {
                           const auto specifiedDeviceName = get_specified_device_name(config);
                           return _metrics->GetDeviceLUID(specifiedDeviceName);
                       });
        registerMetric(ov::device::uuid, true, [this](const Config& config) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
            return decltype(ov::device::uuid)::value_type{devUuid};
        });
        registerMetric(ov::execution_devices, true, [this](const Config& config) {
            if (_metrics->GetAvailableDevicesNames().size() > 1) {
                return std::string("NPU." + config.get<DEVICE_ID>());
            } else {
                return std::string("NPU");
            }
        });
        registerMetric(ov::intel_npu::compiler_version, true, [this](const Config& config) {
            auto compilerType = config.get<COMPILER_TYPE>();
            auto deviceId = config.get<DEVICE_ID>();
            auto device = utils::getDeviceById(_backend, deviceId);

            auto compilationPlatform = utils::getCompilationPlatform(
                config.get<PLATFORM>(),
                device == nullptr ? std::move(deviceId) : device->getName(),
                _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

            CompilerAdapterFactory factory;
            auto dummyCompiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

            return dummyCompiler->get_version();
        });
        registerMetric(ov::internal::caching_properties, false, [this](const Config&) {
            // return a dynamically created list based on what is supported in current configuration
            std::vector<ov::PropertyName> caching_props{};
            // walk the static caching properties, add only what is supported now
            for (auto prop : _cachingProperties) {
                if (_config.isAvailable(prop)) {
                    caching_props.emplace_back(prop);
                }
            }
            // NPUW properties are requested by OV Core during caching and have no effect on the NPU plugin. But we
            // still need to enable those for OV Core to query. add all NPUW properties
            for (auto prop : _cachingProperties) {
                if (prop.find("NPUW") != prop.npos) {
                    caching_props.emplace_back(prop);
                }
            }
            return caching_props;
        });
    }
}

void Properties::registerCompiledModelProperties() {
    // 1. Configs
    // ========

    // Permanent properties
    tryRegisterSimpleProperty<ENABLE_CPU_PINNING>(ov::hint::enable_cpu_pinning);
    tryRegisterSimpleProperty<LOG_LEVEL>(ov::log::level);
    tryRegisterSimpleProperty<LOADED_FROM_CACHE>(ov::loaded_from_cache);
    tryRegisterSimpleProperty<PERFORMANCE_HINT>(ov::hint::performance_mode);
    tryRegisterSimpleProperty<EXECUTION_MODE_HINT>(ov::hint::execution_mode);
    tryRegisterSimpleProperty<PERFORMANCE_HINT_NUM_REQUESTS>(ov::hint::num_requests);
    tryRegisterSimpleProperty<COMPILATION_NUM_THREADS>(ov::compilation_num_threads);
    tryRegisterSimpleProperty<INFERENCE_PRECISION_HINT>(ov::hint::inference_precision);
    tryRegisterSimpleProperty<CACHE_MODE>(ov::cache_mode);

    // Properties we shall only enable if they were set prior-to-compilation
    tryRegisterCompiledModelPropertyIfSet<COMPILER_TYPE>(ov::intel_npu::compiler_type);
    tryRegisterCompiledModelPropertyIfSet<COMPILER_VERSION>(ov::intel_npu::compiler_version);
    tryRegisterCompiledModelPropertyIfSet<WEIGHTS_PATH>(ov::weights_path);
    tryRegisterCompiledModelPropertyIfSet<CACHE_DIR>(ov::cache_dir);
    tryRegisterCompiledModelPropertyIfSet<PERF_COUNT>(ov::enable_profiling);
    tryRegisterCompiledModelPropertyIfSet<PROFILING_TYPE>(ov::intel_npu::profiling_type);
    tryRegisterCompiledModelPropertyIfSet<TURBO>(ov::intel_npu::turbo);
    tryRegisterCompiledModelPropertyIfSet<COMPILATION_MODE_PARAMS>(ov::intel_npu::compilation_mode_params);
    tryRegisterCompiledModelPropertyIfSet<DMA_ENGINES>(ov::intel_npu::dma_engines);
    tryRegisterCompiledModelPropertyIfSet<TILES>(ov::intel_npu::tiles);
    tryRegisterCompiledModelPropertyIfSet<COMPILATION_MODE>(ov::intel_npu::compilation_mode);
    tryRegisterCompiledModelPropertyIfSet<PLATFORM>(ov::intel_npu::platform);
    tryRegisterCompiledModelPropertyIfSet<DYNAMIC_SHAPE_TO_STATIC>(ov::intel_npu::dynamic_shape_to_static);
    tryRegisterCompiledModelPropertyIfSet<BACKEND_COMPILATION_PARAMS>(ov::intel_npu::backend_compilation_params);
    tryRegisterCompiledModelPropertyIfSet<BYPASS_UMD_CACHING>(ov::intel_npu::bypass_umd_caching);
    tryRegisterCompiledModelPropertyIfSet<DEFER_WEIGHTS_LOAD>(ov::intel_npu::defer_weights_load);
    tryRegisterCompiledModelPropertyIfSet<COMPILER_DYNAMIC_QUANTIZATION>(ov::intel_npu::compiler_dynamic_quantization);
    tryRegisterCompiledModelPropertyIfSet<QDQ_OPTIMIZATION>(ov::intel_npu::qdq_optimization);
    tryRegisterCompiledModelPropertyIfSet<QDQ_OPTIMIZATION_AGGRESSIVE>(ov::intel_npu::qdq_optimization_aggressive);
    tryRegisterCompiledModelPropertyIfSet<DISABLE_VERSION_CHECK>(ov::intel_npu::disable_version_check);
    tryRegisterCompiledModelPropertyIfSet<EXPORT_RAW_BLOB>(ov::intel_npu::export_raw_blob);
    tryRegisterCompiledModelPropertyIfSet<IMPORT_RAW_BLOB>(ov::intel_npu::import_raw_blob);
    tryRegisterCompiledModelPropertyIfSet<BATCH_COMPILER_MODE_SETTINGS>(ov::intel_npu::batch_compiler_mode_settings);
    tryRegisterCompiledModelPropertyIfSet<RUN_INFERENCES_SEQUENTIALLY>(ov::intel_npu::run_inferences_sequentially);
    tryRegisterCompiledModelPropertyIfSet<ENABLE_WEIGHTLESS>(ov::enable_weightless);
    tryRegisterCompiledModelPropertyIfSet<SEPARATE_WEIGHTS_VERSION>(ov::intel_npu::separate_weights_version);
    tryRegisterCompiledModelPropertyIfSet<ENABLE_STRIDES_FOR>(ov::intel_npu::enable_strides_for);

    tryRegisterVarpubProperty<BATCH_MODE>(ov::intel_npu::batch_mode, false);
    tryRegisterVarpubProperty<SHARED_COMMON_QUEUE>(ov::intel_npu::shared_common_queue, false);

    tryRegisterCustomProperty(ov::hint::model_priority, true, ov::PropertyMutability::RW, [](const Config& config) {
        return config.get<MODEL_PRIORITY>();
    });

    tryRegisterCustomProperty(ov::workload_type, true, ov::PropertyMutability::RW, [](const Config& config) {
        return config.get<WORKLOAD_TYPE>();
    });

    tryRegisterCustomProperty(ov::cache_encryption_callbacks,
                              true,
                              ov::PropertyMutability::WO,
                              [](const Config& /* unusedConfig */) {
                                  return ov::EncryptionCallbacks{nullptr, nullptr};
                              });

    forceRegisterCustomProperty(ov::hint::model,
                                true,
                                ov::PropertyMutability::RO,
                                [](const Config& /* unusedConfig */) {
                                    return std::shared_ptr<const ov::Model>(nullptr);
                                });
    // 2. Metrics (static device and enviroment properties)
    // ========
    // Note: ov::runtime_requirements and ov::model_name are NOT registered here.
    // They are registered by CompiledModel::CompiledModel() via registerExternalProperty()
    // because their lambdas must capture the IGraph instance, which is only available there.
    // ov::runtime_requirements is only registered if graph->get_compatibility_descriptor().has_value().

    registerMetric(ov::optimal_number_of_infer_requests, true, [](const Config& config) {
        return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(config));
    });
    registerMetric(ov::execution_devices, true, [](const Config&) {
        // TODO: log an error here as the code shouldn't have gotten here
        // this property is implemented in compiled model directly
        // this implementation here serves only to publish it in supported_properties
        return std::string("NPU");
    });
}

ov::Any Properties::getProperty(const std::string& name) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_pType == PropertiesType::PLUGIN) {
        bool propertyIsCompilerConfig = false;
        bool propertyIsRegistered = true;
        // If the property is not registered, there is no point of checking the config.
        if (!isPropertyRegistered(name)) {
            propertyIsRegistered = false;
        } else {
            // Property is already registered but need to re-check if the CompilerTime config is still supported by the
            // current compiler.
            if (_config.hasOpt(name) && !isSpecialBothProperty(name)) {
                auto opt = _config.getOpt(name);
                if (opt.mode() != OptionMode::RunTime) {
                    propertyIsCompilerConfig = true;
                }
            }
        }

        bool needToResetProperties = false;
        if (name == ov::compatibility_check.name() || name == ov::supported_properties.name()) {
            // Mark that properties need to be registered again if the internal config is updated
            needToResetProperties = disable_compatibility_check_if_needed();
        }
        // Special case for Supported Properties and Caching Properties as they are compiler dependent. So we need to
        // check compiler support for those properties on each getProperty call as well.
        if (propertyIsCompilerConfig || !propertyIsRegistered || name == ov::supported_properties.name() ||
            name == ov::internal::caching_properties.name()) {
            std::unique_ptr<ICompilerAdapter> compiler = nullptr;
            auto compilerType = _config.get<COMPILER_TYPE>();
            auto deviceId = _config.get<DEVICE_ID>();
            auto device = utils::getDeviceById(_backend, deviceId);

            auto compilationPlatform = utils::getCompilationPlatform(
                _config.get<PLATFORM>(),
                device == nullptr ? std::move(deviceId) : device->getName(),
                _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

            // Create a compiler to get the type and fetch version and supported options if needed
            CompilerAdapterFactory factory;
            try {
                compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);
            } catch (const std::exception& ex) {
                if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
                    OPENVINO_THROW("Failed to create compiler for getting property ", name, " with error: ", ex.what());
                }

                _logger.warning("Failed to create compiler for getting property %s with error: %s."
                                "Returning only runtime properties and metrics that do not require compiler support.",
                                name.c_str(),
                                ex.what());
            }

            if (compiler != nullptr && !(_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
                                         compilationPlatform == _currentlyUsedPlatform)) {
                // In case properties are not initialized or the compiler/platform was changed since last call -
                // filter out options again
                filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

                _compilerConfigsFilteredByCompiler = true;
                _currentlyUsedCompiler = compilerType;
                _currentlyUsedPlatform = std::move(compilationPlatform);
                needToResetProperties = true;
            }
        }

        if (needToResetProperties) {
            // reset properties for the new options
            registerProperties();
        }
    }

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        if (std::get<1>(configIterator->second) == ov::PropertyMutability::WO) {
            _logger.warning("Trying to get WRITE-ONLY property: %s. Returning empty `ov::Any` object",
                            name.c_str());  // throw OV exception instead
            return ov::Any();
        }
        return std::get<2>(configIterator->second)(_config);
    }
    try {
        return _config.getInternal(name);
    } catch (...) {
        OPENVINO_THROW("Unsupported configuration key: ", name);
    }
}

void Properties::setProperty(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (properties.count(ov::log::level.name()) != 0) {
        _logger.setLevel(properties.at(ov::log::level.name()).as<ov::log::Level>());
    }

    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    if (_pType == PropertiesType::PLUGIN) {
        bool propertyIsCompilerConfig = false;
        bool propertyIsRegistered = true;
        for (const auto& property : properties) {
            if (!isPropertyRegistered(property.first)) {
                propertyIsRegistered = false;
                break;
            }
            // Special case for some both configs. Don't need to check compiler support for these Both properties.
            const bool isNotSpecialBothProperty = !isSpecialBothProperty(property.first);
            if (_config.hasOpt(property.first) && isNotSpecialBothProperty) {
                auto opt = _config.getOpt(property.first);
                if (opt.mode() != OptionMode::RunTime) {
                    propertyIsCompilerConfig = true;
                    break;
                }
            }
        }

        // Check if one of the properties is compiler config which needs to return different values based on compiler
        // and platform configuration
        if (propertyIsCompilerConfig || !propertyIsRegistered) {
            auto compilerType = determineCompilerType(properties);
            auto deviceId = determineDeviceId(properties);
            auto device = utils::getDeviceById(_backend, deviceId);

            auto compilationPlatform = utils::getCompilationPlatform(
                determinePlatform(properties),
                device == nullptr ? std::move(deviceId) : device->getName(),
                _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

            // Create a compiler to get the type and fetch version and supported options if needed
            CompilerAdapterFactory factory;
            compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

            if (!(_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
                  compilationPlatform == _currentlyUsedPlatform)) {
                // In case properties are not initialized or the compiler/platform was changed since last call -
                // filter out options again
                filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

                // reset properties for the new options
                registerProperties();
                _compilerConfigsFilteredByCompiler = true;
                _currentlyUsedCompiler = compilerType;
                _currentlyUsedPlatform = std::move(compilationPlatform);
            }
        }
    }

    std::map<std::string, std::string> cfgs_to_set;
    ov::AnyMap special_cfgs_to_set;
    for (auto&& value : properties) {
        if (_properties.find(value.first) == _properties.end()) {
            // property doesn't exist
            // checking as internal now
            if (compiler != nullptr) {
                if (compiler->is_option_supported(value.first)) {
                    // if compiler reports it supported > registering as internal
                    _config.addOrUpdateInternal(value.first, value.second.as<std::string>());
                } else {
                    OPENVINO_THROW("Unsupported configuration key: ", value.first);
                }
            } else {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
            }
        } else {
            if (std::get<1>(_properties[value.first]) == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", value.first);
            } else if (value.first == ov::cache_encryption_callbacks.name()) {
                special_cfgs_to_set.emplace(value.first, value.second);
            } else {
                cfgs_to_set.emplace(value.first, value.second.as<std::string>());
            }
        }
    }

    if (!cfgs_to_set.empty()) {
        _config.update(cfgs_to_set);
    }

    if (!special_cfgs_to_set.empty()) {
        _config.updateAny(special_cfgs_to_set);
    }
}

bool Properties::isPropertySupported(const std::string& name) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_pType == PropertiesType::PLUGIN) {
        const bool isRegistered = isPropertyRegistered(name);
        const bool isConfigOption = _config.hasOpt(name);

        if (!isRegistered && !isConfigOption) {
            // Property is neither registered nor known by config
            return false;
        }

        if (name == ov::compatibility_check.name()) {
            bool disabled = disable_compatibility_check_if_needed();
            if (disabled) {
                registerProperties();
                return false;
            }
        }

        if (isRegistered) {
            // Registered and not a config option: always supported. Or it is a special both property which is always
            // supported.
            if (!isConfigOption || isSpecialBothProperty(name)) {
                return true;
            }

            // Registered as a config option: runtime mode is always supported.
            auto opt = _config.getOpt(name);
            if (opt.mode() == OptionMode::RunTime) {
                return true;
            }
        }

        // Property is compiler config, need to check compiler support
        std::unique_ptr<ICompilerAdapter> compiler = nullptr;
        auto compilerType = _config.get<COMPILER_TYPE>();
        auto deviceId = _config.get<DEVICE_ID>();
        auto device = utils::getDeviceById(_backend, deviceId);

        auto compilationPlatform = utils::getCompilationPlatform(
            _config.get<PLATFORM>(),
            device == nullptr ? std::move(deviceId) : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        // Create a compiler to get the type and fetch version and supported options if needed
        CompilerAdapterFactory factory;
        try {
            compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);
        } catch (const std::exception& ex) {
            if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
                return false;
            }

            _logger.warning("Failed to create compiler to query property %s with error: %s. "
                            "Registering only runtime properties and metrics that do not require compiler support.",
                            name.c_str(),
                            ex.what());
        }

        if (compiler != nullptr && !(_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
                                     compilationPlatform == _currentlyUsedPlatform)) {
            // In case properties are not initialized or the compiler/platform was changed since last call -
            // filter out options again
            filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

            // reset properties for the new options
            registerProperties();
            _compilerConfigsFilteredByCompiler = true;
            _currentlyUsedCompiler = compilerType;
            _currentlyUsedPlatform = std::move(compilationPlatform);
        }
    }

    if (isPropertyRegistered(name)) {
        return true;
    }

    return false;
}

bool Properties::isPropertyRegistered(const std::string& propertyName) const {
    return _properties.find(propertyName) != _properties.end();
}

FilteredConfig Properties::getConfigForSpecificCompiler(const ov::AnyMap& properties,
                                                        const ICompilerAdapter* compiler) {
    auto [updatedConfig, compilerConfigsFilteredByCompiler, currentlyUsedCompiler, currentlyUsedPlatform, logger] =
        [&]() {
            std::lock_guard<std::mutex> lock(_mutex);
            return std::make_tuple(_config,
                                   _compilerConfigsFilteredByCompiler,
                                   _currentlyUsedCompiler,
                                   _currentlyUsedPlatform,
                                   _logger);
        }();

    std::optional<ov::intel_npu::CompilerType> propertiesCompilerType = std::nullopt;
    std::optional<std::string> propertiesPlatform = std::nullopt;
    if (compilerConfigsFilteredByCompiler) {
        auto compilerType = properties.find(ov::intel_npu::compiler_type.name());
        if (compilerType != properties.end()) {
            propertiesCompilerType = compilerType->second.as<ov::intel_npu::CompilerType>();
        }
    }
    auto platform = properties.find(ov::intel_npu::platform.name());
    if (platform != properties.end()) {
        propertiesPlatform = platform->second.as<std::string>();
    }

    // filter out unsupported options
    if (!(compilerConfigsFilteredByCompiler &&
          propertiesCompilerType.value_or(currentlyUsedCompiler) == currentlyUsedCompiler &&
          propertiesPlatform.value_or(currentlyUsedPlatform) == currentlyUsedPlatform)) {
        // In case the compiler properties are not initialized or the compiler/platform was changed since last call -
        // filter out options again
        filterPropertiesByCompilerSupport(updatedConfig, compiler, _backend, logger);
    }

    const std::map<std::string, std::string> rawConfig = any_copy(properties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (!updatedConfig.hasOpt(key)) {
            // not a known config key
            if (!compiler->is_option_supported(key)) {
                OPENVINO_THROW("[ NOT_FOUND ] Option '", key, "' is not supported for current configuration");
            } else {
                updatedConfig.addOrUpdateInternal(key, value);
            }
        } else if (key == ov::cache_encryption_callbacks.name()) {
            specialCfgsToSet.emplace(key, properties.at(key));
        } else {
            cfgsToSet.emplace(key, value);
        }
    }

    updatedConfig.update(cfgsToSet);
    updatedConfig.updateAny(specialCfgsToSet);

    return std::move(updatedConfig);
}

FilteredConfig Properties::getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties) {
    auto [updatedConfig, compilerConfigsFilteredByCompiler, logger] = [&]() {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::make_tuple(_config, _compilerConfigsFilteredByCompiler, _logger);
    }();

    if (compilerConfigsFilteredByCompiler) {
        disableCompilerProperties(updatedConfig, _backend);
    }

    if (properties.empty()) {
        return std::move(updatedConfig);
    }

    const std::map<std::string, std::string> rawConfig = any_copy(properties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (updatedConfig.hasOpt(key)) {
            const auto optionMode = updatedConfig.getOpt(key).mode();

            if (optionMode == OptionMode::CompileTime) {
                logger.info(
                    "Config key '%s' is recognized as a compiler option, will not be used for current configuration.",
                    key.c_str());
                continue;
            }

            if (optionMode == OptionMode::Both && !updatedConfig.isAvailable(key)) {
                logger.info("Config key '%s' is not enabled by the plugin, will not be used for current configuration.",
                            key.c_str());
                continue;
            }
        }

        if (key == ov::cache_encryption_callbacks.name()) {
            specialCfgsToSet.emplace(key, properties.at(key));
        } else {
            cfgsToSet.emplace(key, value);
        }
    }

    updatedConfig.update(cfgsToSet);
    updatedConfig.updateAny(specialCfgsToSet);

    return std::move(updatedConfig);
}

ov::intel_npu::CompilerType Properties::determineCompilerType(const ov::AnyMap& properties) const {
    // first look if provided config changes compiler type
    auto it = properties.find(ov::intel_npu::compiler_type.name());
    if (it != properties.end()) {
        // if compiler_type is provided by local config = use that
        return COMPILER_TYPE::parse(it->second.as<std::string>());
    }
    // if there is no compiler_type provided = use _config value
    return _config.get<COMPILER_TYPE>();
}

std::string Properties::determinePlatform(const ov::AnyMap& properties) const {
    auto platform = properties.find(ov::intel_npu::platform.name());
    if (platform != properties.end()) {
        return platform->second.as<std::string>();
    }
    return _config.get<PLATFORM>();
}

std::string Properties::determineDeviceId(const ov::AnyMap& properties) const {
    auto device_id = properties.find(std::string(ov::device::id.name()));
    if (device_id != properties.end()) {
        return device_id->second.as<std::string>();
    }
    return _config.get<DEVICE_ID>();
}

bool Properties::disable_compatibility_check_if_needed() {
    // COMPATIBILITY_CHECK is a RunTime option, thus enabled by default
    // The property should be supported only if at least one of the compiler adapters support it.
    // No need to check again if it was enabled already
    // Plugin will prefer the validation to be performed through CID, but it will fallback
    // to the CIP validation otherwise.

    if (_compatibilityCheckFiltered) {
        // The property was already filtered by compiler support, no need to check again
        return false;
    }

    // Mark that the property has been filtered by compiler support, regardless of the result
    _compatibilityCheckFiltered = true;

    CompilerAdapterFactory factory;
    auto compilerType = ov::intel_npu::CompilerType::DRIVER;
    try {
        auto tempCompiler = factory.getCompiler(_backend, compilerType, std::string_view{});
        // If CID is present but does not support the query, fallback to CIP
        if (!tempCompiler->is_option_supported(ov::compatibility_check.name())) {
            compilerType = ov::intel_npu::CompilerType::PLUGIN;
            try {
                tempCompiler = factory.getCompiler(_backend, compilerType, std::string_view{});
                if (!tempCompiler->is_option_supported(ov::compatibility_check.name())) {
                    // Neither of the compiler adapters support the option, it should be disabled
                    _logger.debug("Neither CID nor CIP support the compatibility check! Disabling the property.");
                    _config.enable(ov::compatibility_check.name(), false);
                    return true;  // config was updated
                } else {
                    // CIP is present and supports the option, COMPATIBILITY_CHECK remains enabled
                    _compilerForCompatibilityCheck = ov::intel_npu::CompilerType::PLUGIN;
                }
            } catch (const std::exception&) {
                // CIP is not present either, the property is not supported
                _logger.debug("CIP is not present! Disabling the compatibility check property.");
                _config.enable(ov::compatibility_check.name(), false);
                return true;  // config was updated
            }
        } else {
            // COMPATIBILITY_CHECK remains enabled
            _compilerForCompatibilityCheck = ov::intel_npu::CompilerType::DRIVER;
        }
    } catch (const std::exception&) {
        // If CID is not present (driver is not present either) plugin will not be able to retrieve
        // the device information required for the CIP check, thus the property should not be supported.
        // No need to check the CIP support anymore in this case
        _logger.debug("Driver is not present! Disabling the compatibility check property.");
        _config.enable(ov::compatibility_check.name(), false);
        return true;  // config was updated
    }

    return false;  // config was not updated
}

ov::intel_npu::CompilerType Properties::determineCompilerTypeForCompatibilityCheck() const {
    // The compiler type used for compatibility check is determined based on the support of
    // the COMPATIBILITY_CHECK option in the compiler adapters. If the option is supported
    // in CID, it will be preferred for compatibility check, otherwise CIP will be used
    // if it supports the option.

    return _compilerForCompatibilityCheck;
}

}  // namespace intel_npu
