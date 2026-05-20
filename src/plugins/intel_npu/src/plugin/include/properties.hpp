// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/config/npuw.hpp"
#include "metrics.hpp"

namespace intel_npu {

enum class PropertiesType { PLUGIN, COMPILED_MODEL };

class Properties final {
public:
    /**
     * @brief Properties handler constructor
     * @param pType - type of object this handler gets attached to: PLUGIN or COMPILED_MODEL
     * @param config - reference to the global configuration table of the parent object
     * @param metrics - reference ptr to the metrics object of the parent object (PLUGIN only)
     */
    Properties(const PropertiesType pType,
               const FilteredConfig& config,
               const std::shared_ptr<Metrics>& metrics = nullptr,
               const ov::SoPtr<IEngineBackend>& backend = {nullptr});

    Properties(const Properties& other);
    Properties& operator=(const Properties& other) = delete;

    /**
     * @brief Get the values of a property in a map
     */
    ov::Any getProperty(const std::string& name);

    /**
     * @brief Set the values of a subset of properties, provided as a map
     * @details
     * - checks if the property exists, will report if unsupported
     * - checks if the property is Read-only, will report error if so
     */
    void setProperty(const ov::AnyMap& properties);

    /**
     * @brief Checks if a property is supported by the plugin.
     */
    bool isPropertySupported(const std::string& name);

    /**
     * @brief Get a const reference to the stored config
     */
    const FilteredConfig& getConfig() const {
        return _config;
    }

    /**
     * @brief Updates a copy of the config list based on the provided properties, and returns it.
     * @details
     * - Updates the config with the provided arguments and returns it.
     */
    FilteredConfig getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties);

    /**
     * @brief Updates a copy of the config list based on the provided properties and compiler, and returns it.
     * @details
     * - Checks if the compiler has changed; if so, re-filters configs.
     * - Filters compiler options based on the current compiler.
     * - Updates the config with the provided arguments and returns it.
     */
    FilteredConfig getConfigForSpecificCompiler(const ov::AnyMap& properties, const ICompilerAdapter* compiler);

    std::string determinePlatform(const ov::AnyMap& properties) const;
    std::string determineDeviceId(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerType(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerTypeForCompatibilityCheck() const;

    /**
     * @brief Registers (or replaces) a property with a caller-supplied callback, from outside the class.
     *
     * Intended for properties whose lambdas must capture state that is only available after the Properties
     * object is constructed (e.g. IGraph-derived data in CompiledModel).  Unlike the private helpers,
     * this method uses insert_or_assign so it overwrites any placeholder registered during
     * registerCompiledModelProperties(), and it keeps _supportedProperties in sync.
     *
     * Thread-safe: acquires _mutex internally, so it may be called at any time after construction.
     *
     * @tparam PropName  OpenVINO property object type exposing a .name() accessor.
     * @tparam Func      Callable with signature compatible with ov::Any(const Config&).
     * @param  propName    The OV property whose .name() identifies the option key.
     * @param  visibility  True to advertise the property publicly; false for private/internal.
     * @param  mutability  Desired mutability (RO / RW / WO).
     * @param  retFunc     Callback invoked on each property read; receives the current Config.
     */
    template <typename PropName, typename Func>
    void registerExternalProperty(const PropName& propName,
                                  bool visibility,
                                  ov::PropertyMutability mutability,
                                  Func&& retFunc) {
        const std::string name{propName.name()};
        std::lock_guard<std::mutex> lock(_mutex);
        _properties.insert_or_assign(name, std::make_tuple(visibility, mutability, std::forward<Func>(retFunc)));
        // Keep _supportedProperties consistent: remove old entry (if any), re-add if public
        _supportedProperties.erase(std::remove_if(_supportedProperties.begin(),
                                                  _supportedProperties.end(),
                                                  [&name](const ov::PropertyName& pn) {
                                                      return static_cast<const std::string&>(pn) == name;
                                                  }),
                                   _supportedProperties.end());
        if (visibility) {
            _supportedProperties.emplace_back(ov::PropertyName(name, mutability));
        }
    }

private:
    struct CopyState {
        PropertiesType pType;
        FilteredConfig config;
        std::shared_ptr<Metrics> metrics;
        ov::SoPtr<IEngineBackend> backend;
        Logger logger;
        ov::intel_npu::CompilerType currentlyUsedCompiler;
        std::string currentlyUsedPlatform;
        bool compilerConfigsFilteredByCompiler;
        bool compatibilityCheckFiltered;
        std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>>
            properties;
        std::vector<ov::PropertyName> supportedProperties;
    };

    explicit Properties(CopyState&& state);

    PropertiesType _pType;
    FilteredConfig _config;
    std::shared_ptr<Metrics> _metrics;
    ov::SoPtr<IEngineBackend> _backend;
    Logger _logger;

    ov::intel_npu::CompilerType _currentlyUsedCompiler = ov::intel_npu::CompilerType::PREFER_PLUGIN;
    ov::intel_npu::CompilerType _compilerForCompatibilityCheck = ov::intel_npu::CompilerType::DRIVER;
    std::string _currentlyUsedPlatform;

    // Boolean to check whether properties were filtered with compiler supported properties
    bool _compilerConfigsFilteredByCompiler = false;
    // Boolean to signal that compatibility check was already filtered by compiler support
    bool _compatibilityCheckFiltered = false;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;

    // The compatibility_check property is supported only in case at least one of the compilers (CID or CIP) supports it
    // To avoid loading the compiler library and check the support when the property is registered, the check can
    // be performed at a later stage, when the property is actually queried.
    bool disable_compatibility_check_if_needed();

    /**
     * @brief Checks whether a property was registered by its name
     */
    bool isPropertyRegistered(const std::string& propertyName) const;

    /**
     * @brief Registers a config-backed property using a plain config.get<OPT_TYPE>() callback.
     *
     * Can be used for any property that has an entry in the OptionsDesc table and requires
     * no value manipulation beyond a straight config.get<> read.
     *
     * @tparam OPT_TYPE  Config option type (e.g. PERF_COUNT) whose value is read by
     *                   config.get<OPT_TYPE>().
     * @tparam PropName  OpenVINO property object type exposing a .name() accessor.
     * @param  propName  The OV property whose .name() identifies the option key.
     *
     * @details
     * - If the option is not present in the global config (filtered out as unsupported),
     *   registration is skipped silently.
     * - Visibility (public/private) and mutability (RO/RW) are read from the optionBase
     *   descriptor associated with the option.
     * - For COMPILED_MODEL, mutability is unconditionally forced to RO regardless of the
     *   descriptor value.
     *
     * @note For COMPILED_MODEL, prefer tryRegisterCompiledModelPropertyIfSet() when the
     *       property should only appear if it was explicitly set before compilation.
     */
    template <typename OPT_TYPE, typename PropName>
    void tryRegisterSimpleProperty(const PropName& propName) {
        const std::string o_name{propName.name()};
        if (!_config.isAvailable(o_name)) {
            return;
        }
        const bool isPublic = _config.getOpt(o_name).isPublic();
        const ov::PropertyMutability isMutable = (_pType == PropertiesType::COMPILED_MODEL)
                                                     ? ov::PropertyMutability::RO
                                                     : _config.getOpt(o_name).mutability();
        _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, [](const Config& config) {
                                return config.get<OPT_TYPE>();
                            }));
    }

    /**
     * @brief Like tryRegisterSimpleProperty but derives the option key from OPT_TYPE::key()
     *        instead of an OV property object.
     *
     * Used for NPUW options whose key is declared as a static constexpr member rather than
     * through an OV property object — they cannot use tryRegisterSimpleProperty directly.
     *
     * @tparam OPT_TYPE  Config option type exposing a static key() accessor.
     *
     * @details
     * - Visibility, mutability, and availability checks follow the same rules as
     *   tryRegisterSimpleProperty.
     * - For COMPILED_MODEL, mutability is unconditionally forced to RO.
     */
    template <typename OPT_TYPE>
    void tryRegisterNpuwOptionProperty() {
        const std::string o_name{OPT_TYPE::key()};
        if (!_config.isAvailable(o_name)) {
            return;
        }
        const bool isPublic = _config.getOpt(o_name).isPublic();
        const ov::PropertyMutability isMutable = (_pType == PropertiesType::COMPILED_MODEL)
                                                     ? ov::PropertyMutability::RO
                                                     : _config.getOpt(o_name).mutability();
        _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, [](const Config& config) {
                                return config.get<OPT_TYPE>();
                            }));
    }

    /**
     * @brief Like tryRegisterSimpleProperty but skips registration when the option has not
     *        been explicitly set (i.e. is still at its default value).
     *
     * Intended exclusively for COMPILED_MODEL properties to avoid reporting settings that
     * were never passed to the compiler.  Advertising a default value can be misleading —
     * the default may not have been honoured by the compiler at all, or may be out of sync
     * with the compiler's own internal default.
     *
     * For COMPILED_MODEL, visibility is also unconditionally forced to PUBLIC (in addition
     * to the standard RO-mutability override that all compiled-model registrations apply).
     *
     * @tparam OPT_TYPE  Config option type whose value is retrieved by config.get<OPT_TYPE>().
     * @tparam PropName  OpenVINO property object type exposing a .name() accessor.
     * @param  propName  The OV property whose .name() identifies the option key.
     *
     * @details
     * - First checks whether the option has a previously-set value (_config.has()). If not,
     *   registration is skipped entirely — the property will not appear in supported_properties.
     * - Then checks availability in the global config. If unavailable, registration is skipped.
     * - For COMPILED_MODEL: mutability → RO, visibility → PUBLIC (always).
     *
     * @note **TO BE USED FOR COMPILED_MODEL ONLY** — unconditionally forces the property public.
     */
    template <typename OPT_TYPE, typename PropName>
    void tryRegisterCompiledModelPropertyIfSet(const PropName& propName) {
        const std::string o_name{propName.name()};
        if (!_config.has(o_name)) {
            return;
        }
        if (!_config.isAvailable(o_name)) {
            return;
        }
        bool isPublic = _config.getOpt(o_name).isPublic();
        ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();
        if (_pType == PropertiesType::COMPILED_MODEL) {
            isMutable = ov::PropertyMutability::RO;
            isPublic = true;
        }
        _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, [](const Config& config) {
                                return config.get<OPT_TYPE>();
                            }));
    }

    /**
     * @brief Like tryRegisterSimpleProperty but overrides visibility with the caller-supplied value.
     *
     * Provides the same functionality as tryRegisterSimpleProperty with the additional ability
     * to enforce a specific public/private visibility at registration time, rather than reading
     * it from the option descriptor.  Useful when the runtime context (e.g. a hardware capability
     * check) determines whether the property should be publicly advertised.
     *
     * Mutability is still read from the option descriptor.
     * For COMPILED_MODEL, mutability is unconditionally forced to RO.
     *
     * @tparam OPT_TYPE   Config option type whose value is retrieved by config.get<OPT_TYPE>().
     * @tparam PropName   OpenVINO property object type exposing a .name() accessor.
     * @param  propName   The OV property whose .name() identifies the option key.
     * @param  visibility True to advertise the property publicly; false for private.
     *
     * @see tryRegisterSimpleProperty
     */
    template <typename OPT_TYPE, typename PropName>
    void tryRegisterVarpubProperty(const PropName& propName, bool visibility) {
        const std::string o_name{propName.name()};
        if (!_config.isAvailable(o_name)) {
            return;
        }
        const ov::PropertyMutability isMutable = (_pType == PropertiesType::COMPILED_MODEL)
                                                     ? ov::PropertyMutability::RO
                                                     : _config.getOpt(o_name).mutability();
        _properties.emplace(o_name, std::make_tuple(visibility, isMutable, [](const Config& config) {
                                return config.get<OPT_TYPE>();
                            }));
    }

    /**
     * @brief Like tryRegisterSimpleProperty but accepts a caller-supplied callback instead of
     *        the auto-generated config.get<> one.
     *
     * Use when the property's value requires additional logic beyond a direct config read —
     * for example, a fallback to a metrics query when the config entry has not been set.
     *
     * Visibility and mutability are still read from the option descriptor.
     * For COMPILED_MODEL, mutability is unconditionally forced to RO.
     *
     * @tparam PropName  OpenVINO property object type exposing a .name() accessor.
     * @tparam Func      Callable with signature compatible with ov::Any(const Config&).
     * @param  propName  The OV property whose .name() identifies the option key.
     * @param  retFunc   Callback invoked on each property read; receives the current Config.
     *
     * @details
     * - If the option is not present in the global config, registration is skipped.
     * - Visibility and mutability are derived from the optionBase descriptor, with COMPILED_MODEL
     *   mutability forced to RO.
     *
     * @see tryRegisterSimpleProperty
     */
    template <typename PropName, typename Func>
    void tryRegisterCustomFuncProperty(const PropName& propName, Func&& retFunc) {
        const std::string o_name{propName.name()};
        if (!_config.isAvailable(o_name)) {
            return;
        }
        const bool isPublic = _config.getOpt(o_name).isPublic();
        const ov::PropertyMutability isMutable = (_pType == PropertiesType::COMPILED_MODEL)
                                                     ? ov::PropertyMutability::RO
                                                     : _config.getOpt(o_name).mutability();
        _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, std::forward<Func>(retFunc)));
    }

    /**
     * @brief Registers a fully custom property with all attributes supplied by the caller.
     *
     * Unlike the other registration helpers, this function takes explicit visibility,
     * mutability, and callback — nothing is derived from the option descriptor.
     * It only performs an availability check against the global config; no
     * COMPILED_MODEL-specific overrides (force-RO, force-public, check-if-set) are applied.
     *
     * @tparam PropName    OpenVINO property object type exposing a .name() accessor.
     * @tparam Func        Callable with signature compatible with ov::Any(const Config&).
     * @param  propName    The OV property whose .name() identifies the option key.
     * @param  visibility  True to advertise the property publicly; false for private.
     * @param  mutability  Desired mutability (RO / RW / WO).
     * @param  retFunc     Callback invoked on each property read; receives the current Config.
     *
     * @details
     * - If the option is not present in the global config, registration is skipped.
     *
     * @note Does not enforce RO or PUBLIC for COMPILED_MODEL — use only when the standard
     *       compiled-model overrides are explicitly undesirable.
     */
    template <typename PropName, typename Func>
    void tryRegisterCustomProperty(const PropName& propName,
                                   bool visibility,
                                   ov::PropertyMutability mutability,
                                   Func&& retFunc) {
        const std::string o_name{propName.name()};
        if (!_config.isAvailable(o_name)) {
            return;
        }
        _properties.emplace(o_name, std::make_tuple(visibility, mutability, std::forward<Func>(retFunc)));
    }

    /**
     * @brief Unconditionally registers a fully custom property — no availability check is
     *        performed.
     *
     * Same as tryRegisterCustomProperty but skips the _config.isAvailable() guard entirely.
     * Use only when the property must always be advertised regardless of whether the underlying
     * option was registered in the global config (e.g. ov::hint::model, which is a framework
     * contract rather than a config-backed option).
     *
     * @tparam PropName    OpenVINO property object type exposing a .name() accessor.
     * @tparam Func        Callable with signature compatible with ov::Any(const Config&).
     * @param  propName    The OV property whose .name() identifies the option key.
     * @param  visibility  True to advertise the property publicly; false for private.
     * @param  mutability  Desired mutability (RO / RW / WO).
     * @param  retFunc     Callback invoked on each property read; receives the current Config.
     *
     * @note No availability, COMPILED_MODEL, or any other checks are performed.
     */
    template <typename PropName, typename Func>
    void forceRegisterCustomProperty(const PropName& propName,
                                     bool visibility,
                                     ov::PropertyMutability mutability,
                                     Func&& retFunc) {
        _properties.emplace(propName.name(), std::make_tuple(visibility, mutability, std::forward<Func>(retFunc)));
    }

    /**
     * @brief Registers a read-only metric property backed by a caller-supplied callable.
     *
     * Metrics differ from config-backed properties: they have no entry in the config map
     * and no OptionBase descriptor.  They represent static, read-only characteristics of
     * the device, plugin, or environment (e.g. device name, driver version, available devices).
     *
     * This function supersedes both the former REGISTER_SIMPLE_METRIC (which took a plain
     * value expression evaluated lazily by a capturing lambda) and REGISTER_CUSTOM_METRIC
     * (which took a full lambda).  Both patterns are now expressed as a callable argument:
     * pass a lambda that returns a pre-computed value for the "simple" case, or one that
     * queries a live source (metrics object, backend, etc.) for the "custom" case.
     *
     * No config availability check is performed — metrics are unconditionally registered.
     * Mutability is always RO.
     *
     * @tparam PropName  OpenVINO property object type exposing a .name() accessor.
     * @tparam Func      Callable with signature compatible with ov::Any(const Config&).
     * @param  propName  The OV property whose .name() identifies the metric.
     * @param  visibility True to advertise the metric publicly; false for private/internal.
     * @param  retFunc   Callback invoked on each property read; receives the current Config.
     *
     * @note No compiled-model-specific checks are applied.
     */
    template <typename PropName, typename Func>
    void registerMetric(const PropName& propName, bool visibility, Func&& retFunc) {
        _properties.emplace(propName.name(),
                            std::make_tuple(visibility, ov::PropertyMutability::RO, std::forward<Func>(retFunc)));
    }

    // internal registration functions based on client object
    /**
     * @brief Initialize the properties map and try registering the properties for npu-plugin and compiled-model
     * Can be used for both plugin and compiled-model properties maps, based on the provided pType param to the
     * constructor of this object
     * @details
     * - it will reset the properties map
     * - it will try registering config-backed option-based properties, with data from global configuration (supported,
     * visibilty, mutability, value)
     * - if an option is not present in the global config, it assumes it is not supported and will skip it
     * - it will register metric-based properties, with data from the metrics interface
     * - at the end it populates supported_properties with the now dynamically registered public properties
     */
    void registerProperties();
    void registerPluginProperties();
    void registerCompiledModelProperties();

    const std::vector<ov::PropertyName> _cachingProperties = [] {
        std::vector<ov::PropertyName> properties = {
            ov::cache_mode.name(),
            ov::enable_profiling.name(),
            ov::device::architecture.name(),
            ov::hint::execution_mode.name(),
            ov::hint::inference_precision.name(),
            ov::hint::performance_mode.name(),
            ov::intel_npu::batch_compiler_mode_settings.name(),
            ov::intel_npu::batch_mode.name(),
            ov::intel_npu::compilation_mode.name(),
            ov::intel_npu::compilation_mode_params.name(),
            ov::intel_npu::compiler_dynamic_quantization.name(),
            ov::intel_npu::compiler_type.name(),
            ov::intel_npu::dma_engines.name(),
            ov::intel_npu::driver_version.name(),
            ov::intel_npu::dynamic_shape_to_static.name(),
            ov::intel_npu::enable_strides_for.name(),
            ov::intel_npu::max_tiles.name(),
            ov::intel_npu::stepping.name(),
            ov::intel_npu::tiles.name(),
            ov::intel_npu::turbo.name(),
            ov::intel_npu::qdq_optimization.name(),
            ov::intel_npu::qdq_optimization_aggressive.name(),
        };
        for_each_cached_npuw_option([&](auto tag) {
            using Opt = typename decltype(tag)::type;
            properties.emplace_back(std::string{Opt::key()});
        });
        return properties;
    }();

    const std::vector<ov::PropertyName> _internalSupportedProperties = {ov::internal::caching_properties.name(),
                                                                        ov::internal::caching_with_mmap.name(),
                                                                        ov::internal::cache_header_alignment.name()};

    mutable std::mutex _mutex;
};

}  // namespace intel_npu
