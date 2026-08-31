// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "offline_test_helpers.hpp"

#include <string>
#include <utility>

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {
namespace test {

void registerOfflineOptions(OptionsDesc& options, FilteredConfig& config) {
    options.reset();

#define REGISTER_OPTION(OPT_TYPE)                             \
    do {                                                      \
        auto dummyopt = details::makeOptionModel<OPT_TYPE>(); \
        std::string o_name = dummyopt.key().data();           \
        options.add<OPT_TYPE>();                              \
        config.enable(std::move(o_name), false);              \
    } while (0)

    REGISTER_OPTION(LOG_LEVEL);
    REGISTER_OPTION(COMPILE_LOG_LEVEL);
    REGISTER_OPTION(CACHE_DIR);
    REGISTER_OPTION(CACHE_MODE);
    REGISTER_OPTION(COMPILED_BLOB);
    REGISTER_OPTION(DEVICE_ID);
    REGISTER_OPTION(NUM_STREAMS);
    REGISTER_OPTION(PERF_COUNT);
    REGISTER_OPTION(LOADED_FROM_CACHE);
    REGISTER_OPTION(COMPILATION_NUM_THREADS);
    REGISTER_OPTION(PERFORMANCE_HINT);
    REGISTER_OPTION(EXECUTION_MODE_HINT);
    REGISTER_OPTION(PERFORMANCE_HINT_NUM_REQUESTS);
    OPENVINO_SUPPRESS_DEPRECATED_START
    REGISTER_OPTION(ENABLE_CPU_PINNING);
    OPENVINO_SUPPRESS_DEPRECATED_END
    REGISTER_OPTION(INFERENCE_PRECISION_HINT);
    REGISTER_OPTION(MODEL_PRIORITY);
    REGISTER_OPTION(COMPILATION_MODE_PARAMS);
    REGISTER_OPTION(DMA_ENGINES);
    REGISTER_OPTION(TILES);
    REGISTER_OPTION(COMPILATION_MODE);
    REGISTER_OPTION(COMPILER_TYPE);
    REGISTER_OPTION(COMPILER_VERSION);
    REGISTER_OPTION(PLATFORM);
    REGISTER_OPTION(CREATE_EXECUTOR);
    REGISTER_OPTION(DYNAMIC_SHAPE_TO_STATIC);
    REGISTER_OPTION(PROFILING_TYPE);
    REGISTER_OPTION(BACKEND_COMPILATION_PARAMS);
    REGISTER_OPTION(BATCH_MODE);
    REGISTER_OPTION(BYPASS_UMD_CACHING);
    REGISTER_OPTION(DEFER_WEIGHTS_LOAD);
    REGISTER_OPTION(WEIGHTS_PATH);
    REGISTER_OPTION(RUN_INFERENCES_SEQUENTIALLY);
    REGISTER_OPTION(COMPILER_DYNAMIC_QUANTIZATION);
    REGISTER_OPTION(QDQ_OPTIMIZATION);
    REGISTER_OPTION(QDQ_OPTIMIZATION_AGGRESSIVE);
    REGISTER_OPTION(STEPPING);
    REGISTER_OPTION(DISABLE_VERSION_CHECK);
    REGISTER_OPTION(EXPORT_RAW_BLOB);
    REGISTER_OPTION(IMPORT_RAW_BLOB);
    REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
    REGISTER_OPTION(TURBO);
    REGISTER_OPTION(ENABLE_WEIGHTLESS);
    REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
    REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);
    REGISTER_OPTION(MODEL_SERIALIZER_VERSION);
    REGISTER_OPTION(ENABLE_STRIDES_FOR);
    REGISTER_OPTION(SHARED_COMMON_QUEUE);
    REGISTER_OPTION(CACHE_ENCRYPTION_CALLBACKS);
    REGISTER_OPTION(RUNTIME_REQUIREMENTS);
    REGISTER_OPTION(COMPATIBILITY_CHECK);

    // No backend => MAX_TILES / WORKLOAD_TYPE / DISABLE_IDLE_MEMORY_PRUNING stay unregistered here,
    // exactly like Plugin's init_config() does when BackendsRegistry finds no usable device.

    config.parseEnvVars();

    for_each_exposed_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        REGISTER_OPTION(Opt);
    });

    config.enableRuntimeOptions();
    config.enable(ov::log::level.name(), true);
    config.enable(ov::hint::performance_mode.name(), true);
    config.enable(ov::enable_profiling.name(), true);
    // Normally a real compiler-support probe (PluginPropertyManager::setProperty) enables these for
    // the resolved compiler; tests set them directly on the config, so enable them explicitly here.
    config.enable(ov::intel_npu::platform.name(), true);
    config.enable(ov::intel_npu::compiler_version.name(), true);

#undef REGISTER_OPTION
}

std::shared_ptr<IGraph> compileOffline(const std::shared_ptr<ov::Model>& model, FilteredConfig& config) {
    ov::SoPtr<IEngineBackend> backend{nullptr};
    auto compilerType = ov::intel_npu::CompilerType::PREFER_PLUGIN;
    CompilerAdapterFactory factory;
    auto compiler = factory.getCompiler(backend, compilerType, config.get<PLATFORM>());
    config.update({{ov::intel_npu::compiler_version.name(), std::to_string(compiler->get_version())}});

    // copied behavior from functiona/internal/backend/zero_infer_request_tests.cpp
    if (compiler->is_option_supported(MODEL_SERIALIZER_VERSION::key().data())) {
        config.enable(MODEL_SERIALIZER_VERSION::key().data(), true);
        config.update({{MODEL_SERIALIZER_VERSION::key().data(),
                        MODEL_SERIALIZER_VERSION::toString(ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY)}});
    } else {
        config.enable(MODEL_SERIALIZER_VERSION::key().data(), false);
    }

    return compiler->compile(model, config);
}

}  // namespace test
}  // namespace intel_npu
