// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "offline_test_helpers.hpp"

#include <string>
#include <utility>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {
namespace test {

namespace {

class FakeGraph final : public IGraph {
public:
    FakeGraph() {
        _metadata.name = "FakeCompiledGraph";
    }

    std::optional<bool> is_profiling_blob() const override {
        return std::nullopt;
    }

    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const override {
        static constexpr std::string_view fakeBlob = "FAKE_VCL_BLOB";
        stream.write(fakeBlob.data(), static_cast<std::streamsize>(fakeBlob.size()));
        return {fakeBlob.size(), std::nullopt};
    }

    std::optional<std::string_view> get_compatibility_descriptor() const override {
        static constexpr std::string_view fakeDescriptor = "FAKE_COMPATIBILITY_DESCRIPTOR";
        return fakeDescriptor;
    }

    const NetworkMetadata& get_metadata() const override {
        return _metadata;
    }

protected:
    void initialize_impl(const FilteredConfig&) override {
        _init_completed = true;
    }

private:
    NetworkMetadata _metadata;
};

class FakeCompilerAdapter final : public ICompilerAdapter {
public:
    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>&, const FilteredConfig&) const override {
        return std::make_shared<FakeGraph>();
    }

    std::shared_ptr<IGraph> compileWS(std::shared_ptr<ov::Model>&&, const FilteredConfig&) const override {
        OPENVINO_THROW("compileWS not implemented in FakeCompilerAdapter");
    }

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>&, const FilteredConfig&) const override {
        return {};
    }

    uint32_t get_version() const override {
        return 1u;
    }

    std::vector<std::string> get_supported_options() const override {
        return {};
    }

    bool is_option_supported(const std::string&, const std::optional<std::string>&) const override {
        return true;
    }
};

}  // namespace

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

// Uses a FakeCompilerAdapter instead of the real PluginCompilerAdapter/VCL path, so this is a
// genuine unit test dependency, not a real (if offline) compile - no driver, no external compiler
// library, and no version-compatibility quirks to work around
std::shared_ptr<IGraph> compileOffline(const std::shared_ptr<ov::Model>& model, FilteredConfig& config) {
    FakeCompilerAdapter compiler;
    config.update({{ov::intel_npu::compiler_version.name(), std::to_string(compiler.get_version())}});
    return compiler.compile(model, config);
}

}  // namespace test
}  // namespace intel_npu
