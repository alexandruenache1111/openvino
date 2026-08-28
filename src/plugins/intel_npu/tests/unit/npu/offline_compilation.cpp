// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin_property_manager.hpp"
#include "zero_backend.hpp"

using namespace ov::intel_npu;

namespace {

// Registers the same set of options Plugin::Plugin() registers via its own (file-local) init_config,
// for the case where BackendsRegistry found no usable backend (no driver/device installed). Kept in
// sync by hand; mirrors the equivalent duplicate already used by
// tests/functional/internal/plugin/test_properties.cpp for the same reason: Plugin's option
// registration helper is file-local to plugin.cpp and cannot be called from test code.
void registerOfflineOptions(::intel_npu::OptionsDesc& options, ::intel_npu::FilteredConfig& config) {
    using namespace ::intel_npu;

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

#undef REGISTER_OPTION
}

// Unit-test double for "no backend registered" (mirrors what BackendsRegistry::getEngineBackend()
// returns when no driver/device is found), exercising PluginPropertyManager and the compiler adapter
// directly instead of going through Plugin/ov::Core, so no real driver is ever touched.
class OfflineCompilationUnitTests : public ::testing::TestWithParam<ov::AnyMap> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ParamType>& info) {
        std::string result;
        for (const auto& [key, value] : info.param) {
            result += value.as<std::string>();
        }
        return result;
    }

protected:
    void SetUp() override {
        options = std::make_shared<::intel_npu::OptionsDesc>();
        auto initialConfig = std::make_unique<::intel_npu::FilteredConfig>(options);
        registerOfflineOptions(*options, *initialConfig);

        compilerOptionSupportHelper = std::make_shared<::intel_npu::CompilerOptionSupportHelper>(
            backend,
            ::intel_npu::CompilerAdapterFactory());
        propertiesManager = std::make_unique<::intel_npu::PluginPropertyManager>(*initialConfig,
                                                                                 backend,
                                                                                 compilerOptionSupportHelper,
                                                                                 logger);

        // Applying NPU_PLATFORM through PluginPropertyManager::setProperty (rather than a raw
        // FilteredConfig::update) mirrors Plugin::set_property and is what actually enables the
        // option for the resolved compiler/platform - a bare FilteredConfig::update would reject it
        // as "not available" since registerOfflineOptions() starts every option disabled.
        propertiesManager->setProperty(GetParam());
    }

    ::intel_npu::Logger logger{"OfflineCompilationUnitTests"};
    ov::SoPtr<::intel_npu::IEngineBackend> backend{nullptr};
    std::shared_ptr<::intel_npu::OptionsDesc> options;
    std::shared_ptr<::intel_npu::CompilerOptionSupportHelper> compilerOptionSupportHelper;
    std::unique_ptr<::intel_npu::PluginPropertyManager> propertiesManager;
};

TEST_P(OfflineCompilationUnitTests, ReadMaxTilesAndExpectThrow) {
    OV_EXPECT_THROW_HAS_SUBSTRING(propertiesManager->getProperty(ov::intel_npu::max_tiles.name()),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_P(OfflineCompilationUnitTests, ReadSupportedPropertiesMaxTilesNotPresent) {
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(supportedProperties =
                           propertiesManager->getProperty(ov::supported_properties.name())
                               .as<std::vector<ov::PropertyName>>());
    ASSERT_TRUE(std::find(supportedProperties.begin(),
                          supportedProperties.end(),
                          ov::intel_npu::max_tiles.name()) == supportedProperties.end());
}

TEST_P(OfflineCompilationUnitTests, CompatibilityCheckNotSupportedOffline) {
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(supportedProperties =
                           propertiesManager->getProperty(ov::supported_properties.name())
                               .as<std::vector<ov::PropertyName>>());
    ASSERT_TRUE(std::find(supportedProperties.begin(),
                          supportedProperties.end(),
                          ov::compatibility_check.name()) == supportedProperties.end());
}

// Exercises the real PluginCompilerAdapter/VCL compiler path with no device involved at all - the
// core claim behind "offline compilation".
TEST_P(OfflineCompilationUnitTests, CompilesOfflineViaPluginCompiler) {
    auto compilerType = ov::intel_npu::CompilerType::PREFER_PLUGIN;
    const auto& resolvedConfig = propertiesManager->getConfig();
    const std::string platform = resolvedConfig.get<::intel_npu::PLATFORM>();

    ::intel_npu::CompilerAdapterFactory factory;
    std::unique_ptr<::intel_npu::ICompilerAdapter> compiler;
    OV_ASSERT_NO_THROW(
        compiler = factory.getCompiler(backend, compilerType, platform, compilerOptionSupportHelper->getOptionSupportCache()));
    ASSERT_EQ(compilerType, ov::intel_npu::CompilerType::PLUGIN);

    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    std::shared_ptr<::intel_npu::IGraph> graph;
    OV_ASSERT_NO_THROW(graph = compiler->compile(model, resolvedConfig));
    ASSERT_NE(graph, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    OfflineCompilationPlatforms,
    OfflineCompilationUnitTests,
    ::testing::Values(ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5010}},
                      ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5020}}),
    OfflineCompilationUnitTests::getTestCaseName);

using UnavailableDeviceTests = ::testing::Test;

TEST_F(UnavailableDeviceTests, GetDeviceNotAvailable) {
    std::shared_ptr<intel_npu::ZeroEngineBackend> backend;
    ASSERT_ANY_THROW(backend = std::make_shared<intel_npu::ZeroEngineBackend>());
}

}  // namespace

