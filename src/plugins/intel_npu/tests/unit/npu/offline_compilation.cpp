// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin_property_manager.hpp"
#include "zero_backend.hpp"

using namespace ov::intel_npu;
using namespace intel_npu;

namespace {

// Registers the same option set Plugin::Plugin() registers via its own (file-local) register_options(),
// for the "no backend" case (BackendsRegistry found no usable driver/device). Kept in sync by hand and
// deliberately NOT shared with other unit test fixtures (e.g. compiled_model_test.cpp's
// offline_test_helpers.hpp): this suite exercises the offline compilation path in isolation, so it must
// not depend on plumbing that a change to a CompiledModel-focused fixture could silently break.
void registerOfflineOptions(OptionsDesc& options) {
    options.reset();

#define REGISTER_OPTION(OPT_TYPE) options.add<OPT_TYPE>()

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
    REGISTER_OPTION(INFERENCE_PRECISION_HINT);
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
    REGISTER_OPTION(ALLOW_BYTECODE);
    REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
    REGISTER_OPTION(TURBO);
    REGISTER_OPTION(ENABLE_WEIGHTLESS);
    REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
    REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);
    REGISTER_OPTION(MODEL_SERIALIZER_VERSION);
    REGISTER_OPTION(ENABLE_STRIDES_FOR);
    REGISTER_OPTION(SHARED_COMMON_QUEUE);
    REGISTER_OPTION(CACHE_ENCRYPTION_CALLBACKS);
    REGISTER_OPTION(MAX_TILES);
    REGISTER_OPTION(MODEL_PTR);
    REGISTER_OPTION(DISABLE_IDLE_MEMORY_PRUNING);

    // No backend => MODEL_PRIORITY / WORKLOAD_TYPE stay unregistered here, exactly like
    // Plugin's register_options() does when BackendsRegistry finds no usable device.

    OPENVINO_SUPPRESS_DEPRECATED_START
    REGISTER_OPTION(ENABLE_CPU_PINNING);
    OPENVINO_SUPPRESS_DEPRECATED_END

    for_each_exposed_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        REGISTER_OPTION(Opt);
    });

#undef REGISTER_OPTION
}

// Exercises PluginPropertyManager and CompilerAdapterFactory/PluginCompilerAdapter directly, always
// with a null backend (mirrors what BackendsRegistry::getEngineBackend() returns when no driver/device
// is found), so the offline compilation path is forced deterministically with no ov::Core/Plugin and
// no real NPU driver ever touched.
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
        options = std::make_shared<OptionsDesc>();
        registerOfflineOptions(*options);

        compilerOptionSupportHelper = std::make_shared<CompilerOptionSupportHelper>(backend, CompilerAdapterFactory());
        propertiesManager = std::make_unique<PluginPropertyManager>(options, backend, compilerOptionSupportHelper, logger);

        // Applying NPU_PLATFORM through PluginPropertyManager::setProperty (rather than reaching into a
        // FilteredConfig directly) mirrors Plugin::set_property and is what actually resolves the plugin
        // compiler and enables the option for it.
        propertiesManager->setProperty(GetParam());
    }

    Logger logger{"OfflineCompilationUnitTests"};
    ov::SoPtr<IEngineBackend> backend{nullptr};
    std::shared_ptr<OptionsDesc> options;
    std::shared_ptr<CompilerOptionSupportHelper> compilerOptionSupportHelper;
    std::unique_ptr<PluginPropertyManager> propertiesManager;
};

TEST_P(OfflineCompilationUnitTests, ReadMaxTilesAndExpectThrow) {
    OV_EXPECT_THROW_HAS_SUBSTRING(propertiesManager->getProperty(ov::intel_npu::max_tiles.name()),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_P(OfflineCompilationUnitTests, ReadSupportedPropertiesMaxTilesNotPresent) {
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(
        supportedProperties =
            propertiesManager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>());
    ASSERT_TRUE(std::find(supportedProperties.begin(), supportedProperties.end(), ov::intel_npu::max_tiles.name()) ==
                supportedProperties.end());
}

TEST_P(OfflineCompilationUnitTests, CompatibilityCheckNotSupportedOffline) {
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(
        supportedProperties =
            propertiesManager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>());
    ASSERT_TRUE(std::find(supportedProperties.begin(), supportedProperties.end(), ov::compatibility_check.name()) ==
                supportedProperties.end());
}

// The three methods below are what Plugin::compile_model() calls to resolve device/platform/compiler
// type *before* ever handing off to CompilerAdapterFactory - the last plugin-owned decision point
// before the flow crosses into compiler/driver territory. Each is a pure read of the properties map
// with a config fallback, so none of them touch CompilerAdapterFactory or VCL.
TEST_P(OfflineCompilationUnitTests, DetermineDeviceIdReturnsPropertyOverride) {
    ASSERT_EQ(propertiesManager->determineDeviceId({{ov::device::id.name(), std::string("3")}}), "3");
}

TEST_P(OfflineCompilationUnitTests, DetermineDeviceIdFallsBackToConfigWhenNotProvided) {
    ASSERT_TRUE(propertiesManager->determineDeviceId({}).empty());
}

TEST_P(OfflineCompilationUnitTests, DeterminePlatformReturnsPropertyOverride) {
    ASSERT_EQ(propertiesManager->determinePlatform({{ov::intel_npu::platform.name(), std::string("9999")}}), "9999");
}

TEST_P(OfflineCompilationUnitTests, DeterminePlatformFallsBackToPreviouslyConfiguredValue) {
    const auto expectedPlatform = GetParam().at(ov::intel_npu::platform.name()).as<std::string>();
    ASSERT_EQ(propertiesManager->determinePlatform({}), expectedPlatform);
}

TEST_P(OfflineCompilationUnitTests, DetermineCompilerTypeReturnsPropertyOverride) {
    ASSERT_EQ(propertiesManager->determineCompilerType(
                  {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}}),
              ov::intel_npu::CompilerType::DRIVER);
}

TEST_P(OfflineCompilationUnitTests, DetermineCompilerTypeFallsBackToConfigDefault) {
    // Fixture never sets NPU_COMPILER_TYPE, so this is the option's own default (PREFER_PLUGIN).
    ASSERT_EQ(propertiesManager->determineCompilerType({}), ov::intel_npu::CompilerType::PREFER_PLUGIN);
}

// CompilerAdapterFactory rejects DRIVER compilation without a backend before touching the VCL compiler.
TEST_P(OfflineCompilationUnitTests, GetCompilerDriverTypeWithNullBackendThrows) {
    CompilerAdapterFactory factory;
    auto compilerType = ov::intel_npu::CompilerType::DRIVER;
    OV_EXPECT_THROW_HAS_SUBSTRING(
        factory.getCompiler(backend,
                            compilerType,
                            GetParam().at(ov::intel_npu::platform.name()).as<std::string>(),
                            compilerOptionSupportHelper->getOptionSupportCache()),
        ov::Exception,
        "Could not find an NPU device");
}

INSTANTIATE_TEST_SUITE_P(
    OfflineCompilationPlatforms,
    OfflineCompilationUnitTests,
    ::testing::Values(ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5010}},
                      ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5020}}),
    OfflineCompilationUnitTests::getTestCaseName);

using UnavailableDeviceTests = ::testing::Test;

// should be like this if we get rid of the driver trick in main.cpp
TEST_F(UnavailableDeviceTests, GetDeviceNotAvailable) {
    std::shared_ptr<ZeroEngineBackend> backend;
    try {
        backend = std::make_shared<ZeroEngineBackend>();
    } catch (...) {
        return;
    }
    GTEST_SKIP() << "A real NPU driver/device is present on this host; cannot exercise the no-driver path.";
}

}  // namespace
