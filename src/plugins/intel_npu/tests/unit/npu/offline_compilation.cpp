// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "compiler_option_support_helper.hpp"
#include "include/compiled_model.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "offline_test_helpers.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin_property_manager.hpp"
#include "zero_backend.hpp"

using namespace ov::intel_npu;
using namespace intel_npu;

namespace {

// Shared fixture plumbing for both suites below: constructs PluginPropertyManager/CompiledModel
// directly (no ov::Core/Plugin), always with a null backend so the offline path is forced
// deterministically regardless of what hardware the test happens to run on
class OfflineCompilationTestBase : public ::testing::TestWithParam<ov::AnyMap> {
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
        auto options = std::make_shared<OptionsDesc>();
        FilteredConfig config(options);
        test::registerOfflineOptions(*options, config);

        auto compilerOptionSupportHelper =
            std::make_shared<CompilerOptionSupportHelper>(backend, CompilerAdapterFactory());
        propertiesManager =
            std::make_unique<PluginPropertyManager>(config, backend, compilerOptionSupportHelper, logger);
        plugin = std::make_shared<test::TestPlugin>();
    }

    std::shared_ptr<CompiledModel> compile(const std::shared_ptr<ov::Model>& model, FilteredConfig compileConfig) {
        auto graph = test::compileOffline(model, compileConfig);
        return std::make_shared<CompiledModel>(model, plugin, nullptr, graph, compileConfig, std::nullopt);
    }

    Logger logger{"OfflineCompilationUnitTests"};
    ov::SoPtr<IEngineBackend> backend{nullptr};
    std::unique_ptr<PluginPropertyManager> propertiesManager;
    std::shared_ptr<test::TestPlugin> plugin;
};

using OfflineCompilationUnitTests = OfflineCompilationTestBase;

TEST_P(OfflineCompilationUnitTests, CompileWithCiPWhenDriverNotInstalled) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(compile(model, propertiesManager->getConfigForSpecificCompiler(GetParam())));
}

TEST_P(OfflineCompilationUnitTests, ExpectThrowWhenCreateInferRequestWhenDriverNotInstalled) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    std::shared_ptr<CompiledModel> compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = compile(model, propertiesManager->getConfigForSpecificCompiler(GetParam())));
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel->create_infer_request(),
                                  ov::Exception,
                                  "No available devices. Failed to create infer request!");
}

TEST_P(OfflineCompilationUnitTests, ReadSupportedPropertiesRuntimeRequirementsPresent) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    std::shared_ptr<CompiledModel> compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = compile(model, propertiesManager->getConfigForSpecificCompiler(GetParam())));
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(
        supportedProperties =
            compiledModel->get_property(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>());
    ASSERT_TRUE(std::find(supportedProperties.begin(), supportedProperties.end(), ov::runtime_requirements.name()) !=
                supportedProperties.end());
}

TEST_P(OfflineCompilationUnitTests, ReadRuntimeRequirementsOffline) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    std::shared_ptr<CompiledModel> compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = compile(model, propertiesManager->getConfigForSpecificCompiler(GetParam())));
    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiledModel->get_property(ov::runtime_requirements.name()).as<std::string>());
    ASSERT_FALSE(requirements.empty());
}

INSTANTIATE_TEST_SUITE_P(
    OfflineCompilationPlatforms,
    OfflineCompilationUnitTests,
    ::testing::Values(ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5010},
                                 {ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::PLUGIN}},
                      ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5020},
                                 {ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::PLUGIN}}),
    OfflineCompilationUnitTests::getTestCaseName);

// Tests that only exercise PluginPropertyManager's persisted set_property()/get_property() surface,
// its params stay compiler_type-free, since NPU_COMPILER_TYPE isn't enabled until a compiler is
// actually resolved, and set_property() (unlike getConfigForSpecificCompiler()) has no such
// resolution step of its own.
using OfflinePluginPropertyUnitTests = OfflineCompilationTestBase;

TEST_P(OfflinePluginPropertyUnitTests, CompileWithCiPWhenDriverNotInstalledSetProperty) {
    propertiesManager->setProperty(GetParam());
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(compile(model, propertiesManager->getConfig()));
}

TEST_P(OfflinePluginPropertyUnitTests, ReadMaxTilesAndExpectThrow) {
    propertiesManager->setProperty(GetParam());
    OV_EXPECT_THROW_HAS_SUBSTRING(propertiesManager->getProperty(ov::intel_npu::max_tiles.name()),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_P(OfflinePluginPropertyUnitTests, ReadSupportedPropertiesMaxTilesNotPresent) {
    propertiesManager->setProperty(GetParam());
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(
        supportedProperties =
            propertiesManager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>());
    ASSERT_TRUE(std::find(supportedProperties.begin(), supportedProperties.end(), ov::intel_npu::max_tiles.name()) ==
                supportedProperties.end());
}

TEST_P(OfflinePluginPropertyUnitTests, CompatibilityCheckNotSupportedOffline) {
    propertiesManager->setProperty(GetParam());
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(
        supportedProperties =
            propertiesManager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>());
    ASSERT_TRUE(std::find(supportedProperties.begin(), supportedProperties.end(), ov::compatibility_check.name()) ==
                supportedProperties.end());
}

INSTANTIATE_TEST_SUITE_P(
    OfflineCompilationPlatforms,
    OfflinePluginPropertyUnitTests,
    ::testing::Values(ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5010}},
                      ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5020}}),
    OfflinePluginPropertyUnitTests::getTestCaseName);

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
