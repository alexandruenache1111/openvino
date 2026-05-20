// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <random>
#include <thread>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "metrics.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/log.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"

using ::testing::AllOf;
using ::testing::HasSubstr;

using ConfigParams = std::tuple<std::string,   // Device name
                                std::string>;  // Config name

namespace {
/**
 * @brief Registers one config option in both an OptionsDesc and a FilteredConfig.
 *
 * Shared by all test fixtures that need to populate a synthetic config table.
 * Adds @p OPT_TYPE to the options descriptor and marks the key as disabled;
 * the caller is responsible for enabling it via cfg.enable() or
 * cfg.enableRuntimeOptions().
 */
template <typename OPT_TYPE>
void registerOptionHelper(::intel_npu::OptionsDesc& opts, ::intel_npu::FilteredConfig& cfg) {
    const auto dummyopt = ::intel_npu::details::makeOptionModel<OPT_TYPE>();
    const std::string o_name{dummyopt.key().data()};
    opts.add<OPT_TYPE>();
    cfg.enable(o_name, false);
}
}  // namespace

namespace ov {
namespace test {
namespace behavior {
class PropertiesManagerTests : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<ConfigParams> {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::FilteredConfig npu_config = ::intel_npu::FilteredConfig(options);
    ov::SoPtr<::intel_npu::IEngineBackend> backend;
    std::unique_ptr<::intel_npu::Properties> propertiesManager;

    std::string configuration;
    std::string targetDevice;

    template <typename OPT_TYPE>
    void registerOption() {
        registerOptionHelper<OPT_TYPE>(*options, npu_config);
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConfigParams>& obj) {
        std::string targetDevice;
        std::string configuration;
        std::tie(targetDevice, configuration) = obj.param;

        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "config=" << configuration << "_";

        return result.str();
    }

    void SetUp() override {
        using namespace ::intel_npu;

        std::tie(targetDevice, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        backend = ov::SoPtr<IEngineBackend>(std::make_shared<ZeroEngineBackend>());
        auto metrics = std::make_shared<Metrics>(backend);

        options->reset();

        registerOption<LOG_LEVEL>();
        registerOption<CACHE_DIR>();
        registerOption<CACHE_MODE>();
        registerOption<COMPILED_BLOB>();
        registerOption<DEVICE_ID>();
        registerOption<NUM_STREAMS>();
        registerOption<PERF_COUNT>();
        registerOption<LOADED_FROM_CACHE>();
        registerOption<COMPILATION_NUM_THREADS>();
        registerOption<PERFORMANCE_HINT>();
        registerOption<EXECUTION_MODE_HINT>();
        registerOption<PERFORMANCE_HINT_NUM_REQUESTS>();
        registerOption<ENABLE_CPU_PINNING>();
        registerOption<INFERENCE_PRECISION_HINT>();
        registerOption<MODEL_PRIORITY>();
        registerOption<EXCLUSIVE_ASYNC_REQUESTS>();
        registerOption<COMPILATION_MODE_PARAMS>();
        registerOption<DMA_ENGINES>();
        registerOption<TILES>();
        registerOption<COMPILATION_MODE>();
        registerOption<COMPILER_TYPE>();
        registerOption<PLATFORM>();
        registerOption<CREATE_EXECUTOR>();
        registerOption<DYNAMIC_SHAPE_TO_STATIC>();
        registerOption<PROFILING_TYPE>();
        registerOption<BACKEND_COMPILATION_PARAMS>();
        registerOption<BATCH_MODE>();
        registerOption<BYPASS_UMD_CACHING>();
        registerOption<DEFER_WEIGHTS_LOAD>();
        registerOption<WEIGHTS_PATH>();
        registerOption<RUN_INFERENCES_SEQUENTIALLY>();
        registerOption<COMPILER_DYNAMIC_QUANTIZATION>();
        registerOption<QDQ_OPTIMIZATION>();
        registerOption<QDQ_OPTIMIZATION_AGGRESSIVE>();
        registerOption<STEPPING>();
        registerOption<DISABLE_VERSION_CHECK>();
        registerOption<EXPORT_RAW_BLOB>();
        registerOption<IMPORT_RAW_BLOB>();
        registerOption<BATCH_COMPILER_MODE_SETTINGS>();
        registerOption<TURBO>();
        registerOption<SEPARATE_WEIGHTS_VERSION>();
        registerOption<WS_COMPILE_CALL_NUMBER>();
        registerOption<MODEL_SERIALIZER_VERSION>();
        registerOption<ENABLE_STRIDES_FOR>();
        registerOption<SHARED_COMMON_QUEUE>();

        if (backend) {
            registerOption<MAX_TILES>();

            if (backend->isCommandQueueExtSupported()) {
                registerOption<WORKLOAD_TYPE>();
            }
            if (backend->isContextExtSupported()) {
                registerOption<DISABLE_IDLE_MEMORY_PRUNING>();
            }
        }

        for_each_exposed_npuw_option([this](auto tag) {
            using Opt = typename decltype(tag)::type;
            registerOption<Opt>();
        });

        npu_config.enableRuntimeOptions();

        // Special cases - options with OptionMode::Both must be enabled for the plugin even if the compiler does not
        // support them, because they may be used by the plugin itself or by the driver.
        // We still check compiler support to decide whether these options should be removed from the config string.

        // NPU_TURBO might be supported by the driver
        if (backend && backend->isCommandQueueExtSupported()) {
            npu_config.enable(ov::intel_npu::turbo.name(), true);
        }

        // LOG_LEVEL, PERFORMANCE_HINT and PERF_COUNT are needed by runtime options
        npu_config.enable(ov::log::level.name(), true);
        npu_config.enable(ov::hint::performance_mode.name(), true);
        npu_config.enable(ov::enable_profiling.name(), true);

        if (npu_config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PREFER_PLUGIN && backend != nullptr) {
            auto device = backend->getDevice();
            if (device) {
                auto platformName = device->getName();
                CompilerAdapterFactory compilerFactory;
                auto compileType = compilerFactory.determineAppropriateCompilerTypeBasedOnPlatform(platformName);
                if (compileType == ov::intel_npu::CompilerType::DRIVER) {
                    npu_config.update({{ov::intel_npu::compiler_type.name(), COMPILER_TYPE::toString(compileType)}});
                }
            }
        }

        propertiesManager = std::make_unique<Properties>(PropertiesType::PLUGIN, npu_config, metrics, backend);
    }

    void TearDown() override {
        APIBaseTest::TearDown();
    }
};

TEST_P(PropertiesManagerTests, ExpectRunTimeSpecialBothPropertyIsSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = false;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::log::level(ov::log::Level::INFO)}});
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    ASSERT_TRUE(isSupported);
}

TEST_P(PropertiesManagerTests, ExpectArgumentIsNotSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = true;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    ov::AnyMap arguments = {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                            {"DUMMY_PROPERTY", "DUMMY_VALUE"}};

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);

        try {
            propertiesManager->setProperty(arguments);
            isSupported = true;
        } catch (...) {
            isSupported = false;
        }
    }

    ASSERT_FALSE(isSupported);
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

using ExpectLoadingCompilerPropertySupported = PropertiesManagerTests;

TEST_P(ExpectLoadingCompilerPropertySupported, ExpectCompilerPropertyIsSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = false;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_TRUE(isSupported);
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

// ============================================================================
// CompiledModelPropertiesTests
// Tests for Properties::registerExternalProperty and the compiled-model changes:
//  - ov::model_name  is NOT registered by default; registered via registerExternalProperty()
//  - ov::runtime_requirements is NOT registered by default; registered conditionally
// ============================================================================

/**
 * @brief Minimal fixture for Properties(COMPILED_MODEL) that requires no real backend.
 *
 * Only LOG_LEVEL is registered so the internal logger can be constructed.
 * All other compiled-model options are absent; the tryRegister* helpers gracefully skip them.
 * metrics and backend are passed as nullptr since registerCompiledModelProperties() does not use them.
 */
class CompiledModelPropertiesTests : public ::testing::Test {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::FilteredConfig config{options};
    std::unique_ptr<::intel_npu::Properties> propertiesManager;

    template <typename OPT_TYPE>
    void registerOption() {
        registerOptionHelper<OPT_TYPE>(*options, config);
    }

    void SetUp() override {
        registerOption<::intel_npu::LOG_LEVEL>();
        config.enableRuntimeOptions();
        config.enable(ov::log::level.name(), true);
        // metrics and backend default to null — safe for COMPILED_MODEL since
        // registerCompiledModelProperties() does not use them.
        propertiesManager =
            std::make_unique<::intel_npu::Properties>(::intel_npu::PropertiesType::COMPILED_MODEL, config);
    }

    bool isSupportedProperty(const std::string& name) {
        const auto supported =
            propertiesManager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
        return std::any_of(supported.begin(), supported.end(), [&name](const ov::PropertyName& pn) {
            return static_cast<std::string>(pn) == name;
        });
    }
};

// ---- ov::model_name ---------------------------------------------------------
// ov::model_name was removed from registerCompiledModelProperties().
// CompiledModel::CompiledModel() calls registerExternalProperty() to provide the real graph name.

TEST_F(CompiledModelPropertiesTests, ModelNameNotRegisteredByDefault_GetPropertyThrows) {
    ASSERT_THROW(propertiesManager->getProperty(ov::model_name.name()), ov::Exception);
}

TEST_F(CompiledModelPropertiesTests, ModelNameNotRegisteredByDefault_NotInSupportedProperties) {
    ASSERT_FALSE(isSupportedProperty(ov::model_name.name()));
}

TEST_F(CompiledModelPropertiesTests, RegisterExternalModelName_GetPropertyReturnsValue) {
    const std::string expectedName = "my_test_model";
    propertiesManager->registerExternalProperty(ov::model_name,
                                                true,
                                                ov::PropertyMutability::RO,
                                                [expectedName](const ::intel_npu::Config&) -> ov::Any {
                                                    return expectedName;
                                                });

    ASSERT_EQ(propertiesManager->getProperty(ov::model_name.name()).as<std::string>(), expectedName);
}

TEST_F(CompiledModelPropertiesTests, RegisterExternalModelName_AppearsInSupportedProperties) {
    propertiesManager->registerExternalProperty(ov::model_name,
                                                true,
                                                ov::PropertyMutability::RO,
                                                [](const ::intel_npu::Config&) -> ov::Any {
                                                    return std::string("some_model");
                                                });

    ASSERT_TRUE(isSupportedProperty(ov::model_name.name()));
}

// ---- ov::runtime_requirements -----------------------------------------------
// ov::runtime_requirements was removed from registerCompiledModelProperties().
// CompiledModel::CompiledModel() only registers it when graph->get_compatibility_descriptor() has a value.

TEST_F(CompiledModelPropertiesTests, RuntimeRequirementsNotRegisteredByDefault_GetPropertyThrows) {
    ASSERT_THROW(propertiesManager->getProperty(ov::runtime_requirements.name()), ov::Exception);
}

TEST_F(CompiledModelPropertiesTests, RuntimeRequirementsNotRegisteredByDefault_NotInSupportedProperties) {
    ASSERT_FALSE(isSupportedProperty(ov::runtime_requirements.name()));
}

TEST_F(CompiledModelPropertiesTests, RegisterExternalRuntimeRequirements_GetPropertyReturnsValue) {
    const std::string expectedReqs = "platform=NPU3720;ov_version=2025.4.0";
    propertiesManager->registerExternalProperty(ov::runtime_requirements,
                                                true,
                                                ov::PropertyMutability::RO,
                                                [expectedReqs](const ::intel_npu::Config&) -> ov::Any {
                                                    return expectedReqs;
                                                });

    ASSERT_EQ(propertiesManager->getProperty(ov::runtime_requirements.name()).as<std::string>(), expectedReqs);
}

TEST_F(CompiledModelPropertiesTests, RegisterExternalRuntimeRequirements_AppearsInSupportedProperties) {
    propertiesManager->registerExternalProperty(ov::runtime_requirements,
                                                true,
                                                ov::PropertyMutability::RO,
                                                [](const ::intel_npu::Config&) -> ov::Any {
                                                    return std::string("");
                                                });

    ASSERT_TRUE(isSupportedProperty(ov::runtime_requirements.name()));
}

// ---- Conditional registration mirrors CompiledModel constructor logic --------

// When the compatibility descriptor is absent, runtime_requirements is NOT registered —
// mirrors the `if (_graph->get_compatibility_descriptor().has_value())` guard in the constructor.
TEST_F(CompiledModelPropertiesTests, RuntimeRequirements_NotRegisteredWhenCompatDescAbsent) {
    const std::optional<std::string> compatDesc = std::nullopt;
    if (compatDesc.has_value()) {
        propertiesManager->registerExternalProperty(ov::runtime_requirements,
                                                    true,
                                                    ov::PropertyMutability::RO,
                                                    [&compatDesc](const ::intel_npu::Config&) -> ov::Any {
                                                        return *compatDesc;
                                                    });
    }

    ASSERT_FALSE(isSupportedProperty(ov::runtime_requirements.name()));
    ASSERT_THROW(propertiesManager->getProperty(ov::runtime_requirements.name()), ov::Exception);
}

// When the compatibility descriptor is present, runtime_requirements IS registered.
TEST_F(CompiledModelPropertiesTests, RuntimeRequirements_RegisteredWhenCompatDescPresent) {
    const std::optional<std::string> compatDesc = std::string("platform=NPU3720;ov_version=2025.4.0");
    if (compatDesc.has_value()) {
        propertiesManager->registerExternalProperty(ov::runtime_requirements,
                                                    true,
                                                    ov::PropertyMutability::RO,
                                                    [compatDesc](const ::intel_npu::Config&) -> ov::Any {
                                                        return compatDesc.value();
                                                    });
    }

    ASSERT_TRUE(isSupportedProperty(ov::runtime_requirements.name()));
    ASSERT_EQ(propertiesManager->getProperty(ov::runtime_requirements.name()).as<std::string>(), compatDesc.value());
}

// ---- registerExternalProperty: general behaviour ----------------------------

// Calling registerExternalProperty twice must overwrite the first registration (insert_or_assign semantics).
TEST_F(CompiledModelPropertiesTests, RegisterExternalProperty_OverwritesPreviousRegistration) {
    propertiesManager->registerExternalProperty(ov::model_name,
                                                true,
                                                ov::PropertyMutability::RO,
                                                [](const ::intel_npu::Config&) -> ov::Any {
                                                    return std::string("first");
                                                });
    ASSERT_EQ(propertiesManager->getProperty(ov::model_name.name()).as<std::string>(), "first");

    propertiesManager->registerExternalProperty(ov::model_name,
                                                true,
                                                ov::PropertyMutability::RO,
                                                [](const ::intel_npu::Config&) -> ov::Any {
                                                    return std::string("second");
                                                });
    ASSERT_EQ(propertiesManager->getProperty(ov::model_name.name()).as<std::string>(), "second");
}

// A private (visibility=false) property is still readable but must NOT appear in supported_properties.
TEST_F(CompiledModelPropertiesTests, RegisterExternalProperty_PrivateIsReadableButNotInSupportedProperties) {
    propertiesManager->registerExternalProperty(ov::model_name,
                                                false,
                                                ov::PropertyMutability::RO,
                                                [](const ::intel_npu::Config&) -> ov::Any {
                                                    return std::string("private_value");
                                                });

    ASSERT_EQ(propertiesManager->getProperty(ov::model_name.name()).as<std::string>(), "private_value");
    ASSERT_FALSE(isSupportedProperty(ov::model_name.name()));
}

using ExpectLoadingCompilerPropertyNotSupported = PropertiesManagerTests;

TEST_P(ExpectLoadingCompilerPropertyNotSupported, ExpectCompilerPropertyIsNotSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = true;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_FALSE(isSupported);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
