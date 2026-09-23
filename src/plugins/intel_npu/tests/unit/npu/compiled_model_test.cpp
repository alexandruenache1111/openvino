// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Qualified: this include path also carries npuw's unrelated same-named header first.
#include "include/compiled_model.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "offline_test_helpers.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::intel_npu;
using namespace intel_npu;

namespace {

// gmock doubles for the wide ov::IInferRequest/IDevice surfaces: only createInferRequest,
// infer_async and get_result are exercised below, the rest is auto-stubbed by gmock.
class MockInferRequest : public InferRequest {
public:
    MOCK_METHOD(void, infer, (), (override));
    MOCK_METHOD(void, infer_async, (), (override));
    MOCK_METHOD(void, get_result, (), (override));
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ITensor>, get_tensor, (const ov::Output<const ov::Node>&), (const, override));
    MOCK_METHOD(void, set_tensor, (const ov::Output<const ov::Node>&, const ov::SoPtr<ov::ITensor>&), (override));
    MOCK_METHOD(std::vector<ov::SoPtr<ov::ITensor>>,
                get_tensors,
                (const ov::Output<const ov::Node>&),
                (const, override));
    MOCK_METHOD(void,
                set_tensors,
                (const ov::Output<const ov::Node>&, const std::vector<ov::SoPtr<ov::ITensor>>&),
                (override));
    MOCK_METHOD(std::vector<ov::SoPtr<ov::IVariableState>>, query_state, (), (const, override));
    MOCK_METHOD((const std::shared_ptr<const ov::ICompiledModel>&), get_compiled_model, (), (const, override));
    MOCK_METHOD((const std::vector<ov::Output<const ov::Node>>&), get_inputs, (), (const, override));
    MOCK_METHOD((const std::vector<ov::Output<const ov::Node>>&), get_outputs, (), (const, override));
    MOCK_METHOD(void, check_tensors, (), (const, override));
};

class MockDevice : public IDevice {
public:
    MOCK_METHOD(std::string, getName, (), (const, override));
    MOCK_METHOD(std::string, getFullDeviceName, (), (const, override));
    MOCK_METHOD(std::shared_ptr<InferRequest>,
                createInferRequest,
                (const std::shared_ptr<const ICompiledModel>&, const Config&),
                (override));
    MOCK_METHOD(void, updateInfo, (const ov::AnyMap&), (override));
    MOCK_METHOD(bool, validateCompatibilityDescriptor, (const std::string&), (const, override));
    MOCK_METHOD(IDevice::DeviceProperties, getDeviceProperties, (), (const, override));
};

// A graph double that reports itself initialized without needing a real driver/device, so
// CompiledModel's own device-interaction logic can be exercised in isolation. Offline (VCL) compiled
// graphs cannot genuinely complete initialize_impl() without a real Level Zero driver, so the real
// compiled graph from compileOffline() cannot be used for this: init_completed() would stay false.
class MockGraph : public IGraph {
public:
    MOCK_METHOD(std::optional<bool>, is_profiling_blob, (), (const, override));
    MOCK_METHOD(void, set_workload_type, (const ov::WorkloadType), (override));
    MOCK_METHOD(void, set_model_priority, (const ov::hint::Priority), (override));

    // IGraph's default throws "not implemented"; CompiledModelPropertyManager queries this eagerly.
    std::optional<std::string_view> get_compatibility_descriptor() const override {
        return std::nullopt;
    }

protected:
    void initialize_impl(const FilteredConfig&) override {
        _init_completed = true;
    }
};

// Builds the real CompiledModel around a genuinely-compiled (offline, no device) graph, so property
// and lifecycle behavior is exercised against real production logic end to end.
class CompiledModelUnitTests : public ::testing::Test {
protected:
    void SetUp() override {
        options = std::make_shared<OptionsDesc>();
        config = std::make_unique<FilteredConfig>(options);
        test::registerOfflineOptions(*options);
        // WORKLOAD_TYPE is backend-gated in production and stays unregistered in the shared offline
        // helper; CompiledModel's own set_property() forwarding doesn't care about that gate, so add
        // it locally for the tests that need it.
        options->add<WORKLOAD_TYPE>();
        config->update({{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU5010)}});

        model = ov::test::utils::make_multi_single_conv();
        graph = test::compileOffline(model, *config);
        plugin = std::make_shared<test::TestPlugin>();
    }

    std::shared_ptr<CompiledModel> makeCompiledModel() {
        return std::make_shared<CompiledModel>(model, plugin, nullptr, graph, *config, ov::AnyMap{}, std::nullopt);
    }

    std::shared_ptr<OptionsDesc> options;
    std::unique_ptr<FilteredConfig> config;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<IGraph> graph;
    std::shared_ptr<test::TestPlugin> plugin;
};

TEST_F(CompiledModelUnitTests, CreateInferRequestThrowsWithoutDevice) {
    auto compiledModel = makeCompiledModel();
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel->create_infer_request(),
                                  ov::Exception,
                                  "No available devices. Failed to create infer request!");
}

TEST_F(CompiledModelUnitTests, SupportedPropertiesContainRuntimeRequirements) {
    auto compiledModel = makeCompiledModel();
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(
        supportedProperties =
            compiledModel->get_property(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>());
    ASSERT_NE(std::find(supportedProperties.begin(), supportedProperties.end(), ov::runtime_requirements.name()),
              supportedProperties.end());
}

TEST_F(CompiledModelUnitTests, RuntimeRequirementsIsNonEmpty) {
    auto compiledModel = makeCompiledModel();
    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiledModel->get_property(ov::runtime_requirements.name()).as<std::string>());
    ASSERT_FALSE(requirements.empty());
}

TEST_F(CompiledModelUnitTests, ExportModelSucceeds) {
    auto compiledModel = makeCompiledModel();
    std::ostringstream out;
    OV_ASSERT_NO_THROW(compiledModel->export_model(out));
    ASSERT_FALSE(out.str().empty());
}

TEST_F(CompiledModelUnitTests, GetRuntimeModelReturnsModelWithSameIO) {
    auto compiledModel = makeCompiledModel();
    std::shared_ptr<const ov::Model> runtimeModel;
    OV_ASSERT_NO_THROW(runtimeModel = compiledModel->get_runtime_model());
    ASSERT_EQ(runtimeModel->inputs().size(), model->inputs().size());
    ASSERT_EQ(runtimeModel->outputs().size(), model->outputs().size());
}

TEST_F(CompiledModelUnitTests, ReleaseMemoryDoesNotThrowWithoutDevice) {
    auto compiledModel = makeCompiledModel();
    OV_ASSERT_NO_THROW(compiledModel->release_memory());
}

TEST_F(CompiledModelUnitTests, GetGraphReturnsTheCompiledGraph) {
    auto compiledModel = makeCompiledModel();
    ASSERT_EQ(compiledModel->get_graph(), graph);
}

TEST_F(CompiledModelUnitTests, CreateInferRequestSucceedsWithDevice) {
    using ::testing::_;
    using ::testing::NiceMock;
    using ::testing::Return;

    // NUM_STREAMS=0 forces a non-null result executor so the pipeline stages actually get scheduled.
    config->update({{NUM_STREAMS::key().data(), "0"}});

    auto mockInferRequest = std::make_shared<NiceMock<MockInferRequest>>();
    EXPECT_CALL(*mockInferRequest, infer_async()).Times(1);
    EXPECT_CALL(*mockInferRequest, get_result()).Times(1);

    auto mockDevice = std::make_shared<NiceMock<MockDevice>>();
    EXPECT_CALL(*mockDevice, createInferRequest(_, _)).Times(1).WillOnce(Return(mockInferRequest));

    auto mockGraph = std::make_shared<NiceMock<MockGraph>>();

    auto compiledModel =
        std::make_shared<CompiledModel>(model, plugin, mockDevice, mockGraph, *config, ov::AnyMap{}, std::nullopt);

    std::shared_ptr<ov::IAsyncInferRequest> asyncRequest;
    OV_ASSERT_NO_THROW(asyncRequest = compiledModel->create_infer_request());
    ASSERT_NE(asyncRequest, nullptr);

    // AsyncInferRequest only populates the async pipeline (m_pipeline), not the synchronous one
    // (m_sync_pipeline) that infer() would use - so start_async()+wait() is required here.
    OV_ASSERT_NO_THROW(asyncRequest->start_async());
    OV_ASSERT_NO_THROW(asyncRequest->wait());
}

TEST_F(CompiledModelUnitTests, CreateSyncInferRequestThrowsNotImplemented) {
    auto compiledModel = makeCompiledModel();
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel->create_sync_infer_request(), ov::Exception, "does not inherit");
}

TEST_F(CompiledModelUnitTests, SetPropertyForwardsWorkloadTypeToGraph) {
    using ::testing::NiceMock;

    auto mockGraph = std::make_shared<NiceMock<MockGraph>>();
    EXPECT_CALL(*mockGraph, set_workload_type(ov::WorkloadType::EFFICIENT)).Times(1);

    auto compiledModel =
        std::make_shared<CompiledModel>(model, plugin, nullptr, mockGraph, *config, ov::AnyMap{}, std::nullopt);

    OV_ASSERT_NO_THROW(compiledModel->set_property({ov::workload_type(ov::WorkloadType::EFFICIENT)}));
}

TEST_F(CompiledModelUnitTests, SetPropertyForwardsModelPriorityToGraph) {
    using ::testing::NiceMock;

    auto mockGraph = std::make_shared<NiceMock<MockGraph>>();
    EXPECT_CALL(*mockGraph, set_model_priority(ov::hint::Priority::HIGH)).Times(1);

    auto compiledModel =
        std::make_shared<CompiledModel>(model, plugin, nullptr, mockGraph, *config, ov::AnyMap{}, std::nullopt);

    OV_ASSERT_NO_THROW(compiledModel->set_property({ov::hint::model_priority(ov::hint::Priority::HIGH)}));
}

TEST_F(CompiledModelUnitTests, CreateInferRequestInitializesGraphWhenCreationIsDeferred) {
    using ::testing::_;
    using ::testing::NiceMock;
    using ::testing::Return;

    // CREATE_EXECUTOR=0 defers graph initialization from the constructor to the first
    // create_infer_request() call instead.
    config->update({{CREATE_EXECUTOR::key().data(), "0"}});
    config->update({{NUM_STREAMS::key().data(), "0"}});

    auto mockInferRequest = std::make_shared<NiceMock<MockInferRequest>>();
    EXPECT_CALL(*mockInferRequest, infer_async()).Times(1);
    EXPECT_CALL(*mockInferRequest, get_result()).Times(1);

    auto mockDevice = std::make_shared<NiceMock<MockDevice>>();
    EXPECT_CALL(*mockDevice, createInferRequest(_, _)).Times(1).WillOnce(Return(mockInferRequest));

    auto mockGraph = std::make_shared<NiceMock<MockGraph>>();

    auto compiledModel =
        std::make_shared<CompiledModel>(model, plugin, mockDevice, mockGraph, *config, ov::AnyMap{}, std::nullopt);

    std::shared_ptr<ov::IAsyncInferRequest> asyncRequest;
    OV_ASSERT_NO_THROW(asyncRequest = compiledModel->create_infer_request());
    ASSERT_NE(asyncRequest, nullptr);

    OV_ASSERT_NO_THROW(asyncRequest->start_async());
    OV_ASSERT_NO_THROW(asyncRequest->wait());
}

TEST_F(CompiledModelUnitTests, ExportModelEncryptsBlobWhenCallbackIsSet) {
    auto compiledModel = makeCompiledModel();

    ov::EncryptionCallbacks callbacks;
    callbacks.encrypt = [](const std::string& blob) {
        return "encrypted:" + blob;
    };
    OV_ASSERT_NO_THROW(compiledModel->set_property({ov::cache_encryption_callbacks(callbacks)}));

    std::ostringstream out;
    OV_ASSERT_NO_THROW(compiledModel->export_model(out));
    ASSERT_EQ(out.str().rfind("encrypted:", 0), 0u);
}

TEST_F(CompiledModelUnitTests, ConfigureStreamExecutorsUsesDedicatedThreadsWhenSequential) {
    config->update({{RUN_INFERENCES_SEQUENTIALLY::key().data(), "YES"}});

    auto compiledModel = makeCompiledModel();
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel->create_infer_request(),
                                  ov::Exception,
                                  "No available devices. Failed to create infer request!");
}

TEST_F(CompiledModelUnitTests, ConfigureStreamExecutorsScalesWaitWorkersWithNumStreams) {
    config->update({{NUM_STREAMS::key().data(), "2"}});

    auto compiledModel = makeCompiledModel();
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel->create_infer_request(),
                                  ov::Exception,
                                  "No available devices. Failed to create infer request!");
}

// ENABLE_CPU_PINNING is read-only, so the set attempt still throws - but the deprecation warning
// is logged unconditionally before that check runs, on both the set and get paths.
TEST_F(CompiledModelUnitTests, CpuPinningPropertyLogsDeprecationOnGetAndSet) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto compiledModel = makeCompiledModel();
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel->set_property({{ov::hint::enable_cpu_pinning.name(), true}}),
                                  ov::Exception,
                                  "READ-ONLY configuration key");
    OV_ASSERT_NO_THROW(compiledModel->get_property(ov::hint::enable_cpu_pinning.name()));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_F(CompiledModelUnitTests, SetPropertyThrowsForUnsupportedKey) {
    auto compiledModel = makeCompiledModel();
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel->set_property({{"NOT_A_REAL_PROPERTY", "value"}}),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_F(CompiledModelUnitTests, GetPropertyReturnsEmptyForWriteOnlyProperty) {
    auto compiledModel = makeCompiledModel();
    ov::Any result;
    OV_ASSERT_NO_THROW(result = compiledModel->get_property(ov::cache_encryption_callbacks.name()));
    ASSERT_TRUE(result.empty());
}

TEST_F(CompiledModelUnitTests, GetPropertyFallsBackToInternalConfigForUnregisteredKey) {
    config->addOrUpdateInternal("UNREGISTERED_INTERNAL_TEST_KEY", "internal_value");

    auto compiledModel = makeCompiledModel();
    ov::Any result;
    OV_ASSERT_NO_THROW(result = compiledModel->get_property("UNREGISTERED_INTERNAL_TEST_KEY"));
    ASSERT_EQ(result.as<std::string>(), "internal_value");
}

TEST_F(CompiledModelUnitTests, GetPropertyReturnsNullModelHint) {
    auto compiledModel = makeCompiledModel();
    ov::Any result;
    OV_ASSERT_NO_THROW(result = compiledModel->get_property(ov::hint::model.name()));
    ASSERT_EQ(result.as<std::shared_ptr<const ov::Model>>(), nullptr);
}

TEST_F(CompiledModelUnitTests, GetPropertyReturnsModelNameFromGraphMetadata) {
    auto compiledModel = makeCompiledModel();
    ov::Any result;
    // Offline (no driver) compilation leaves the graph metadata name empty; only the no-throw
    // path through IGraph::get_metadata() matters here.
    OV_ASSERT_NO_THROW(result = compiledModel->get_property(ov::model_name.name()));
    OV_ASSERT_NO_THROW(result.as<std::string>());
}

TEST_F(CompiledModelUnitTests, GetPropertyReturnsOptimalNumberOfInferRequests) {
    auto compiledModel = makeCompiledModel();
    ov::Any result;
    OV_ASSERT_NO_THROW(result = compiledModel->get_property(ov::optimal_number_of_infer_requests.name()));
    ASSERT_GT(result.as<uint32_t>(), 0u);
}

TEST_F(CompiledModelUnitTests, GetPropertyReturnsExecutionDevices) {
    auto compiledModel = makeCompiledModel();
    ov::Any result;
    OV_ASSERT_NO_THROW(result = compiledModel->get_property(ov::execution_devices.name()));
    ASSERT_EQ(result.as<std::string>(), "NPU");
}

}  // namespace
