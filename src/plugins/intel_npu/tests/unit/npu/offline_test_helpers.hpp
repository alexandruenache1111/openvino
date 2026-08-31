// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace intel_npu {
namespace test {

// Duplicating logic because init_config() can be used only within plugin.cpp
// Does same thing as Plugin::getConfigForSpecificCompiler() but without a backend, so it can be used here
// This means this function should be maintained manually in sync with the one in production
// Deliberately mirrors production: backend-gated options (MAX_TILES, WORKLOAD_TYPE,
// DISABLE_IDLE_MEMORY_PRUNING) stay unregistered here, exactly like the real one when no backend exists
void registerOfflineOptions(OptionsDesc& options, FilteredConfig& config);

// Compiles a real model through the real /VCL path with no backend involved
std::shared_ptr<IGraph> compileOffline(const std::shared_ptr<ov::Model>& model, FilteredConfig& config);

// Minimal ov::IPlugin clone: CompiledModel only needs a plugin pointer for base bookkeeping
class TestPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    void set_property(const ov::AnyMap&) override {}

    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
    }
};

}  // namespace test
}  // namespace intel_npu
