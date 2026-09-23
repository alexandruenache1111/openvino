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

// Duplicating logic because register_options() is file-local to plugin.cpp and can't be called from
// test code. Mirrors Plugin's option registration for the "no backend" case; maintained by hand in
// sync with the production version.
void registerOfflineOptions(OptionsDesc& options);

// Compiles via a fake ICompilerAdapter (no real VCL, no backend) and returns the resulting fake IGraph.
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
