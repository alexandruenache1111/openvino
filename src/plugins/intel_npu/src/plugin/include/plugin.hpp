// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "backends_registry.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "metadata.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "plugin_property_manager.hpp"

namespace intel_npu {

class Plugin : public ov::IPlugin {
public:
    Plugin();

    Plugin(const Plugin&) = delete;

    Plugin& operator=(const Plugin&) = delete;

    ~Plugin() = default;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    bool is_property_supported(const std::string& name, const ov::AnyMap& arguments = {}) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remoteProperties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remoteProperties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& compiledBlob,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& compiledBlob,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    // Permanently updates the global log level baseline from a LOG_LEVEL in @p properties (used by set_property).
    void update_log_level(const ov::AnyMap& properties) const;

    // Returns an RAII guard that applies a per-call LOG_LEVEL from @p properties for the current thread only,
    // restoring the previous level when the returned optional is destroyed. Empty when no LOG_LEVEL was provided.
    // Used by compile_model/import_model/query_model so a per-call level is scoped to the call and never leaks into
    // the persistent baseline or races other threads. Keep the returned value alive for the whole call.
    [[nodiscard]] std::optional<Logger::GlobalLevelGuard> scoped_log_level(const ov::AnyMap& properties) const;

    /**
     * @brief Parses the compiled model found within the stream and tensor and returns a wrapper over the L0 handle that
     * can be used for running predictions.
     * @details The binary data corresponding to the compiled model is made of NPU plugin metadata, the schedule of
     * the model and its weights. If weights separation has been enabled, the size of the weights is reduced, and there
     * will be one or multiple weights initialization schedules found there as well.
     *
     * @param tensorBig Contains the whole binary object.
     * @param metadata Parsed metadata at the end of the blob. Can be nullptr if compatibility checks were disabled.
     * @param properties Configuration taking the form of an "ov::AnyMap".
     * @return A compiled model
     */
    std::shared_ptr<ov::ICompiledModel> parse(const ov::Tensor& tensorBig,
                                              std::unique_ptr<MetadataBase> metadata,
                                              const ov::AnyMap& properties) const;

    std::unique_ptr<BackendsRegistry> _backendsRegistry;

    //  _backend might not be set by the plugin; certain actions, such as offline compilation, might be supported.
    //  Appropriate checks are needed in plugin/metrics/properties when actions depend on a backend.
    ov::SoPtr<IEngineBackend> _backend;

    mutable Logger _logger;
    std::unique_ptr<PluginPropertyManager> _propertiesManager;

    static std::atomic<int> _compiledModelLoadCounter;
};

}  // namespace intel_npu
