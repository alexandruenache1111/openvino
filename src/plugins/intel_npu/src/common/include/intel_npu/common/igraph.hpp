// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

/**
 * @brief Abstract interface representing a compiled NPU graph.
 *
 * @details An `IGraph` encapsulates a compiled model (blob) that has been loaded onto —
 * or is ready to be loaded onto — the NPU device.  It is the central object shared
 * between the compiler adapter layer (which produces the graph) and the backend layer
 * (which executes it through Level-Zero pipelines).
 *
 * Initialization is lazy and thread-safe: the first inference request to use the graph
 * calls `initialize()`, which holds `_initialize_mutex` and — if `_init_completed` is
 * still `false` — invokes `initialize_impl()`. The subclass override of
 * `initialize_impl()` is responsible for publishing `_init_completed = true` with
 * release ordering once setup has succeeded, so that all subsequent callers short-circuit.
 * Concrete subclasses — `Graph` (static model) and `DynamicGraph` (dynamic shapes) —
 * override `initialize_impl()` and the other pure/defaulted virtual methods to provide
 * driver-specific behaviour.
 *
 * @note Instances are always managed through `std::shared_ptr<IGraph>`.
 *       Calling code must not hold a raw owning pointer to this object.
 *
 * @note The mutex `_initialize_mutex` also guards zero pipeline construction inside
 *       a compiled model; do not acquire it recursively.
 */
class IGraph : public std::enable_shared_from_this<IGraph> {
public:
    IGraph() = default;

    /**
     * @brief Serialises the compiled model to a stream for later import.
     *
     * @details Writes the compiled blob and any required metadata to @p stream.
     * When weights separation is enabled the stream contains both a main schedule
     * binary and one or more init-schedule binaries; the returned sizes allow the
     * caller to locate each section within the stream.
     *
     * @param stream Output stream that receives the serialised blob.
     * @return A pair where:
     *   - `first`  is the byte size of the main compiled-model binary, and
     *   - `second` is, when weights separation is active, a vector whose elements
     *              are the byte sizes of each init-schedule binary (one entry per
     *              init schedule); `std::nullopt` when weights separation is disabled.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const;

    /**
     * @brief Decodes raw profiling data produced by the NPU into OpenVINO profiling records.
     *
     * @param profData Raw profiling buffer obtained from the device after inference.
     * @return A vector of `ov::ProfilingInfo` entries, one per profiled layer/operation.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData) const;

    /**
     * @brief Binds a host buffer to a graph argument (input or output tensor) by index.
     *
     * @param id   Zero-based driver index of the argument (see `IODescriptor::indexUsedByDriver`).
     * @param data Pointer to the host buffer.  The buffer must remain valid until the
     *             inference completes; the caller is responsible for lifetime management.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void set_argument_value(uint32_t id, const void* data) const;

    /**
     * @brief Binds a non-contiguous host buffer with explicit element strides to a graph argument.
     *
     * @details Use this overload when the tensor layout in memory does not match the
     * packed (row-major) layout expected by default.  Each entry in @p strides is
     * the element stride (not byte stride) for the corresponding dimension.
     *
     * @param id      Zero-based driver index of the argument.
     * @param data    Pointer to the first element of the host buffer.
     * @param strides Per-dimension element strides; `strides[i]` is the number of
     *                elements between consecutive indices along dimension `i`.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void set_argument_value_with_strides(uint32_t id,
                                                 const void* data,
                                                 const std::vector<size_t>& strides) const;

    /**
     * @brief Performs one-time, thread-safe initialisation of the graph.
     *
     * @details Acquires `_initialize_mutex` and, if `_init_completed` is still `false`,
     * delegates to `initialize_impl()`. Concurrent callers are serialised by the mutex,
     * and subsequent calls short-circuit once the subclass has marked initialisation as
     * complete (see the contract documented on `initialize_impl()`).
     *
     * @param config Plugin configuration snapshot used during initialisation
     *               (e.g., to select pipeline parameters or device-specific settings).
     */
    void initialize(const FilteredConfig& config);

    virtual ~IGraph() = default;

    /**
     * @brief Returns the I/O metadata for the compiled model.
     *
     * @return A const reference to the `NetworkMetadata` describing the inputs, outputs,
     *         states, and shapes of the compiled model.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual const NetworkMetadata& get_metadata() const;

    /**
     * @brief Returns the Level-Zero graph handle for direct driver interactions.
     *
     * @return The `ze_graph_handle_t` associated with this compiled graph.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual ze_graph_handle_t get_handle() const;

    /**
     * @brief Updates the network name stored inside the graph's metadata.
     *
     * @param name New name to assign to the network.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void update_network_name(std::string_view name);

    /**
     * @brief Returns the command queue descriptor used when submitting work for this graph.
     *
     * @return A `CommandQueueDesc` encoding the queue group index, priority, and other
     *         scheduling attributes required to create a matching command queue.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual CommandQueueDesc get_command_queue_desc() const;

    /**
     * @brief Applies an OpenVINO workload-type hint to the graph's scheduling behaviour.
     *
     * @param workloadType One of the `ov::WorkloadType` enumerators (e.g., `DEFAULT`,
     *                     `EFFICIENT`).
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void set_workload_type(const ov::WorkloadType workloadType);

    /**
     * @brief Applies an OpenVINO model-priority hint to the graph's scheduling behaviour.
     *
     * @param modelPriority One of `ov::hint::Priority` (`LOW`, `MEDIUM`, `HIGH`).
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void set_model_priority(const ov::hint::Priority modelPriority);

    /**
     * @brief Returns a reference to the initialisation mutex.
     *
     * @details Exposed so that the compiled-model layer can hold the same lock while
     * constructing the zero pipeline, ensuring that graph initialisation and pipeline
     * creation are mutually exclusive.
     *
     * @return A reference to `_initialize_mutex`.
     */
    std::mutex& get_mutex() {
        return _initialize_mutex;
    }

    /**
     * @brief Returns `true` once `initialize()` has completed successfully.
     *
     * @details The read uses `std::memory_order_acquire` so that all writes performed
     * inside `initialize_impl()` are visible to the caller after this returns `true`.
     *
     * @return `true` if the graph has been fully initialised; `false` otherwise.
     */
    bool init_completed() const {
        return _init_completed.load(std::memory_order_acquire);
    }

    /**
     * @brief Records the last Level-Zero event submitted on a specific command list.
     *
     * @details The pipeline uses this to chain dependency between successive submissions
     * on the same command list index.
     *
     * @param event              The event signalled at the end of the most recent submission.
     * @param indexOfCommandList Zero-based index identifying the command list (equals the
     *                           pipeline slot / batch index).
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList);

    /**
     * @brief Retrieves the last Level-Zero event submitted on a specific command list.
     *
     * @param indexOfCommandList Zero-based index identifying the command list.
     * @return A const reference to the shared event handle (may hold `nullptr` before any
     *         submission).
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual const std::shared_ptr<Event>& get_last_submitted_event(size_t indexOfCommandList) const;

    /**
     * @brief Resizes the internal per-command-list event storage to accommodate @p batch entries.
     *
     * @details Called when the batch size changes and the number of active command lists
     * must grow to match.
     *
     * @param batch New number of command lists (batch size).
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void resize_last_submitted_event(size_t batch);

    /**
     * @brief Sets the batch size used during inference.
     *
     * @param batch Number of independent inference requests bundled into a single
     *              Level-Zero submission.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void set_batch_size(std::size_t batch);

    /**
     * @brief Returns the current batch size if one has been set.
     *
     * @return The batch size, or `std::nullopt` if batching is not active for this graph.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual const std::optional<std::size_t> get_batch_size() const;

    /**
     * @brief Returns a monotonically increasing identifier for the next pipeline instance.
     *
     * @details Each call increments an internal counter and returns its previous value.
     * The returned ID is used to label a zero pipeline so that `set_last_submitted_id` /
     * `get_last_submitted_id` can track which pipeline submitted last.
     *
     * @return A unique `uint32_t` ID for the calling pipeline.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual uint32_t get_unique_id();

    /**
     * @brief Records the unique ID of the most recently submitted pipeline.
     *
     * @details The backend calls this after enqueuing a command list so that a subsequent
     * caller can detect whether a newer submission has overtaken the current one.
     *
     * @param id_index ID previously obtained from `get_unique_id()`.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual void set_last_submitted_id(uint32_t id_index);

    /**
     * @brief Returns the unique ID of the most recently submitted pipeline.
     *
     * @return The ID recorded by the last `set_last_submitted_id()` call, or `0` if
     *         no submission has been recorded yet.
     * @throws ov::Exception if the concrete subclass has not implemented this method.
     */
    virtual uint32_t get_last_submitted_id() const;

    /**
     * @brief Releases device-side memory associated with the graph without destroying the object.
     *
     * @details Delegates to the Level-Zero graph extension to free the device allocation,
     * reducing memory pressure when the graph is temporarily idle.
     *
     * @note The base-class implementation is a no-op; only `Graph` provides a real
     *       implementation through `ZeGraphExtWrappers::evict_memory()`.
     * @note `_init_completed` is **not** cleared by this call, so `initialize()` will not
     *       re-allocate the device memory afterwards. Subsequent inference on an evicted
     *       graph requires the caller to ensure the device allocation is restored by
     *       other means (or to recreate the graph object).
     */
    virtual void evict_memory();

    /**
     * @brief Returns whether the compiled blob contains profiling instrumentation.
     *
     * @return `true` if the blob was compiled with profiling enabled,
     *         `false` if it was not, or `std::nullopt` if the information is unavailable.
     * @note Pure virtual; every concrete subclass must implement this.
     */
    virtual std::optional<bool> is_profiling_blob() const = 0;

    /**
     * @brief Returns an opaque descriptor string used for forward/backward compatibility checks.
     *
     * @details The descriptor encodes version or capability information that the plugin
     * compares against the current driver/compiler to detect incompatible blobs.
     *
     * @return A `string_view` over the descriptor stored in the concrete object, or
     *         `std::nullopt` when the subclass has no descriptor to expose. The
     *         base-class implementation never returns — it always throws.
     * @throws ov::Exception in the base-class implementation (must be overridden).
     */
    virtual std::optional<std::string_view> get_compatibility_descriptor() const;

protected:
    /**
     * @brief Performs the actual, non-thread-safe initialisation work.
     *
     * @details Called by `initialize()` while `_initialize_mutex` is held, and only when
     * `_init_completed` is still `false`. Subclasses override this to create Level-Zero
     * pipelines, allocate device memory, or perform any other one-time setup.
     *
     * @param config Plugin configuration snapshot forwarded from `initialize()`.
     * @throws ov::Exception in the base-class implementation (must be overridden by
     *         any concrete subclass that is intended to be initialised).
     *
     * @note Contract for overrides: on successful completion, the override **must**
     *       publish `_init_completed.store(true, std::memory_order_release)` so that
     *       future calls to `initialize()` short-circuit. On a failure path the flag
     *       must be left as `false` to allow a retry.
     */
    virtual void initialize_impl(const FilteredConfig& config);

    /**
     * @brief Mutex that serialises `initialize_impl()` across concurrent callers.
     *
     * @details Also exposed via `get_mutex()` to the compiled-model layer so that
     * zero-pipeline construction and graph initialisation are mutually exclusive.
     */
    std::mutex _initialize_mutex;

    /**
     * @brief Flag set to `true` (with `memory_order_release`) once `initialize_impl()` returns.
     *
     * @details Checked with `memory_order_acquire` in `initialize()` and `init_completed()`
     * to guarantee that all initialisation side-effects are visible before the flag is observed
     * as `true`.
     */
    std::atomic<bool> _init_completed{false};
};

}  // namespace intel_npu