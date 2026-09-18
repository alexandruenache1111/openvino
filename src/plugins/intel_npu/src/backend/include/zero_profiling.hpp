// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>

#include <chrono>
#include <climits>
#include <map>
#include <string>
#include <vector>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {
namespace zeroProfiling {

using LayerStatistics = std::vector<ov::ProfilingInfo>;

constexpr uint32_t POOL_SIZE = 1;

struct ProfilingPool {
    ProfilingPool(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                  const std::shared_ptr<IGraph>& graph,
                  uint32_t profiling_count)
        : _init_structs(init_structs),
          _graph(graph),
          _profiling_count(profiling_count) {}
    ProfilingPool(const ProfilingPool&) = delete;
    ProfilingPool& operator=(const ProfilingPool&) = delete;
    bool create();

    ~ProfilingPool();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    std::shared_ptr<IGraph> _graph;
    const uint32_t _profiling_count;

    ze_graph_profiling_pool_handle_t _handle = nullptr;
};

struct ProfilingQuery {
    ProfilingQuery(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, uint32_t index)
        : _init_structs(init_structs),
          _index(index) {}
    ProfilingQuery(const ProfilingQuery&) = delete;
    ProfilingQuery& operator=(const ProfilingQuery&) = delete;
    void create(const std::shared_ptr<ProfilingPool>& profiling_pool);
    ze_graph_profiling_query_handle_t getHandle() const {
        return _handle;
    }
    LayerStatistics getLayerStatistics() const;
    template <class ProfilingData>
    std::vector<ProfilingData> getData() const;
    ~ProfilingQuery();

private:
    void queryGetData(const ze_graph_profiling_type_t profilingType, uint32_t* pSize, uint8_t* pData) const;
    void getProfilingProperties(ze_device_profiling_data_properties_t* properties) const;
    void verifyProfilingProperties() const;

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    const uint32_t _index;

    std::shared_ptr<ProfilingPool> _profiling_pool = nullptr;

    ze_graph_profiling_query_handle_t _handle = nullptr;
};

extern template std::vector<uint8_t> ProfilingQuery::getData<uint8_t>() const;

using NpuInferStatistics = std::vector<ov::ProfilingInfo>;

/// Rolling statistics over a series of inference durations, stored in nanoseconds so that host and device
/// measurements share one representation and keep sub-microsecond precision until they are reported.
struct InferDurationStats {
    void record(int64_t duration_ns);

    /// Appends the summary rows, each name carrying the given prefix
    void appendTo(NpuInferStatistics& stats, const std::string& prefix, bool include_samples) const;

private:
    static constexpr uint32_t _log_maxsize = 1024;
    /// doubles as the warm-up window: the DVFS governor is observed to keep hunting for ~190 inferences
    static constexpr uint32_t _early_log_maxsize = 256;

    int64_t _min_ns = LLONG_MAX;
    int64_t _max_ns = 0;
    int64_t _accu_ns = 0;
    uint32_t _cnt = 0;
    /// duration of the very first infer, kept apart so warm-up cost can be excluded from the steady-state numbers
    int64_t _first_ns = 0;
    int64_t _max_after_first_ns = 0;
    /// iteration that produced each maximum, so a reproducible spike can be located within the run
    uint32_t _max_iteration = 0;
    uint32_t _max_after_first_iteration = 0;
    int64_t _max_after_warmup_ns = 0;
    uint32_t _max_after_warmup_iteration = 0;
    uint32_t _logidx = 0;
    /// rolling buffer holding the last <_log_maxsize> durations
    int64_t _duration_log[_log_maxsize] = {};
    /// the rolling buffer has long overwritten the start of a long run, so keep the warm-up ramp separately
    int64_t _early_duration_log[_early_log_maxsize] = {};

    /// Returns the buffered durations in chronological order (oldest first)
    std::vector<int64_t> chronologicalDurations() const;

    /// Index of the first inference that reached steady state, or _cnt if the run never got there
    uint32_t detectWarmupEnd() const;
};

struct NpuInferProfiling final {
    NpuInferProfiling(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                     ov::log::Level loglevel,
                     size_t timestamp_count);
    NpuInferProfiling(const NpuInferProfiling&) = delete;
    NpuInferProfiling& operator=(const NpuInferProfiling&) = delete;
    NpuInferProfiling(NpuInferProfiling&&) = delete;
    NpuInferProfiling& operator=(NpuInferProfiling&&) = delete;

    void sampleNpuTimestamps(size_t timestamp_index);

    void* getTimestampStart(size_t timestamp_index) const;
    void* getTimestampEnd(size_t timestamp_index) const;

    /// Host-side span covering submission and synchronisation as well as device execution;
    /// its gap to the device numbers is the overhead outside the NPU.
    void markHostInferStart();
    void markHostInferEnd();

    NpuInferStatistics getNpuInferStatistics() const;

    ~NpuInferProfiling();

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    ov::log::Level _loglevel;
    Logger _logger;
    ze_device_properties_t _dev_properties = {};

    InferDurationStats _device_stats;
    InferDurationStats _host_stats;
    std::chrono::steady_clock::time_point _host_infer_start;
    bool _host_infer_started = false;
    std::vector<void*> _npu_ts_infer_start;
    std::vector<void*> _npu_ts_infer_end;

    /// Helper function to convert npu clockcycles to nsec
    int64_t convertCCtoNS(int64_t val_cc) const;
};

}  // namespace zeroProfiling
}  // namespace intel_npu
