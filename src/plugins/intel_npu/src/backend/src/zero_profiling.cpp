// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_profiling.hpp"

#include <ze_graph_profiling_ext.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "intel_npu/config/options.hpp"
#include "intel_npu/profiling.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "zero_profiling.hpp"

namespace intel_npu {
namespace zeroProfiling {

/// @brief Type trait mapping from ZE data type to enum value
template <typename T>
struct ZeProfilingTypeId {};

template <>
struct ZeProfilingTypeId<ze_profiling_layer_info> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_LAYER_LEVEL;
};

template <>
struct ZeProfilingTypeId<ze_profiling_task_info> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_TASK_LEVEL;
};

template <>
struct ZeProfilingTypeId<uint8_t> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_RAW;
};

bool ProfilingPool::create() {
    // To avoid  dynamic infer flow + perf_count_enabled calling static_cast<ze_graph_handle_t> which is not supported
    // in dynamic graph
    if (_graph->get_kind() == GraphKind::Dynamic) {
        return false;
    }

    auto ret = _init_structs->getProfilingDdiTable().pfnProfilingPoolCreate(
        static_cast<ze_graph_handle_t>(_graph->get_handle()),
        _profiling_count,
        &_handle);
    return ((ZE_RESULT_SUCCESS == ret) && (_handle != nullptr));
}

ProfilingPool::~ProfilingPool() {
    if (_handle && _init_structs->getContext()) {
        _init_structs->getProfilingDdiTable().pfnProfilingPoolDestroy(_handle);
    }
}

void ProfilingQuery::create(const std::shared_ptr<ProfilingPool>& profiling_pool) {
    _profiling_pool =
        profiling_pool;  // store profiling pool to make sure it is keeped alive until profiling query is destroyed
    THROW_ON_FAIL_FOR_LEVELZERO(
        "pfnProfilingQueryCreate",
        _init_structs->getProfilingDdiTable().pfnProfilingQueryCreate(_profiling_pool->_handle, _index, &_handle));
}

LayerStatistics ProfilingQuery::getLayerStatistics() const {
    verifyProfilingProperties();
    auto layerData = getData<ze_profiling_layer_info>();
    return profiling::convertLayersToIeProfilingInfo(layerData);
}

ProfilingQuery::~ProfilingQuery() {
    if (_handle && _init_structs->getContext()) {
        _init_structs->getProfilingDdiTable().pfnProfilingQueryDestroy(_handle);
    }
}

void ProfilingQuery::queryGetData(const ze_graph_profiling_type_t profilingType,
                                  uint32_t* pSize,
                                  uint8_t* pData) const {
    if (_handle && pSize) {
        THROW_ON_FAIL_FOR_LEVELZERO(
            "pfnProfilingQueryGetData",
            _init_structs->getProfilingDdiTable().pfnProfilingQueryGetData(_handle, profilingType, pSize, pData));
    }
}

template <class ProfilingData>
std::vector<ProfilingData> ProfilingQuery::getData() const {
    ze_graph_profiling_type_t type = ZeProfilingTypeId<ProfilingData>::value;
    uint32_t size = 0;

    // Obtain the size of the buffer
    queryGetData(type, &size, nullptr);

    OPENVINO_ASSERT(size % sizeof(ProfilingData) == 0);

    // Allocate enough memory and copy the buffer
    std::vector<ProfilingData> profilingData(size / sizeof(ProfilingData));
    queryGetData(type, &size, reinterpret_cast<uint8_t*>(profilingData.data()));
    return profilingData;
}

template std::vector<uint8_t> ProfilingQuery::getData<uint8_t>() const;

void ProfilingQuery::getProfilingProperties(ze_device_profiling_data_properties_t* properties) const {
    if (_handle && properties) {
        THROW_ON_FAIL_FOR_LEVELZERO(
            "getProfilingProperties",
            _init_structs->getProfilingDdiTable().pfnDeviceGetProfilingDataProperties(_init_structs->getDevice(),
                                                                                      properties));
    }
}

void ProfilingQuery::verifyProfilingProperties() const {
    if (!_handle) {
        OPENVINO_THROW("No available profiling data.");
    }
    const auto stringifyVersion = [](auto version) -> std::string {
        return std::to_string(ZE_MAJOR_VERSION(version)) + "." + std::to_string(ZE_MINOR_VERSION(version));
    };

    ze_device_profiling_data_properties_t profProp = {};
    profProp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROFILING_DATA_PROPERTIES;
    getProfilingProperties(&profProp);
    const auto currentProfilingVersion = ze_profiling_data_ext_version_t::ZE_PROFILING_DATA_EXT_VERSION_CURRENT;

    if (ZE_MAJOR_VERSION(profProp.extensionVersion) != ZE_MAJOR_VERSION(currentProfilingVersion)) {
        OPENVINO_THROW("Unsupported NPU driver.",
                       "Profiling API version: plugin: ",
                       stringifyVersion(currentProfilingVersion),
                       ", driver: ",
                       stringifyVersion(profProp.extensionVersion));
    }
    if (currentProfilingVersion > profProp.extensionVersion) {
        auto log = Logger::global().clone("ZeroProfilingQuery");
        log.warning("Outdated NPU driver detected. Some features might not be available! "
                    "Profiling API version: plugin: %s, driver: %s",
                    stringifyVersion(currentProfilingVersion).c_str(),
                    stringifyVersion(profProp.extensionVersion).c_str());
    }
}

namespace {

int64_t nsToUs(int64_t duration_ns) {
    /// round to nearest instead of truncating: sub-millisecond inferences lose a lot of accuracy otherwise
    return (duration_ns + 500) / 1000;
}

ov::ProfilingInfo makeInferProfilingInfo(const std::string& name, int64_t duration_us) {
    return ov::ProfilingInfo{ov::ProfilingInfo::Status::EXECUTED,
                             std::chrono::microseconds(duration_us),
                             std::chrono::microseconds(duration_us),
                             name,
                             name,
                             name,
                             std::chrono::microseconds::zero()};
}

/// Nearest-rank percentile over an ascending-sorted sample set
int64_t percentileOfSorted(const std::vector<int64_t>& sorted, double percentile) {
    const auto rank = static_cast<size_t>(std::ceil(percentile / 100.0 * static_cast<double>(sorted.size())));
    return sorted[std::min(sorted.size(), std::max<size_t>(rank, 1)) - 1];
}

}  // namespace

void InferDurationStats::record(int64_t duration_ns) {
    if (duration_ns < _min_ns)
        _min_ns = duration_ns;
    if (duration_ns > _max_ns) {
        _max_ns = duration_ns;
        _max_iteration = _cnt;
    }
    _accu_ns += duration_ns;
    if (_cnt == 0) {
        _first_ns = duration_ns;
    } else if (duration_ns > _max_after_first_ns) {
        _max_after_first_ns = duration_ns;
        _max_after_first_iteration = _cnt;
    }
    if (_cnt < _early_log_maxsize) {
        _early_duration_log[_cnt] = duration_ns;
    } else if (duration_ns > _max_after_warmup_ns) {
        _max_after_warmup_ns = duration_ns;
        _max_after_warmup_iteration = _cnt;
    }
    _cnt++;
    _duration_log[_logidx++] = duration_ns;
    if (_logidx >= _log_maxsize)
        _logidx = 0;
}

std::vector<int64_t> InferDurationStats::chronologicalDurations() const {
    const uint32_t stored = std::min(_cnt, _log_maxsize);
    std::vector<int64_t> durations;
    durations.reserve(stored);

    /// once the buffer has rolled over, the oldest sample sits at the current write index
    const uint32_t oldest = (_cnt <= _log_maxsize) ? 0 : _logidx;
    for (uint32_t i = 0; i < stored; i++) {
        durations.push_back(_duration_log[(oldest + i) % _log_maxsize]);
    }
    return durations;
}

uint32_t InferDurationStats::detectWarmupEnd() const {
    /// At top frequency every sample lands within a fraction of a percent of MIN, while the DVFS ramp runs
    /// 10-50% above it. Requiring several consecutive settled samples rejects the brief fast states the
    /// governor visits while it is still hunting.
    constexpr int64_t tolerance_percent = 2;
    constexpr uint32_t sustain = 8;

    const int64_t threshold = _min_ns + _min_ns * tolerance_percent / 100;
    const uint32_t stored = std::min(_cnt, _early_log_maxsize);

    uint32_t settled_run = 0;
    for (uint32_t i = 0; i < stored; i++) {
        if (_early_duration_log[i] <= threshold) {
            if (++settled_run == sustain) {
                return i + 1 - sustain;
            }
        } else {
            settled_run = 0;
        }
    }
    return _cnt;
}

void InferDurationStats::appendTo(NpuInferStatistics& stats, const std::string& prefix, bool include_samples) const {
    /// sanity check to avoid division by 0
    if (_cnt == 0) {
        return;
    }

    const std::vector<int64_t> durations = chronologicalDurations();
    /// index of the oldest buffered sample within the whole inference history
    const uint32_t first_logged_iteration = _cnt - static_cast<uint32_t>(durations.size());

    if (include_samples) {
        /// the first <_early_log_maxsize> durations, kept because the rolling buffer has long overwritten them
        if (first_logged_iteration != 0) {
            for (uint32_t i = 0; i < std::min(_cnt, _early_log_maxsize); i++) {
                stats.push_back(makeInferProfilingInfo(prefix + "EARLY_" + std::to_string(i),
                                                       nsToUs(_early_duration_log[i])));
                stats.back().exec_type = "INFER_REQ";
                stats.back().node_type = "INFER_REQ";
            }
        }

        /// the last <_log_maxsize> durations, labelled with their true iteration number
        for (size_t i = 0; i < durations.size(); i++) {
            stats.push_back(
                makeInferProfilingInfo(prefix + std::to_string(first_logged_iteration + i), nsToUs(durations[i])));
            stats.back().exec_type = "INFER_REQ";
            stats.back().node_type = "INFER_REQ";
        }
    }

    /// mean over every inference since the request was created
    stats.push_back(makeInferProfilingInfo(prefix + "AVG", nsToUs(_accu_ns / _cnt)));
    /// fastest inference of the whole run
    stats.push_back(makeInferProfilingInfo(prefix + "MIN", nsToUs(_min_ns)));
    /// slowest inference of the whole run, tagged with the iteration it happened on
    stats.push_back(makeInferProfilingInfo(prefix + "MAX", nsToUs(_max_ns)));
    stats.back().exec_type = "iter " + std::to_string(_max_iteration);

    /// duration of inference 0, which alone pays the driver and firmware one-time setup cost
    stats.push_back(makeInferProfilingInfo(prefix + "FIRST", nsToUs(_first_ns)));
    if (_cnt > 1) {
        /// AVG with inference 0 taken out
        stats.push_back(
            makeInferProfilingInfo(prefix + "AVG_NO_FIRST", nsToUs((_accu_ns - _first_ns) / (_cnt - 1))));
        /// MAX with inference 0 taken out
        stats.push_back(makeInferProfilingInfo(prefix + "MAX_NO_FIRST", nsToUs(_max_after_first_ns)));
        stats.back().exec_type = "iter " + std::to_string(_max_after_first_iteration);
    }

    /// Warm-up is only excludable once the run actually reached steady state; a short run that never got
    /// there has no steady-state figure to report, and saying nothing is better than reporting a cold one.
    const uint32_t warmup_end = detectWarmupEnd();
    if (warmup_end < _cnt) {
        const int64_t warmup_accu = std::accumulate(_early_duration_log, _early_duration_log + warmup_end, int64_t{0});

        /// how long the DVFS ramp lasted: total wall time in realTime, inference count in execType
        stats.push_back(makeInferProfilingInfo(prefix + "WARMUP", nsToUs(warmup_accu)));
        stats.back().exec_type = "iter " + std::to_string(warmup_end);

        /// mean of every inference after the ramp - the steady-state figure
        stats.push_back(
            makeInferProfilingInfo(prefix + "AVG_NO_WARMUP", nsToUs((_accu_ns - warmup_accu) / (_cnt - warmup_end))));

        /// worst inference after the ramp; _max_after_warmup_ns only covers iterations past the early buffer,
        /// so the settled part of that buffer has to be folded in to keep this exact
        int64_t max_ns = _max_after_warmup_ns;
        uint32_t max_iter = _max_after_warmup_iteration;
        for (uint32_t i = warmup_end; i < std::min(_cnt, _early_log_maxsize); i++) {
            if (_early_duration_log[i] > max_ns) {
                max_ns = _early_duration_log[i];
                max_iter = i;
            }
        }
        stats.push_back(makeInferProfilingInfo(prefix + "MAX_NO_WARMUP", nsToUs(max_ns)));
        stats.back().exec_type = "iter " + std::to_string(max_iter);
    }

    /// Order-statistics over the rolling window: immune to a single warm-up or scheduling outlier, unlike AVG/MAX.
    std::vector<int64_t> sorted = durations;
    std::sort(sorted.begin(), sorted.end());

    /// AVG/MIN/MAX span the whole run; everything below spans only the buffered window, so report its mean too -
    /// a gap between AVG and WINDOW_AVG means the run drifted rather than stayed in steady state.
    const double mean_ns = std::accumulate(sorted.cbegin(), sorted.cend(), 0.0) / static_cast<double>(sorted.size());
    stats.push_back(makeInferProfilingInfo(prefix + "WINDOW_AVG", nsToUs(static_cast<int64_t>(mean_ns))));

    /// the middle sample: half the window was faster, half slower
    stats.push_back(makeInferProfilingInfo(prefix + "MEDIAN", nsToUs(percentileOfSorted(sorted, 50.0))));
    /// 90% of the window came in at or below this
    stats.push_back(makeInferProfilingInfo(prefix + "P90", nsToUs(percentileOfSorted(sorted, 90.0))));
    /// 99% of the window came in at or below this - the tail, without letting one outlier define it
    stats.push_back(makeInferProfilingInfo(prefix + "P99", nsToUs(percentileOfSorted(sorted, 99.0))));

    const double variance_ns =
        std::accumulate(sorted.cbegin(),
                        sorted.cend(),
                        0.0,
                        [mean_ns](double acc, int64_t ns) {
                            return acc + (static_cast<double>(ns) - mean_ns) * (static_cast<double>(ns) - mean_ns);
                        }) /
        static_cast<double>(sorted.size());
    /// spread of the window around its mean: large next to MEDIAN means the run was noisy and not worth trusting
    stats.push_back(makeInferProfilingInfo(prefix + "STDDEV", nsToUs(static_cast<int64_t>(std::sqrt(variance_ns)))));
}

NpuInferStatistics NpuInferProfiling::getNpuInferStatistics() const {
    NpuInferStatistics npuPerfCounts;
    _device_stats.appendTo(npuPerfCounts, "", _loglevel >= ov::log::Level::WARNING);
    _host_stats.appendTo(npuPerfCounts, "E2E_", false);
    return npuPerfCounts;
}

NpuInferProfiling::NpuInferProfiling(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                                         ov::log::Level loglevel,
                                                                         size_t timestamp_count)
    : _init_structs(init_structs),
      _loglevel(loglevel),
            _logger("InferProfiling", loglevel),
            _npu_ts_infer_start(timestamp_count, nullptr),
            _npu_ts_infer_end(timestamp_count, nullptr) {
    /// Fetch and store the device timer resolution
    _dev_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_init_structs->getDevice(), &_dev_properties));
    /// Request mem allocations
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                     nullptr,
                                     ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED};
    try {
        for (size_t i = 0; i < timestamp_count; ++i) {
            THROW_ON_FAIL_FOR_LEVELZERO(
                "zeMemAllocHost",
                zeMemAllocHost(_init_structs->getContext(), &desc, sizeof(uint64_t), 64, &_npu_ts_infer_start[i]));
            THROW_ON_FAIL_FOR_LEVELZERO(
                "zeMemAllocHost",
                zeMemAllocHost(_init_structs->getContext(), &desc, sizeof(uint64_t), 64, &_npu_ts_infer_end[i]));
        }
    } catch (...) {
        for (size_t i = 0; i < timestamp_count; ++i) {
            if (_npu_ts_infer_start[i] != nullptr) {
                zeMemFree(_init_structs->getContext(), _npu_ts_infer_start[i]);
            }
            if (_npu_ts_infer_end[i] != nullptr) {
                zeMemFree(_init_structs->getContext(), _npu_ts_infer_end[i]);
            }
        }
        throw;
    }
}

void NpuInferProfiling::sampleNpuTimestamps(size_t timestamp_index) {
    OPENVINO_ASSERT(timestamp_index < _npu_ts_infer_start.size(), "Invalid NPU timestamp index");
    int64_t infer_duration_cc = static_cast<int64_t>(
        *(reinterpret_cast<uint64_t*>(_npu_ts_infer_end[timestamp_index])) -
        *(reinterpret_cast<uint64_t*>(_npu_ts_infer_start[timestamp_index])));

    _device_stats.record(convertCCtoNS(infer_duration_cc));
}

void* NpuInferProfiling::getTimestampStart(size_t timestamp_index) const {
    OPENVINO_ASSERT(timestamp_index < _npu_ts_infer_start.size(), "Invalid NPU timestamp index");
    return _npu_ts_infer_start[timestamp_index];
}

void* NpuInferProfiling::getTimestampEnd(size_t timestamp_index) const {
    OPENVINO_ASSERT(timestamp_index < _npu_ts_infer_end.size(), "Invalid NPU timestamp index");
    return _npu_ts_infer_end[timestamp_index];
}

void NpuInferProfiling::markHostInferStart() {
    _host_infer_start = std::chrono::steady_clock::now();
    _host_infer_started = true;
}

void NpuInferProfiling::markHostInferEnd() {
    /// guard against a pull() that was not preceded by a push() on this pipeline
    if (!_host_infer_started) {
        return;
    }
    _host_stats.record(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                           _host_infer_start)
                           .count());
    _host_infer_started = false;
}

int64_t NpuInferProfiling::convertCCtoNS(int64_t val_cc) const {
    const int64_t resolution = static_cast<int64_t>(_dev_properties.timerResolution);
    return (val_cc * 1000 * 1000 * 1000 + resolution / 2) / resolution;
}

NpuInferProfiling::~NpuInferProfiling() {
    /// deallocate npu_ts_infer_start and npu_ts_infer_end, allocated externally by ze driver
    auto context = _init_structs->getContext();
    if (context == nullptr) {
        return;
    }
    for (auto timestamp : _npu_ts_infer_start) {
        if (timestamp != nullptr) {
            auto ze_ret = zeMemFree(context, timestamp);
            if (ZE_RESULT_SUCCESS != ze_ret) {
                _logger.error("zeMemFree on npu timestamp start failed %#X", uint64_t(ze_ret));
            }
        }
    }
    for (auto timestamp : _npu_ts_infer_end) {
        if (timestamp != nullptr) {
            auto ze_ret = zeMemFree(context, timestamp);
            if (ZE_RESULT_SUCCESS != ze_ret) {
                _logger.error("zeMemFree on npu timestamp end failed %#X", uint64_t(ze_ret));
            }
        }
    }
}

}  // namespace zeroProfiling
}  // namespace intel_npu
