// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/logger/logger.hpp"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>

#include "openvino/core/log_util.hpp"

#define DEFAULT_COLOR "\033[0m"
#define RED           "\033[31m"
#define GREEN         "\033[32m"
#define YELLOW        "\033[33m"
#define BLUE          "\033[34m"
#define CYAN          "\033[36m"

namespace intel_npu {

std::string printFormattedCStr(const char* fmt, ...) {
    // Process with original buffer
    const int bufferSize = 256;
    std::va_list args;
    va_start(args, fmt);
    std::va_list argsForFinalBuffer;
    va_copy(argsForFinalBuffer, args);
    char buffer[bufferSize];
    auto requiredBytes = vsnprintf(buffer, bufferSize, fmt, args);
    va_end(args);

    if (requiredBytes < 0) {
        va_end(argsForFinalBuffer);
        return std::string("vsnprintf got error from fmt: ") + fmt;
    } else if (requiredBytes >= bufferSize) {
        // vsnprintf returns the length excluding the NUL; buffer holds bufferSize bytes = (bufferSize - 1) chars +
        // NUL. So requiredBytes == bufferSize already means the first call truncated - reformat into a big-enough
        // buffer. std::string(requiredBytes, 0) gives requiredBytes writable chars plus the implicit NUL slot, i.e.
        // requiredBytes + 1 bytes, which is exactly what vsnprintf needs.
        std::string out(requiredBytes, 0);
        vsnprintf(out.data(), requiredBytes + 1, fmt, argsForFinalBuffer);
        va_end(argsForFinalBuffer);
        return out;
    }

    va_end(argsForFinalBuffer);
    return buffer;
}

//
// Logger
//
static const char* logLevelPrintout[] = {"NONE", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"};

// Thread-safe backing store for the process-wide log level.
//
// The effective level is the composition of two things:
//   - a persistent baseline (atomic), set once from env/build and optionally updated by set_property; shared by all
//     threads. Reads/writes are atomic, so there is no data race even under concurrent access.
//   - a per-thread override (thread_local optional), installed for the duration of a single plugin call by
//     GlobalLevelGuard. Being thread-local, one thread's per-call level can never be observed - or clobbered - by
//     another thread. This is what makes per-call LOG_LEVEL both correct (scoped to the call) and race-free.
//
// A Logger created via Logger::global() or Logger::followingGlobal() reads through this store instead of holding its
// own level, so it always reflects the current baseline/override without anyone mutating a shared Logger instance.
class GlobalLevelStore {
public:
    ov::log::Level get() const {
        if (const auto& perThread = perThreadOverride()) {
            return *perThread;
        }
        // Relaxed is sufficient: _baseline is the only state readers synchronize on here - no other write needs to
        // become visible alongside it, so there is no happens-before relationship to establish beyond the atomicity
        // of this load itself.
        return _baseline.load(std::memory_order_relaxed);
    }

    void setBaseline(ov::log::Level lvl) {
        // See get(): relaxed matches, no other memory access is ordered against this store.
        _baseline.store(lvl, std::memory_order_relaxed);
    }

    // Installs a per-thread override and returns the previous one so it can be restored (supports nesting).
    std::optional<ov::log::Level> exchangeOverride(std::optional<ov::log::Level> next) {
        auto previous = perThreadOverride();
        perThreadOverride() = next;
        return previous;
    }

private:
    static std::optional<ov::log::Level>& perThreadOverride() {
        thread_local std::optional<ov::log::Level> perThread;
        return perThread;
    }

    std::atomic<ov::log::Level> _baseline{ov::log::Level::NO};
};

GlobalLevelStore& Logger::globalStore() {
    // Meyers singleton: the baseline is seeded once from the build default and, in developer/debug builds, the
    // OV_NPU_LOG_LEVEL environment variable. Thereafter the baseline is only changed via Logger::global().setLevel()
    // (atomic), and per-call overrides go through the thread-local slot - so no shared Logger instance is mutated.
    // GlobalLevelStore holds an atomic and is therefore non-copyable; seed it in place after construction.
    static GlobalLevelStore store;
    static std::once_flag seeded;
    std::call_once(seeded, [] {
#if defined(NPU_PLUGIN_DEVELOPER_BUILD) || !defined(NDEBUG)
        ov::log::Level logLvl = ov::log::Level::WARNING;
        if (const auto env = std::getenv("OV_NPU_LOG_LEVEL")) {
            try {
                std::istringstream is(env);
                is >> logLvl;
            } catch (...) {
                // Use default log level
            }
        }
        store.setBaseline(logLvl);
#else
        store.setBaseline(ov::log::Level::ERR);
#endif
    });
    return store;
}

Logger::Logger(const char* name, ov::log::Level lvl) : _name(name), _logLevel(lvl) {}

Logger Logger::followingGlobal(const char* name) {
    Logger logger(name);
    logger._followsGlobal = true;
    return logger;
}

Logger& Logger::global() {
    static Logger log = Logger::followingGlobal("global");
    return log;
}

Logger::GlobalLevelGuard::GlobalLevelGuard(ov::log::Level lvl)
    : _previous(Logger::globalStore().exchangeOverride(lvl)) {}

Logger::GlobalLevelGuard::~GlobalLevelGuard() {
    if (_armed) {
        Logger::globalStore().exchangeOverride(_previous);
    }
}

Logger::GlobalLevelGuard::GlobalLevelGuard(GlobalLevelGuard&& other) noexcept
    : _armed(other._armed),
      _previous(other._previous) {
    other._armed = false;
}

Logger Logger::clone(const char* name) const {
    Logger logger(name, level());
    return logger;
}

ov::log::Level Logger::level() const {
    return _followsGlobal ? globalStore().get() : _logLevel;
}

Logger& Logger::setLevel(ov::log::Level lvl) {
    if (_followsGlobal) {
        // Setting the level on a global-following logger updates the shared persistent baseline (thread-safe),
        // never a per-instance field.
        globalStore().setBaseline(lvl);
    } else {
        _logLevel = lvl;
    }
    return *this;
}

bool Logger::isActive(ov::log::Level msgLevel) const {
    return static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(level());
}

namespace {

const char* getColor(ov::log::Level msgLevel) {
    switch (msgLevel) {
    case ov::log::Level::ERR:
        return RED;
    case ov::log::Level::WARNING:
        return YELLOW;
    case ov::log::Level::INFO:
        return CYAN;
    case ov::log::Level::DEBUG:
        return GREEN;
    case ov::log::Level::TRACE:
        return BLUE;
    default:
        return DEFAULT_COLOR;
    }
}

}  // namespace

void Logger::addEntryPackedActive(ov::log::Level msgLevel, std::string_view msg) const {
    char timeStr[] = "undefined_time";
    std::time_t now = std::time(nullptr);
    // localtime() returns a pointer to a single shared static std::tm, so concurrent loggers would race on it. Use
    // the reentrant, per-platform variant into a local buffer instead.
    std::tm localTimeBuf{};
#if defined(_WIN32)
    const bool haveLocalTime = (localtime_s(&localTimeBuf, &now) == 0);
#else
    const bool haveLocalTime = (localtime_r(&now, &localTimeBuf) != nullptr);
#endif
    if (haveLocalTime) {
        std::strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &localTimeBuf);
    }

    using namespace std::chrono;
    uint32_t ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() % 1000;
    try {
        std::stringstream logStream;
        logStream << getColor(msgLevel) << "[" << logLevelPrintout[static_cast<int32_t>(msgLevel) + 1] << "] "
                  << timeStr << "." << ms << " [" << _name << "] " << msg << DEFAULT_COLOR;
        static std::mutex logMtx;
        std::lock_guard<std::mutex> logMtxLock(logMtx);
        ov::util::log_message(logStream.str());
    } catch (const std::exception& e) {
        std::cerr << "Exception caught in Logger::addEntryPackedActive - " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened in Logger::addEntryPackedActive" << std::endl;
    }
}

}  // namespace intel_npu
