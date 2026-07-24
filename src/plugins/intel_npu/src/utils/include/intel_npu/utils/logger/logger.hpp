// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Class for pretty-logging.
//

#pragma once

#include <iostream>
#include <optional>
#include <sstream>

#include "openvino/runtime/properties.hpp"

namespace intel_npu {

//
// Logger
//

std::string printFormattedCStr(const char* fmt, ...)
#if defined(__clang__)
    ;
#elif defined(__GNUC__) || defined(__GNUG__)
    __attribute__((format(printf, 1, 2)));
#else
    ;
#endif

// Thread-safe backing store for the process-wide log level (baseline + per-thread override). Definition lives in
// logger.cpp: only Logger::globalStore() needs it, and an incomplete type is enough for that declaration.
class GlobalLevelStore;

class Logger {
public:
    static Logger& global();

    Logger(const char* name, ov::log::Level lvl = ov::log::Level::NO);
    Logger(const std::string&, ov::log::Level lvl = ov::log::Level::NO) = delete;
    Logger(const std::string_view& name, ov::log::Level lvl = ov::log::Level::NO) = delete;
    Logger(const Logger& log) = default;

    // Creates a logger that reflects the process-wide level store (baseline + per-thread override) rather than a
    // fixed snapshot. Use this for long-lived loggers that must honor a per-call LOG_LEVEL without being mutated.
    static Logger followingGlobal(const char* name);

    // RAII scope that overrides the process-wide log level for the current thread only, restoring the previous value
    // on destruction. Use this to apply a per-call LOG_LEVEL for the duration of a compile/import/query call: the
    // change is visible to every global-following logger on this thread and is rolled back when the call returns,
    // without touching the shared baseline or racing other threads.
    class GlobalLevelGuard {
    public:
        explicit GlobalLevelGuard(ov::log::Level lvl);
        ~GlobalLevelGuard();
        GlobalLevelGuard(const GlobalLevelGuard&) = delete;
        GlobalLevelGuard& operator=(const GlobalLevelGuard&) = delete;
        // Movable so it can be held in a std::optional and returned by value; a moved-from guard is disarmed and
        // restores nothing on destruction.
        GlobalLevelGuard(GlobalLevelGuard&& other) noexcept;
        GlobalLevelGuard& operator=(GlobalLevelGuard&&) = delete;

    private:
        bool _armed = true;
        std::optional<ov::log::Level> _previous;
    };

    Logger clone(const char* name) const;
    Logger clone(const std::string& name) const = delete;
    Logger clone(const std::string_view& name) const = delete;

    void setName(const std::string& name) = delete;
    void setName(const std::string_view& name) = delete;

    auto name() const {
        return _name;
    }

    void setName(const char* name) {
        _name = name;
    }

    ov::log::Level level() const;

    Logger& setLevel(ov::log::Level lvl);

    bool isActive(ov::log::Level msgLevel) const;

    template <typename... Args>
    void error(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::ERR, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warning(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::WARNING, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::INFO, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::DEBUG, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void trace(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::TRACE, format, std::forward<Args>(args)...);
    }

private:
    // Returns the shared store used by global() and followingGlobal() loggers.
    static GlobalLevelStore& globalStore();

    void addEntryPackedActive(ov::log::Level msgLevel, const std::string_view msg) const;

    template <typename... Args>
    void addEntryPacked(ov::log::Level msgLevel, const char* format, Args&&... args) const {
        if (!isActive(msgLevel)) {
            return;
        }
        addEntryPackedActive(msgLevel, printFormattedCStr(format, std::forward<Args>(args)...));
    }

    void addEntryPacked(ov::log::Level msgLevel, const char* msg) const {
        if (!isActive(msgLevel)) {
            return;
        }
        addEntryPackedActive(msgLevel, msg);
    }

private:
    const char* _name;
    // When true, this logger reads its level from the shared global store (baseline + per-thread override) and
    // ignores _logLevel. When false, it is an ordinary value-semantic logger holding its own fixed level.
    bool _followsGlobal = false;
    ov::log::Level _logLevel = ov::log::Level::NO;
};

}  // namespace intel_npu
