// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/log.hpp"
#include "openvino/core/log_util.hpp"

using intel_npu::Logger;

namespace {

// These tests cover the thread-safety and scoping guarantees of the process-wide NPU log level:
//   - a per-call level applied via Logger::GlobalLevelGuard is thread-local: it is visible to global-following
//     loggers on the same thread and invisible to other threads;
//   - the guard restores the previous level on scope exit (no leak into the persistent baseline);
//   - concurrent logging together with concurrent level changes does not data-race (meant to be run under TSan).
// They need no NPU device, so they live in the unit test target.

// Namespace-scope so the stress-test worker lambda can use them without an explicit capture (MSVC does not treat a
// constexpr local as implicitly capturable).
constexpr int kStressThreads = 8;
constexpr int kStressIterations = 200;

class LoggerThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Establish a known, stable baseline for every test and confirm the follower observes it.
        Logger::global().setLevel(ov::log::Level::WARNING);
    }

    void TearDown() override {
        Logger::global().setLevel(ov::log::Level::WARNING);
    }
};

// A global-following logger reports the baseline set on Logger::global(), and setLevel on any follower updates the
// shared baseline (there is a single source of truth, not a per-instance copy).
TEST_F(LoggerThreadSafetyTest, FollowingLoggerTracksSharedBaseline) {
    Logger follower = Logger::followingGlobal("follower");
    EXPECT_EQ(follower.level(), ov::log::Level::WARNING);

    Logger::global().setLevel(ov::log::Level::DEBUG);
    EXPECT_EQ(follower.level(), ov::log::Level::DEBUG);

    // Setting the level through the follower updates the same shared baseline.
    follower.setLevel(ov::log::Level::ERR);
    EXPECT_EQ(Logger::global().level(), ov::log::Level::ERR);
}

// The GlobalLevelGuard overrides the level for the current scope and restores it on exit.
TEST_F(LoggerThreadSafetyTest, GuardRestoresPreviousLevelOnScopeExit) {
    Logger::global().setLevel(ov::log::Level::WARNING);
    {
        Logger::GlobalLevelGuard guard(ov::log::Level::TRACE);
        EXPECT_EQ(Logger::global().level(), ov::log::Level::TRACE);
    }
    EXPECT_EQ(Logger::global().level(), ov::log::Level::WARNING);
}

// Guards nest correctly: the innermost override wins, and each scope restores the one it replaced.
TEST_F(LoggerThreadSafetyTest, GuardsNest) {
    Logger::global().setLevel(ov::log::Level::WARNING);
    {
        Logger::GlobalLevelGuard outer(ov::log::Level::INFO);
        EXPECT_EQ(Logger::global().level(), ov::log::Level::INFO);
        {
            Logger::GlobalLevelGuard inner(ov::log::Level::TRACE);
            EXPECT_EQ(Logger::global().level(), ov::log::Level::TRACE);
        }
        EXPECT_EQ(Logger::global().level(), ov::log::Level::INFO);
    }
    EXPECT_EQ(Logger::global().level(), ov::log::Level::WARNING);
}

// The core of the per-call design: a per-call override on one thread must not be observed by another thread, and
// must not disturb the shared baseline the other thread reads. This is what makes concurrent compile_model calls
// with different LOG_LEVEL values safe.
TEST_F(LoggerThreadSafetyTest, PerCallOverrideIsThreadLocal) {
    Logger::global().setLevel(ov::log::Level::WARNING);

    std::atomic<bool> otherThreadEnteredScope{false};
    std::atomic<bool> mainThreadChecked{false};
    ov::log::Level levelSeenByOtherThread = ov::log::Level::NO;
    ov::log::Level baselineSeenByMainThread = ov::log::Level::NO;

    std::thread other([&] {
        Logger::GlobalLevelGuard guard(ov::log::Level::TRACE);
        levelSeenByOtherThread = Logger::global().level();  // must be its own override
        otherThreadEnteredScope = true;

        // Hold the override until the main thread has observed its own (baseline) value.
        while (!mainThreadChecked.load()) {
            std::this_thread::yield();
        }
    });

    while (!otherThreadEnteredScope.load()) {
        std::this_thread::yield();
    }

    // While the other thread holds a TRACE override, this thread (no override) must still see the WARNING baseline.
    baselineSeenByMainThread = Logger::global().level();
    mainThreadChecked = true;
    other.join();

    EXPECT_EQ(levelSeenByOtherThread, ov::log::Level::TRACE);
    EXPECT_EQ(baselineSeenByMainThread, ov::log::Level::WARNING);
}

// Stress: many threads concurrently log, change the baseline, and install per-call overrides. With a redirected
// OV log callback in place, this exercises both the NPU logger's level store and the core log_message path. The
// test asserts nothing beyond "does not crash / no sanitizer report" - it is a race detector, not a value check.
TEST_F(LoggerThreadSafetyTest, ConcurrentLoggingAndLevelChangesDoNotRace) {
    std::atomic<int> messageCount{0};
    const std::function<void(std::string_view)> sink = [&](std::string_view) {
        messageCount.fetch_add(1, std::memory_order_relaxed);
    };
    ov::util::set_log_callback(sink);
    // Restore the default OV callback no matter how the test exits (sink must outlive the callback registration).
    struct ResetGuard {
        ~ResetGuard() {
            ov::util::reset_log_callback();
        }
    } resetGuard;

    std::vector<std::thread> threads;
    threads.reserve(kStressThreads);
    for (int t = 0; t < kStressThreads; ++t) {
        threads.emplace_back([t] {
            Logger logger = Logger::followingGlobal("stress");
            for (int i = 0; i < kStressIterations; ++i) {
                if ((t + i) % 3 == 0) {
                    // Writer: permanently move the shared baseline.
                    Logger::global().setLevel((i % 2 == 0) ? ov::log::Level::TRACE : ov::log::Level::ERR);
                } else if ((t + i) % 3 == 1) {
                    // Per-call override on this thread only.
                    Logger::GlobalLevelGuard guard(ov::log::Level::DEBUG);
                    logger.error("stress error t=%d i=%d", t, i);
                    logger.debug("stress debug t=%d i=%d", t, i);
                } else {
                    // Reader/logger without an override.
                    logger.warning("stress warning t=%d i=%d", t, i);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // Baseline is a normal, readable value after all the churn; exact value is irrelevant.
    const auto finalLevel = Logger::global().level();
    EXPECT_TRUE(finalLevel == ov::log::Level::TRACE || finalLevel == ov::log::Level::ERR ||
                finalLevel == ov::log::Level::WARNING);
    SUCCEED();
}

}  // namespace
