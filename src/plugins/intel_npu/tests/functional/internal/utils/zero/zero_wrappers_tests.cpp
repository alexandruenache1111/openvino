// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/zero/zero_wrappers_tests.hpp"

#include "npu_test_env_cfg.hpp"
#include "utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ZeroWrappersTests,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ZeroWrappersTests::getTestCaseName);
