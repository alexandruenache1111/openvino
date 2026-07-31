// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>

class RandomGenerator {
public:
    static std::mt19937& get();

private:
    static RandomGenerator& getInstance();

    RandomGenerator() = default;
    RandomGenerator(const RandomGenerator&) = delete;
    RandomGenerator& operator=(const RandomGenerator&) = delete;
    ~RandomGenerator() = default;

    std::mt19937 generator{std::random_device{}()};
};

RandomGenerator& RandomGenerator::getInstance() {
    static RandomGenerator instance;
    return instance;
}

std::mt19937& RandomGenerator::get() {
    return getInstance().generator;
}
