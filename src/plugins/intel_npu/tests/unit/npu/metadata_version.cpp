// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "metadata.hpp"

using namespace intel_npu;

using MetadataUnitTests = ::testing::Test;

TEST_F(MetadataUnitTests, readUnversionedBlob) {
    std::stringstream stream(" ELF");

    auto storedMeta = read_metadata_from(stream);
    ASSERT_EQ(storedMeta, nullptr);
}

TEST_F(MetadataUnitTests, writeAndReadMetadataFromBlob) {
    std::stringstream stream;
    size_t blobSize = 0;
    auto meta = Metadata<CURRENT_METADATA_VERSION>(stream.tellg(), blobSize);

    OV_ASSERT_NO_THROW(meta.write(stream));
    OV_ASSERT_NO_THROW(stream.write(reinterpret_cast<const char*>(&blobSize), sizeof(blobSize)));
    OV_ASSERT_NO_THROW(stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size()));

    auto storedMeta = read_metadata_from(stream);
    ASSERT_NE(storedMeta, nullptr);
    ASSERT_TRUE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidOpenvinoVersion) {
    size_t blobSize = 0;
    std::stringstream stream;
    auto meta = Metadata<CURRENT_METADATA_VERSION>(stream.tellg(), blobSize);

    OpenvinoVersion badOvVersion("just_some_wrong_ov_version");
    meta.set_ov_version(badOvVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));
    OV_ASSERT_NO_THROW(stream.write(reinterpret_cast<const char*>(&blobSize), sizeof(blobSize)));
    OV_ASSERT_NO_THROW(stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size()));

    auto storedMeta = read_metadata_from(stream);
    ASSERT_NE(storedMeta, nullptr);
    ASSERT_FALSE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidMetadataVersion) {
    size_t blobSize = 0;
    std::stringstream stream;
    auto meta = Metadata<CURRENT_METADATA_VERSION>(stream.tellg(), blobSize);

    constexpr uint32_t dummy_version = make_version(0x00007E57, 0x0000AC3D);
    meta.set_version(dummy_version);

    OV_ASSERT_NO_THROW(meta.write(stream));
    OV_ASSERT_NO_THROW(stream.write(reinterpret_cast<const char*>(&blobSize), sizeof(blobSize)));
    OV_ASSERT_NO_THROW(stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size()));

    auto storedMeta = read_metadata_from(stream);
    ASSERT_EQ(storedMeta, nullptr);
}
