// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include <cstring>
#include <optional>
#include <sstream>

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace {

size_t getFileSize(std::istream& stream) {
    auto log = intel_npu::Logger::global().clone("getFileSize");
    if (!stream) {
        OPENVINO_THROW("Stream is in bad status! Please check the passed stream status!");
    }

    const size_t streamStart = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const size_t streamEnd = stream.tellg();
    stream.seekg(streamStart, std::ios_base::beg);

    log.debug("Read blob size: streamStart=%zu, streamEnd=%zu", streamStart, streamEnd);

    if (streamEnd < streamStart) {
        OPENVINO_THROW("Invalid stream size: streamEnd (",
                    streamEnd,
                    ") is not larger than streamStart (",
                    streamStart,
                    ")!");
    }

    return streamEnd - streamStart;
}

}  // namespace

namespace intel_npu {

OpenvinoVersion::OpenvinoVersion(std::string_view version)
    : _version(version),
      _size(static_cast<uint32_t>(version.size())) {}

void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&_size), sizeof(_size));
    _version.resize(_size);
    stream.read(_version.data(), _size);
}

void OpenvinoVersion::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_size), sizeof(_size));
    stream.write(_version.data(), _size);
}

Metadata<METADATA_VERSION_1_0>::Metadata(std::optional<std::string_view> ovVersion, size_t ovHeaderOffset, uint64_t blobDataSize)
    : _version{METADATA_VERSION_1_0},
      _ovVersion{ovVersion.value_or(ov::get_openvino_version().buildNumber)},
      _ovHeaderOffset{ovHeaderOffset},
      _blobDataSize{blobDataSize} {}

void Metadata<METADATA_VERSION_1_0>::read(std::istream& stream) {
    _ovVersion.read(stream);
}

void Metadata<METADATA_VERSION_1_0>::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_version), sizeof(_version));
    _ovVersion.write(stream);
}

std::unique_ptr<MetadataBase> create_metadata(uint32_t version, size_t ovHeaderOffset, uint64_t blobDataSize) {
    switch (version) {
    case METADATA_VERSION_1_0:
        return std::make_unique<Metadata<METADATA_VERSION_1_0>>(std::nullopt, ovHeaderOffset, blobDataSize);

    default:
        return nullptr;
    }
}

std::string OpenvinoVersion::get_version() {
    return _version;
}

bool Metadata<METADATA_VERSION_1_0>::is_compatible() {
    Logger logger("NPUPlugin", Logger::global().level());
    // checking if we can import the blob
    if (_ovVersion.get_version() != ov::get_openvino_version().buildNumber) {
        logger.warning("Imported blob OpenVINO version: %s, but the current OpenVINO version is: %s",
                       _ovVersion.get_version().c_str(),
                       ov::get_openvino_version().buildNumber);

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
        if (auto envVar = std::getenv("NPU_DISABLE_VERSION_CHECK")) {
            if (envVarStrToBool("NPU_DISABLE_VERSION_CHECK", envVar)) {
                return true;
            }
        }
#endif
        return false;
    }
    return true;
}

std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream) {
    Logger logger("NPUPlugin", Logger::global().level());
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string blobMagicBytes;
    blobMagicBytes.resize(magicBytesSize);

    size_t currentStreamPos = stream.tellg();
    size_t streamSize = getFileSize(stream);
    stream.seekg(streamSize - magicBytesSize, std::ios::beg);
    stream.read(blobMagicBytes.data(), magicBytesSize);
    if (MAGIC_BYTES != blobMagicBytes) {
        logger.error("Blob is missing NPU metadata!");
        return nullptr;
    }

    uint64_t blobDataSize;
    stream.seekg(streamSize - magicBytesSize - sizeof(blobDataSize), std::ios::beg);
    stream.read(reinterpret_cast<char*>(&blobDataSize), sizeof(blobDataSize));
    stream.seekg(currentStreamPos + blobDataSize, std::ios::beg);

    uint32_t metaVersion;
    stream.read(reinterpret_cast<char*>(&metaVersion), sizeof(metaVersion));

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        storedMeta = create_metadata(metaVersion, currentStreamPos, blobDataSize);
        storedMeta->read(metadataStream);
    } catch (...) {
        logger.warning("Imported blob metadata version: %d.%d, but the current version is: %d.%d",
                       get_major(metaVersion),
                       get_minor(metaVersion),
                       get_major(CURRENT_METADATA_VERSION),
                       get_minor(CURRENT_METADATA_VERSION));
    }
    stream.seekg(currentStreamPos, std::ios::beg);
    return storedMeta;
}

std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream, const std::shared_ptr<ov::AlignedBuffer>& modelBuffer) {
    Logger logger("NPUPlugin", Logger::global().level());
    if (modelBuffer == nullptr) {
        return read_metadata_from(stream);
    }
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string blobMagicBytes;
    blobMagicBytes.resize(magicBytesSize);

    size_t currentStreamPos = stream.tellg();
    size_t streamSize = modelBuffer->size();

    blobMagicBytes.assign(reinterpret_cast<const char*>(modelBuffer->get_ptr(streamSize - magicBytesSize)), magicBytesSize);
    if (MAGIC_BYTES != blobMagicBytes) {
        logger.error("Blob is missing NPU metadata!");
        return nullptr;
    }

    uint64_t blobDataSize;
    blobDataSize = *reinterpret_cast<uint64_t*>(modelBuffer->get_ptr(streamSize - magicBytesSize - sizeof(blobDataSize)));

    uint32_t metaVersion;
    metaVersion = *reinterpret_cast<uint32_t*>(modelBuffer->get_ptr(currentStreamPos + blobDataSize));

    stream.seekg(blobDataSize + sizeof(metaVersion), std::ios::cur);
    try {
        auto storedMeta = create_metadata(metaVersion, currentStreamPos, blobDataSize);
        storedMeta->read(stream);
    } catch(...) {
        logger.warning("Imported blob metadata version: %d.%d, but the current version is: %d.%d",
                       get_major(metaVersion),
                       get_minor(metaVersion),
                       get_major(CURRENT_METADATA_VERSION),
                       get_minor(CURRENT_METADATA_VERSION));

        OPENVINO_THROW("NPU metadata mismatch.");
    }
    return storedMeta;
}

void Metadata<METADATA_VERSION_1_0>::set_version(uint32_t newVersion) {
    _version = newVersion;
}

void Metadata<METADATA_VERSION_1_0>::set_ov_version(const OpenvinoVersion& newVersion) {
    _ovVersion = newVersion;
}

uint64_t Metadata<METADATA_VERSION_1_0>::get_blob_size() const {
    return _blobDataSize;
}

size_t Metadata<METADATA_VERSION_1_0>::get_ov_header_offset() const {
    return _ovHeaderOffset;
}

}  // namespace intel_npu
