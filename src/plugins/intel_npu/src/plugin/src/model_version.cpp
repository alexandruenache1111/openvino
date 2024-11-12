// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_version.hpp"

#include <cstring>
#include <sstream>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/version.hpp"

namespace intel_npu {

OpenvinoVersion::OpenvinoVersion(const std::string& version) {
    this->version = version;
    this->size = static_cast<uint32_t>(version.size());
}

void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    stream.read(&version[0], size);
}

Metadata<2, 0>::Metadata() : version{2, 0}, ovVersion{ov::get_openvino_version().buildNumber} {}

void Metadata<2, 0>::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&dummy_extra), sizeof(dummy_extra));
    ovVersion.read(stream);
}

void Metadata<2, 0>::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&version.major), sizeof(version.major));
    stream.write(reinterpret_cast<const char*>(&version.minor), sizeof(version.minor));

    stream.write(reinterpret_cast<const char*>(&dummy_extra), sizeof(dummy_extra));

    stream.write(reinterpret_cast<const char*>(&ovVersion.size), sizeof(ovVersion.size));
    stream.write(ovVersion.version.c_str(), ovVersion.version.size());
}

std::unique_ptr<MetadataBase> createMetadata(int major, int minor) {
    switch (major) {
    case 2:
        switch (minor) {
        case 0:
            return std::make_unique<Metadata<2, 0>>();

        default:
            return nullptr;
        }

    default:
        return nullptr;
    }
}

bool Metadata<2, 0>::isCompatible() {
    // checking if we still support the format
    // but is checking `Major` redundant since it's checked in createMetadata?
    if (version.major != CURRENT_METAVERSION_MAJOR || version.minor != CURRENT_METAVERSION_MINOR) {
        return false;
    }
    // Checking if we can import the blob
    return ovVersion.version == ov::get_openvino_version().buildNumber;
}

std::unique_ptr<MetadataBase> read_metadata_from(std::vector<uint8_t>& blob) {
    Logger _logger("NPUPlugin", Logger::global().level());
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string blobMagicBytes(magicBytesSize, '\0');

    auto metadataIterator = blob.end() - magicBytesSize;
    memcpy(blobMagicBytes.data(), &(*metadataIterator), magicBytesSize);
    if (MAGIC_BYTES != blobMagicBytes) {
        _logger.error("Blob is not versioned");
        return nullptr;
    }

    size_t blobDataSize;
    metadataIterator -= sizeof(blobDataSize);
    memcpy(&blobDataSize, &(*metadataIterator), sizeof(blobDataSize));
    metadataIterator = blob.begin() + blobDataSize;

    std::stringstream metadataStream;
    metadataStream.write(reinterpret_cast<const char*>(&(*metadataIterator)),
                         blob.end() - metadataIterator - sizeof(blobDataSize));

    MetadataVersion metaVersion;
    metadataStream.read(reinterpret_cast<char*>(&metaVersion.major), sizeof(metaVersion.major));
    metadataStream.read(reinterpret_cast<char*>(&metaVersion.minor), sizeof(metaVersion.minor));

    auto storedMeta = createMetadata(metaVersion.major, metaVersion.minor);
    if (storedMeta != nullptr) {
        storedMeta->read(metadataStream);
    }
    return storedMeta;
}

}  // namespace intel_npu
