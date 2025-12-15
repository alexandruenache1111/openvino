// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <ostream>
#include <istream>
#include <vector>

// #include "compiled_model.hpp"
// #include "intel_npu/common/igraph.hpp"
#include "metadata.hpp"

using namespace intel_npu;

struct ICapability;

std::vector<std::shared_ptr<ICapability>> caps;

enum CAPABILITIES {
    ELF_BLOB = 1,
    WEIGHTS_SEPARATION,
    BATCH_SIZE,
    INPUT_OUTPUT_LAYOUTS,
    OR_GROUP,
    EXPRESSION
};

struct ICapability {
    int64_t type;
    int64_t length;
    // bool required;

    virtual void write(std::ostream& stream) = 0;

    virtual void read_value(std::istream& stream) = 0;

    virtual bool is_supported() = 0;

    // virtual void get_size() = 0;
};

// no idea how to represent this atm
struct CapabilityExpression : ICapability {
    std::string expression;

    // some constructor

    void write(std::ostream& stream) override;

    void read_value(std::istream& stream) override;

    bool isCompatible(); // ?
};

void readCapabilities(std::istream& stream, std::vector<std::shared_ptr<ICapability>>& caps);

void export_blob(std::ostream& stream) {

}

struct OrGroup : ICapability {
    std::vector<std::shared_ptr<ICapability>> group;

    int64_t groupSize;

    int64_t type;
    int64_t length;

    OrGroup(int64_t type, int64_t length) : type(type), length(length) {}

    void read_value(std::istream& stream) override {
        stream.read(reinterpret_cast<char*>(&groupSize), sizeof(groupSize));

        group.reserve(groupSize);

        for (int i = 0; i < groupSize; i++) {
            readCapabilities(stream, group);
        }
    }

    OrGroup(std::vector<std::shared_ptr<ICapability>> caps) : group(std::move(caps)) {
        type = OR_GROUP;
        length = sizeof(group.size());
        for (auto cap : group) {
            length += sizeof(type) + sizeof(length); // header size
            length += cap->length; // payload length
        }
    }

    void write(std::ostream& stream) override {
        stream.write(reinterpret_cast<const char*>(&type), sizeof(type));
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
        
        groupSize = group.size();
        stream.write(reinterpret_cast<const char*>(&groupSize), sizeof(groupSize));
        

        for (auto cap : group) {
            cap->write(stream);
        }
    }

    bool is_supported() override {
        for (auto cap : group) {
            if (cap->is_supported()) {
                return true;
            }
        }
        return false;
    }
};

struct CapabilityELFBlob : ICapability {
    // std::shared_ptr<intel_npu::IGraph> graph;
    int64_t type;
    int64_t length;

    // explicit CapabilityELFBlob(std::shared_ptr<IGraph> graph) : graph(graph) {
    //     type = ELF_BLOB;
    //     length = get_size(graph);
    // }

    void write(std::ostream& stream) override {
        stream.write(reinterpret_cast<const char*>(&type), sizeof(type));
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));

        // graph->export_blob(stream);
        export_blob(stream);
    }

    CapabilityELFBlob(int64_t type, int64_t length) : type(type), length(length) {}

    void read_value(std::istream& stream) override {
        std::vector<uint8_t*> blob;
        blob.reserve(length);

        stream.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(length));
    }

    bool is_supported() override {
        return true;
    }
};

// struct CapabilityWeightsSeparation : ICapability {
//     // uint64_t numSizes; // technically speaking, this is an auxiliary field since it is used only for read/write
//     // std::vector<uint64_t> numInits;
//     std::vector<CapabilityELFBlob> blobs;

//     explicit CapabilityWeightsSeparation(std::vector<uint64_t>& numInits) : numInits(numInits) {}

//     void write(std::ostream& stream) override {
//         stream.write(type);
//         stream.write(length);

//         stream.write(blobs.size());

//         for(int i = 0; i < blobs.size(); i++) {
//             blobs[i].write(stream);
//         }
//     }

//     void read_value(std::istream& stream) override {
//         stream.read(numBlobs);

//         blobs.reserve(numBlobs);

//         for (int i = 0; i < numBlobs; i++) {
//             stream.read(tag_i);
//             stream.read(length_i);

//             auto cap = CapabilityELFBlob(tag_i, length_i);
//             cap.read_value(stream);
//             blobs[i] = cap;
//         }
//     }
// };

struct CapabilityWeightsSeparation : ICapability {
    int64_t type;
    int64_t length;
    
    // uint64_t numSizes; // technically speaking, this is an auxiliary field since it is used only for read/write
    std::vector<uint64_t> numInits;

    CapabilityWeightsSeparation(int64_t type, std::vector<uint64_t>& numInits) : type(type), numInits(numInits) {
        length = numInits.size();
    }

    void write(std::ostream& stream) override {
        stream.write(reinterpret_cast<const char*>(&type), sizeof(type));
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));

        int64_t numSizes = numInits.size();
        stream.write(reinterpret_cast<const char*>(&numSizes), sizeof(numSizes));

        stream.write(reinterpret_cast<const char*>(numInits.data()), numSizes);
    }

    CapabilityWeightsSeparation(int64_t type, int64_t length) : type(type), length(length) {}

    void read_value(std::istream& stream) override {
        int64_t numSizes;
        stream.read(reinterpret_cast<char*>(&numSizes), sizeof(numSizes));

        // numInits.reserve(numBlobs);
        numInits.reserve(numSizes);

        stream.read(reinterpret_cast<char*>(numInits.data()), numSizes);
    }

    bool is_supported() override {
        return true;
    }
};

struct CapabilityBatchSize : ICapability {
    int64_t batchSize;

    int64_t type;
    int64_t length;

    CapabilityBatchSize(uint64_t batchSize) : batchSize(batchSize) {
        type = CAPABILITIES::BATCH_SIZE;
        length = sizeof(batchSize);
    }

    CapabilityBatchSize() {
        type = CAPABILITIES::BATCH_SIZE;
        length = sizeof(batchSize);
    }

    void write(std::ostream& stream) override {
        stream.write(reinterpret_cast<const char*>(&type), sizeof(type));
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));

        stream.write(reinterpret_cast<const char*>(&batchSize), sizeof(batchSize));
    }

    void read_value(std::istream& stream) override {
        stream.read(reinterpret_cast<char*>(&batchSize), sizeof(batchSize));
    }

    bool is_supported() override {
        return true;
    }
};

struct CapabilityInputOutputLayouts : ICapability {
    std::vector<ov::Layout> inputLayouts;
    std::vector<ov::Layout> outputLayouts;

    // some constructor here

    // TODO: discuss about having a standardized format to serialize container elements such as
    // std::vector<int> or std::vector<std::string>
    // why? probably because having methods serializePOD and serializeVector to delegate
    void write(std::ostream& stream) override {
        const uint64_t numberOfInputLayouts = inputLayouts.size();
        const uint64_t numberOfOutputLayouts = outputLayouts.size();
        stream.write(reinterpret_cast<const char*>(&numberOfInputLayouts), sizeof(numberOfInputLayouts));
        stream.write(reinterpret_cast<const char*>(&numberOfOutputLayouts), sizeof(numberOfOutputLayouts));

        const auto writeLayouts = [&](const std::optional<std::vector<ov::Layout>>& layouts) {
            if (layouts.has_value()) {
                for (const ov::Layout& layout : layouts.value()) {
                    const std::string layoutString = layout.to_string();
                    const uint16_t stringLength = static_cast<uint16_t>(layoutString.size());
                    stream.write(reinterpret_cast<const char*>(&stringLength), sizeof(stringLength));
                    stream.write(layoutString.c_str(), stringLength);
                }
            }
        };
        writeLayouts(inputLayouts);
        writeLayouts(outputLayouts);
    }

    void read_value(std::istream& stream) override {
        uint64_t numberOfInputLayouts, numberOfOutputLayouts;

        stream.read(reinterpret_cast<char*>(&numberOfInputLayouts), sizeof(numberOfInputLayouts));
        stream.read(reinterpret_cast<char*>(&numberOfOutputLayouts), sizeof(numberOfOutputLayouts));

        const auto readNLayouts = [&](const uint64_t numberOfLayouts, const char* loggerAddition) {
            std::vector<ov::Layout> layouts;
            if (!numberOfLayouts) {
                return layouts;
            }

            uint16_t stringLength;
            layouts = std::vector<ov::Layout>();
            layouts.reserve(numberOfLayouts);
            for (uint64_t layoutIndex = 0; layoutIndex < numberOfLayouts; ++layoutIndex) {
                stream.read(reinterpret_cast<char*>(&stringLength), sizeof(stringLength));

                std::string layoutString(stringLength, 0);
                stream.read(const_cast<char*>(layoutString.c_str()), stringLength);

                try {
                    layouts.push_back(ov::Layout(std::move(layoutString)));
                } catch (const ov::Exception&) {
                    // _logger.warning("Error encountered while constructing an ov::Layout object. %s index: %d. Value "
                    //                 "read from blob: %s. A default value will be used instead.",
                    //                 loggerAddition,
                    //                 layoutIndex,
                    //                 layoutString.c_str());
                    layouts.push_back(ov::Layout());
                }
            }
            return layouts;
        };

        inputLayouts = readNLayouts(numberOfInputLayouts, "Input");
        outputLayouts = readNLayouts(numberOfOutputLayouts, "Output");
    }
};

struct UnknownCapability : ICapability {
    const int type = -1;
    int length;

    explicit UnknownCapability(int length) : length(length) {}

    void write(std::ostream& stream) override {
        OPENVINO_THROW("Not supported");
    }

    UnknownCapability(int64_t length) : length(length) {}

    void read_value(std::istream& stream) override {
        OPENVINO_THROW("Not supported");
    }

    bool is_supported() override {
        return false;
    }
};

void readCapabilities(std::istream& stream, std::vector<std::shared_ptr<ICapability>>& caps) {
    int64_t tag, length;
    std::shared_ptr<ICapability> cap;

    while (stream.tellg() != std::ios::end) {
        stream.read(reinterpret_cast<char*>(tag), sizeof(tag));
        stream.read(reinterpret_cast<char*>(length), sizeof(length));

        switch (tag) {
            case ELF_BLOB:
                cap = std::make_shared<CapabilityELFBlob>(tag, length);
                cap->read_value(stream);
                caps[tag] = cap;
                break;

            case BATCH_SIZE:
                cap = std::make_shared<CapabilityBatchSize>();
                cap->read_value(stream);
                caps[tag] = cap;
                break;

            case WEIGHTS_SEPARATION:
                cap = std::make_shared<CapabilityWeightsSeparation>(tag, length);
                cap->read_value(stream);
                caps[tag] = cap;
                break;

            case OR_GROUP:
                cap = std::make_shared<OrGroup>(tag, length);
                cap->read_value(stream);
                caps[tag] = cap;
                break;

            default:
                // unknown property found
                // jump to the next one since we know the length from header
                caps[tag] = std::make_shared<UnknownCapability>(length);
                break;
        }
    }
}

bool eval() {
    for (auto cap : caps) {
        // required or not might count in decision
        // UnknownCapability::isSupported() -> always false
        if (!cap->is_supported()) {
            return false;
        }
    }
}

void write_metadata(std::ostream& stream) {
    constexpr std::string_view MAGIC_BYTES = "OVNPU";
    constexpr uint32_t METADATA_VERSION = 0x30000; // 3.0
    OpenvinoVersion ovVersion(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR, OPENVINO_VERSION_PATCH);

    stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size());
    stream.write(reinterpret_cast<const char*>(&METADATA_VERSION), sizeof(METADATA_VERSION));
    ovVersion.write(stream);
}

bool read_and_validate_metadata(std::istream& stream) {
    constexpr std::string_view MAGIC_BYTES = "OVNPU";
    constexpr uint32_t METADATA_VERSION = 0x30000; // 3.0
    OpenvinoVersion ovVersion(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR, OPENVINO_VERSION_PATCH);

    std::string magic_read;
    uint32_t meta_version_read;
    
    stream.read(reinterpret_cast<char*>(&magic_read), 5);
    if (magic_read != MAGIC_BYTES) {
        std::cout << "bad magic\n";
        return false;
    }

    stream.read(reinterpret_cast<char*>(&meta_version_read), sizeof(meta_version_read));
    if (meta_version_read != METADATA_VERSION) {
        std::cout << "bad metadata version\n";
        return false;
    }

    OpenvinoVersion ov_version_read(1, 1, 1); // dummy values since there is no default constructor
    ov_version_read.read(stream);

    if (ov_version_read != ovVersion) {
        std::cout << "bad ov version\n";
        return false;
    }
    return true;
}

// to have in mind that not always some capabilities need to be written to a blob
void write_capabilities(std::ostream& stream) {
    for (auto cap : caps) {
        cap->write(stream);
    }
}

int main() {
    std::stringstream stream;

    write_metadata(stream);

    std::vector<uint64_t> numInits = {1, 2, 3, 4};
    std::shared_ptr<CapabilityWeightsSeparation> ws_write = std::make_shared<CapabilityWeightsSeparation>(1, numInits);

    std::shared_ptr<CapabilityBatchSize> bs_write = std::make_shared<CapabilityBatchSize>(10);

    // TODO: with some prints, extract some data from a blob for CapabilityInputOutputLayouts

    caps.push_back(ws_write);
    caps.push_back(bs_write);

    write_capabilities(stream);

    // thought: is it worth having it written to a file? for testing purposes

    std::cout << stream.rdbuf() << "\n\n";
    
    read_and_validate_metadata(stream);

    caps.clear();

    readCapabilities(stream, caps);

    return 0;
}
