//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "include/model/ModelHal.hpp"
#include "include/model/UscriptHal.hpp"

#include <Firmware.hpp>
#include <common/Utils.hpp>

#include <ethosn_command_stream/CommandStreamBuilder.hpp>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

namespace ethosn
{
namespace control_unit
{
// Defined in PleKernelBinaries.hpp
extern const size_t g_PleKernelBinariesSize;
extern const uint8_t g_PleKernelBinaries[];
}    // namespace control_unit
}    // namespace ethosn

std::vector<std::string> Split(std::string s, char delim)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> results;
    while (std::getline(ss, token, delim))
    {
        results.push_back(token);
    }
    return results;
}

uint32_t HexStringToUInt(std::string s)
{
    uint32_t result;
    std::istringstream(s) >> std::hex >> result;
    return result;
}

// Executes a binary command stream from a file passed on the command-line.
// The command-line also accepts buffers (inputs, weights, etc.).
int main(int argc, char** argv)
{
    using namespace ethosn::control_unit;
    using namespace ethosn::command_stream;

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << R"( <path to binary command stream file> [arg]...

Where arg is either:
    dram,<offset>,<filename>
        Which preloads DRAM with the given .hex file at the given offset in DRAM.

    buffer,<buffer ID>,<DRAM address>,<size>
        Which binds the given buffer ID to the given DRAM address and size.

    ple,<program ID>,<DRAM address>,<DRAM size>
        Which binds the given PLE program ID to the given DRAM address.

    debug,<mask>,<verbosity>
        Which enables the model's debug printing with the given mask and verbosity

    trace,<filename>
        Which will dump a bennto API trace file to the given filename.
)";
        return 1;
    }

#if defined(CONTROL_UNIT_ASSERTS)
    utils::g_AssertCallback = utils::DefaultAssert;
#endif

    // Extract ModelHal constructor arguments from command line
    std::vector<std::string> args;
    std::string modelHalOptions;
    for (uint32_t argIdx = 2; argIdx < static_cast<uint32_t>(argc) - 1; ++argIdx)
    {
        if (strstr(argv[argIdx], "--model-options"))
        {
            modelHalOptions = argv[argIdx + 1];
        }
        else
        {
            args.push_back(argv[argIdx]);
        }
    }

    ModelHal model(ModelHal::CreateWithCmdLineOptions(modelHalOptions.c_str()));
    UscriptHal<ModelHal> uscript(model, "config.txt", true);

    // Load PLE kernel data into bennto
    constexpr uint64_t pleKernelDataAddr = 0x10000000;

    if (bennto_load_mem_array(model.GetBenntoHandle(), g_PleKernelBinaries, pleKernelDataAddr,
                              g_PleKernelBinariesSize) != BERROR_OK)
    {
        std::cerr << "Failed to load PLE kernel data" << std::endl;
        return 1;
    }

    Firmware<UscriptHal<ModelHal>> fw(uscript, pleKernelDataAddr);

    std::vector<ethosn_buffer_desc> buffers;

    // Open the file with binary command stream data and load it into memory.
    std::ifstream commandStreamFile(argv[1], std::ios_base::binary | std::ios_base::ate);
    if (!commandStreamFile.is_open())
    {
        std::cerr << "Failed to open command stream file: " << argv[1] << std::endl;
        return 1;
    }

    const uint32_t csSize = static_cast<uint32_t>(commandStreamFile.tellg());
    std::vector<uint8_t> commandStreamData(csSize);

    commandStreamFile.seekg(0);
    commandStreamFile.read(reinterpret_cast<char*>(commandStreamData.data()), static_cast<std::streampos>(csSize));

    if (commandStreamFile.fail())
    {
        std::cerr << "Failed to read command stream file: " << argv[1] << std::endl;
        return 1;
    }

    // Set the command stream as the zeroth buffer
    ethosn_buffer_desc desc = { reinterpret_cast<uint64_t>(commandStreamData.data()), csSize, ETHOSN_BUFFER_CMD_FW };
    buffers.push_back(desc);

    // Set up buffers from command-line args
    for (const std::string& arg : args)
    {
        auto options = Split(arg, ',');
        // FORMAT: "dram,%x,%s"
        if ((options[0] == "dram") && (options.size() == 3))
        {
            uint32_t dramAddressStart = HexStringToUInt(options[1]);
            if (bennto_load_mem_file(model.GetBenntoHandle(), options[2].c_str(), dramAddressStart) != BERROR_OK)
            {
                std::cerr << "Failed to load DRAM hex file: " << options[2] << std::endl;
                return 1;
            }
            uscript.RecordDramLoad(dramAddressStart, options[2]);
        }
        // FORMAT: "buffer,%x,%x,%x"
        else if ((options[0] == "buffer") && (options.size() == 5))
        {
            uint32_t bufferId         = HexStringToUInt(options[1]);
            uint32_t dramAddressStart = HexStringToUInt(options[2]);
            uint32_t size             = HexStringToUInt(options[3]);
            uint32_t bufferType       = HexStringToUInt(options[4]);
            ASSERT_MSG(bufferType < ETHOSN_BUFFER_MAX, "Wrong buffertype: %d, expected 0-%d", bufferType,
                       ETHOSN_BUFFER_MAX - 1);
            buffers.resize(std::max<size_t>(buffers.size(), 1 + bufferId), { 0U, 0U, ETHOSN_BUFFER_CMD_FW });
            buffers[bufferId] = { dramAddressStart, size, bufferType };
        }
        else
        {
            std::cerr << "Invalid argument: " << arg.c_str() << std::endl;
            return 1;
        }
    }

    uint32_t i = 0;
    for (const ethosn_buffer_desc& bInfo : buffers)
    {
        if (bInfo.size == 0)
        {
            std::cerr << "Missing BufferInfo " << i << " on command line" << std::endl;
            return 1;
        }
        ++i;
    }

    const uint32_t numBuffers = static_cast<uint32_t>(buffers.size());

    std::vector<uint32_t> inferenceData;

    ethosn_buffer_array bufferArray;
    bufferArray.num_buffers = numBuffers;
    EmplaceBack<ethosn_buffer_array>(inferenceData, bufferArray);

    for (const ethosn_buffer_desc& bufInfo : buffers)
    {
        EmplaceBack(inferenceData, bufInfo);
    }

    auto& inference = reinterpret_cast<const Inference&>(*inferenceData.data());

    model.ClearSram();

    if (!fw.RunInference(inference).success)
    {
        std::cerr << "Failed to execute command stream" << std::endl;
        return 1;
    }

    return 0;
}
