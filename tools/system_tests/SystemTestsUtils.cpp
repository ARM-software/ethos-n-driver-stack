//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SystemTestsUtils.hpp"

#include <ethosn_utils/System.hpp>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <utility>
#if defined __unix__
#include "../../kernel-module/tests/ethosn-tests-uapi.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#endif

#if defined(__unix__)
#include <linux/dma-buf.h>
#include <linux/version.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#include <linux/dma-heap.h>
#endif
#elif defined(_MSC_VER)
#include <io.h>
// Alias for ssize_t so that we can write code with the correct types for Linux, and it also
// works for Windows. On Windows, read() and write() return int whereas on Linux they return
// ssize_t.
using ssize_t = int;
#define KERNEL_VERSION(a, b, c) 0
#endif

using namespace ethosn::driver_library;

namespace ethosn
{
namespace system_tests
{

utils::log::Logger<utils::log::Severity::Debug> g_Logger;

template <typename T>
Stats::Stats(const std::vector<T>& data)
    : Stats()
{
    m_DataTypeMin = std::numeric_limits<T>::lowest();
    m_DataTypeMax = std::numeric_limits<T>::max();

    uint64_t total        = 0;
    uint64_t totalSquared = 0;
    for (T x : data)
    {
        m_Frequencies[x]++;
        total += x;
        totalSquared += static_cast<int64_t>(x) * static_cast<int64_t>(x);
        m_Count++;
        m_Max = std::max<int64_t>(m_Max, x);
    }

    m_Mean              = static_cast<float>(total) / static_cast<float>(m_Count);
    m_Variance          = static_cast<float>(totalSquared) / (static_cast<float>(m_Count) - m_Mean * m_Mean);
    m_StandardDeviation = sqrtf(m_Variance);

    // Calculate mode and median
    size_t modeFrequency   = 0;
    size_t cumulativeIndex = 0;
    for (auto vf : m_Frequencies)
    {
        int64_t value = vf.first;
        size_t freq   = vf.second;
        if (freq > modeFrequency)
        {
            m_Mode        = static_cast<float>(value);
            modeFrequency = freq;
        }

        if (cumulativeIndex < m_Count / 2 && cumulativeIndex + freq > m_Count / 2)
        {
            m_Median = static_cast<float>(value);
        }
        cumulativeIndex += freq;
    }
}

void Stats::PrintHistogram(std::ostream& stream)
{
    // Group frequencies into 16 buckets
    constexpr uint32_t numBuckets            = 16;
    const int64_t bucketSize                 = (m_DataTypeMax - m_DataTypeMin + 1) / numBuckets;
    std::array<size_t, numBuckets> histogram = {};
    size_t maxBucketSize                     = 0;
    for (auto vf : m_Frequencies)
    {
        int64_t value = vf.first;
        size_t freq   = vf.second;
        size_t bucket = (value - m_DataTypeMin) / bucketSize;
        histogram[bucket] += freq;
        maxBucketSize = std::max(maxBucketSize, histogram[bucket]);
    }

    // If we dont protect against maxBucketSize == 0, we could end up dividing by zero later on
    if (maxBucketSize == 0)
    {
        stream << "ERROR: NO HISTOGRAM DATA DETECTED" << std::endl;
        return;
    }

    int64_t min = m_DataTypeMin;
    int64_t max = m_DataTypeMin + bucketSize - 1;
    for (size_t b : histogram)
    {
        stream << std::setw(4) << min << " - " << std::setw(4) << max << ": " << std::setw(5) << b << " "
               << std::string((20 * b + maxBucketSize - 1) / maxBucketSize, '#') << std::endl;
        min += bucketSize;
        max += bucketSize;
    }
}

bool CompareTensors(const BaseTensor& a, const BaseTensor& b, float tolerance)
{
    if (a.GetDataType() != b.GetDataType())
    {
        throw std::invalid_argument("Data types must match");
    }

    switch (a.GetDataType())
    {
        case DataType::S8:
            return CompareArrays(a.GetData<int8_t>(), b.GetData<int8_t>(), tolerance);
        case DataType::U8:
            return CompareArrays(a.GetData<uint8_t>(), b.GetData<uint8_t>(), tolerance);
        case DataType::S32:
            return CompareArrays(a.GetData<int32_t>(), b.GetData<int32_t>(), tolerance);
        case DataType::F32:
            return CompareArrays(a.GetData<float>(), b.GetData<float>(), tolerance);
        default:
            throw std::invalid_argument("Unknown datatype");
    }
}

template <typename T>
std::string DumpOutputToFiles(const std::vector<T>& output,
                              const std::vector<T>& refOutput,
                              const std::string& filePrefix,
                              const std::string& outputName,
                              size_t runNumber)
{
    // Remove all forward/backslashes in string so we can save to a file.
    std::string formattedOutputName = outputName;
    std::replace_if(formattedOutputName.begin(), formattedOutputName.end(),
                    [](const char c) { return c == '/' || c == '\\'; }, '-');

    std::string referenceOutputFilename = std::string(filePrefix) + "-run0-" + formattedOutputName + ".hex";
    std::string actualOutputFilename =
        std::string(filePrefix) + "-run" + std::to_string(runNumber) + "-" + formattedOutputName + ".hex";

    std::cout << "Histogram of differences for output mismatch " << formattedOutputName << ":" << std::endl;
    const std::vector<T> absDiff = GetAbsoluteDifferences(output, refOutput);
    Stats differenceStats(absDiff);
    differenceStats.PrintHistogram(std::cout);

    DumpData(referenceOutputFilename.c_str(), refOutput);
    DumpData(actualOutputFilename.c_str(), output);

    std::string res = "Mismatch in output from run 0 and run " + std::to_string(runNumber) +
                      ". See above histogram of differences.\nSee files to compare: " + referenceOutputFilename + " " +
                      actualOutputFilename + "\n";

    return res;
}

template <typename T>
std::string DumpFiles(const std::vector<T>& ethosn, const std::vector<T>& cpu, std::string& outputName, float tolerance)
{
    // Remove all forward/backslashes in string so we can save to a file.
    std::replace(outputName.begin(), outputName.end(), '/', '-');
    std::replace(outputName.begin(), outputName.end(), '\\', '-');
    std::string referenceOutputFilename = std::string("armnn-") + outputName + ".hex";
    std::string actualOutputFilename    = std::string("ethosn-") + outputName + ".hex";
    // Produce absolute difference above tolerance and zeros
    // so they can be diff'ed to see where the errors are
    std::string absdiffFilename = std::string("absdiff-") + outputName + ".hex";
    const char* zerosFilename   = "zeros.hex";

    const std::vector<T> absDiff = GetAbsoluteDifferences(ethosn, cpu);

    std::vector<T> aux(absDiff.size(), 0);
    DumpData(zerosFilename, aux);

    const int iTolerance            = static_cast<int>(tolerance);
    const auto removeToleratedError = [iTolerance](const T v) { return (v > iTolerance) ? v : 0; };
    std::transform(absDiff.begin(), absDiff.end(), aux.begin(), removeToleratedError);

    std::cout << "Histogram of differences for output " << outputName << ":" << std::endl;

    Stats differenceStats(absDiff);
    differenceStats.PrintHistogram(std::cout);

    DumpData(referenceOutputFilename.c_str(), cpu);
    DumpData(actualOutputFilename.c_str(), ethosn);
    DumpData(absdiffFilename.c_str(), aux);

    std::string res = "Output " + outputName + " mismatch. Max difference is " +
                      std::to_string(static_cast<int>(differenceStats.m_Max)) +
                      ". See above histogram of differences." + "\nSee files to compare: " + referenceOutputFilename +
                      " " + actualOutputFilename +
                      "\nCompare files to see differences above tolerance: " + absdiffFilename + " " + zerosFilename;

    return res;
}

template <>
std::string DumpFiles<float>(const std::vector<float>& ethosn,
                             const std::vector<float>& cpu,
                             std::string& outputName,
                             float tolerance)
{
    // Remove all forward/backslashes in string so we can save to a file.
    std::replace(outputName.begin(), outputName.end(), '/', '-');
    std::replace(outputName.begin(), outputName.end(), '\\', '-');
    std::string referenceOutputFilename = std::string("armnn-") + outputName + ".hex";
    std::string actualOutputFilename    = std::string("ethosn-") + outputName + ".hex";
    // Produce absolute difference above tolerance and zeros
    // so they can be diff'ed to see where the errors are
    std::string absdiffFilename = std::string("absdiff-") + outputName + ".hex";
    const char* zerosFilename   = "zeros.hex";

    const std::vector<float> absDiff = GetAbsoluteDifferences(ethosn, cpu);

    std::vector<float> aux(absDiff.size(), 0);
    DumpData(zerosFilename, aux);

    const auto removeToleratedError = [tolerance](const float v) { return (v > tolerance) ? v : 0; };
    std::transform(absDiff.begin(), absDiff.end(), aux.begin(), removeToleratedError);

    const float maxDifference = *std::max_element(aux.begin(), aux.end());

    DumpData(referenceOutputFilename.c_str(), cpu);
    DumpData(actualOutputFilename.c_str(), ethosn);
    DumpData(absdiffFilename.c_str(), aux);

    std::string res = "Output " + outputName + " mismatch. Max difference is " + std::to_string(maxDifference) + "." +
                      "\nSee files to compare: " + referenceOutputFilename + " " + actualOutputFilename +
                      "\nCompare files to see differences above tolerance: " + absdiffFilename + " " + zerosFilename;

    return res;
}

std::string DumpOutputToFiles(const BaseTensor& output,
                              const BaseTensor& refOutput,
                              const std::string& filePrefix,
                              const std::string& outputName,
                              size_t runNumber)
{
    if (output.GetDataType() != refOutput.GetDataType())
    {
        throw std::invalid_argument("Output data types must match");
    }

    switch (output.GetDataType())
    {
        case DataType::U8:
            return DumpOutputToFiles(output.GetData<uint8_t>(), refOutput.GetData<uint8_t>(), filePrefix, outputName,
                                     runNumber);
        case DataType::S8:
            return DumpOutputToFiles(output.GetData<int8_t>(), refOutput.GetData<int8_t>(), filePrefix, outputName,
                                     runNumber);
        default:
            throw std::invalid_argument("Unknown output data type");
    }
}

std::string DumpFiles(const BaseTensor& ethosn, const BaseTensor& cpu, std::string& outputName, float tolerance)
{
    if (ethosn.GetDataType() != cpu.GetDataType())
    {
        throw std::invalid_argument("Data types must match");
    }

    switch (ethosn.GetDataType())
    {
        case DataType::U8:
            return DumpFiles(ethosn.GetData<uint8_t>(), cpu.GetData<uint8_t>(), outputName, tolerance);
        case DataType::S8:
            return DumpFiles(ethosn.GetData<int8_t>(), cpu.GetData<int8_t>(), outputName, tolerance);
        case DataType::S32:
            return DumpFiles(ethosn.GetData<int32_t>(), cpu.GetData<int32_t>(), outputName, tolerance);
        case DataType::F32:
            return DumpFiles(ethosn.GetData<float>(), cpu.GetData<float>(), outputName, tolerance);
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

void DebugTensor(const char* const name, const BaseTensor& tensor, const size_t max)
{
    switch (tensor.GetDataType())
    {
        case DataType::U8:
            DebugVector(name, tensor.GetData<uint8_t>(), max);
            break;
        case DataType::S8:
            DebugVector(name, tensor.GetData<int8_t>(), max);
            break;
        case DataType::S32:
            DebugVector(name, tensor.GetData<int32_t>(), max);
            break;
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

ethosn::system_tests::OwnedTensor
    ConvertNhwcToNhwcb(const BaseTensor& nhwcInPtr, uint32_t tensorHeight, uint32_t tensorWidth, uint32_t tensorDepth)
{
    OwnedTensor result = MakeTensor(nhwcInPtr.GetDataType(), GetTotalSizeNhwcb(tensorWidth, tensorHeight, tensorDepth));
    switch (nhwcInPtr.GetDataType())
    {
        case DataType::S8:
            ConvertNhwcToNhwcb(nhwcInPtr.GetDataPtr<int8_t>(), result->GetDataPtr<int8_t>(), tensorHeight, tensorWidth,
                               tensorDepth);
            break;
        case DataType::U8:
            ConvertNhwcToNhwcb(nhwcInPtr.GetDataPtr<uint8_t>(), result->GetDataPtr<uint8_t>(), tensorHeight,
                               tensorWidth, tensorDepth);
            break;
        case DataType::S32:
            ConvertNhwcToNhwcb(nhwcInPtr.GetDataPtr<int32_t>(), result->GetDataPtr<int32_t>(), tensorHeight,
                               tensorWidth, tensorDepth);
            break;
        default:
            throw std::invalid_argument("Unknown datatype");
    }
    return result;
}

ethosn::system_tests::OwnedTensor
    ConvertNhwcbToNhwc(const BaseTensor& nhwcbInPtr, uint32_t tensorHeight, uint32_t tensorWidth, uint32_t tensorDepth)
{
    OwnedTensor result = MakeTensor(nhwcbInPtr.GetDataType(), tensorWidth * tensorHeight * tensorDepth);
    switch (nhwcbInPtr.GetDataType())
    {
        case DataType::S8:
            ConvertNhwcbToNhwc(nhwcbInPtr.GetDataPtr<int8_t>(), result->GetDataPtr<int8_t>(), tensorHeight, tensorWidth,
                               tensorDepth);
            break;
        case DataType::U8:
            ConvertNhwcbToNhwc(nhwcbInPtr.GetDataPtr<uint8_t>(), result->GetDataPtr<uint8_t>(), tensorHeight,
                               tensorWidth, tensorDepth);
            break;
        case DataType::S32:
            ConvertNhwcbToNhwc(nhwcbInPtr.GetDataPtr<int32_t>(), result->GetDataPtr<int32_t>(), tensorHeight,
                               tensorWidth, tensorDepth);
            break;
        default:
            throw std::invalid_argument("Unknown datatype");
    }
    return result;
}

bool DumpData(const char* filename, const BaseTensor& t)
{
    switch (t.GetDataType())
    {
        case DataType::U8:
            return DumpData(filename, t.GetData<uint8_t>());
        case DataType::S8:
            return DumpData(filename, t.GetData<int8_t>());
        case DataType::S32:
            return DumpData(filename, t.GetData<int32_t>());
        case DataType::F32:
            return DumpData(filename, t.GetData<float>());
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

void CopyBuffers(const std::vector<ethosn::driver_library::Buffer*>& sourceBuffers,
                 const std::vector<uint8_t*>& destPointers)
{
    assert(sourceBuffers.size() == destPointers.size());

    std::vector<uint8_t*>::const_iterator destPointersIt = destPointers.begin();
    for (Buffer* sourceBuffer : sourceBuffers)
    {
        uint8_t* sourceBufferData = sourceBuffer->Map();
        std::copy(sourceBufferData, sourceBufferData + sourceBuffer->GetSize(), *destPointersIt);
        sourceBuffer->Unmap();
        ++destPointersIt;
    }
}

bool IsStatisticalOutputGood(const MultipleInferenceOutputs& output)
{
    for (uint32_t k = 0; k < output.size(); ++k)
    {
        if (!IsStatisticalOutputGood(output[k]))
            return false;
    }
    return true;
}

template <typename T>
bool IsStatisticalOutputGood(const std::vector<T>& data, std::string name)
{
    // Analyse the distribution of the outputs to make sure the test case is valid (e.g. not all 0xFF)
    Stats stats(data);
    // The simplest check would be to make sure the standard deviation is above a threshold,
    // but this would fail to catch cases where all the values are 0 or 255.
    // To catch this, we count the number of unique values that appear a 'reasonable' number of times in the output
    // and make sure there are 'enough' of these.
    size_t uniqueValues = std::count_if(stats.m_Frequencies.begin(), stats.m_Frequencies.end(), [&](auto vf) {
        return static_cast<float>(vf.second) / static_cast<float>(stats.m_Count) > (1.0f / 512.0f);
    });
    // Ideally all 255 values would be present but this is too restrictive, so we settle for 20.
    // However if there are not enough values then it is not reasonable to expect 20,
    // so we scale the threshold down with the number of values, allowing about 10 duplicates per unique value.
    // For very small quantities though this would allow all values being the same which is not good,
    // so for these we require each value to be unique.
    // clang-format off
    size_t requiredUniqueValues = stats.m_Count <= 5 ? stats.m_Count :
        stats.m_Count <= 200 ? 5 + (stats.m_Count - 5) / 10 :
        20u;
    // clang-format on
    if (uniqueValues < requiredUniqueValues)
    {
        std::cout << "Histogram of " << std::move(name) << ":" << std::endl;
        stats.PrintHistogram(std::cout);
        std::cout << uniqueValues << " significantly unique values." << std::endl;
        return false;
    }
    return true;
}

bool IsStatisticalOutputGood(const InferenceOutputs& output)
{
    for (uint32_t k = 0; k < output.size(); ++k)
    {
        bool good;
        switch (output[k]->GetDataType())
        {
            case DataType::S8:
                good = IsStatisticalOutputGood(output[k]->GetData<int8_t>(), "reference output " + std::to_string(k));
                break;
            case DataType::U8:
                good = IsStatisticalOutputGood(output[k]->GetData<uint8_t>(), "reference output " + std::to_string(k));
                break;
            default:
                throw std::exception();
        }

        if (!good)
        {
            return false;
        }
    }
    return true;
}

std::string GetCacheFilename(const std::string& sourceFilename, const std::string& cacheFolderOverride)
{
    std::string baseName;
    std::string sourceFolder;
    size_t lastDirSepIdx = sourceFilename.find_last_of("/\\");
    if (lastDirSepIdx == std::string::npos)
    {
        baseName = sourceFilename;
    }
    else
    {
        baseName     = sourceFilename.substr(lastDirSepIdx + 1);
        sourceFolder = sourceFilename.substr(0, lastDirSepIdx + 1);
    }
    std::string rootFolder = cacheFolderOverride.size() > 0 ? cacheFolderOverride : (sourceFolder + "armnn-cache");
    return rootFolder + "/" + baseName + ".armnn";
}

InferenceOutputs RunNetworkCached(const std::string& cacheFilename, std::function<InferenceInputs()> runNetworkFunc)
{
    InferenceOutputs output;

    // Read cached output if provided
    if (cacheFilename.size() > 0)
    {
        bool isLittleEndian = IsLittleEndian();
        ETHOSN_UNUSED(isLittleEndian);
        assert(isLittleEndian);
        std::ifstream ifs(cacheFilename, std::ios_base::binary | std::ios_base::in);
        if (ifs.is_open())
        {
            uint64_t numOutputs;
            ifs.read(reinterpret_cast<char*>(&numOutputs), sizeof(numOutputs));
            size_t cacheSize = sizeof(uint64_t) + numOutputs * (sizeof(uint64_t) + sizeof(uint8_t));
            std::vector<char> cacheHeader(cacheSize);

            ifs.seekg(0, std::ios_base::beg);
            ifs.read(cacheHeader.data(), cacheSize);

            output = GetOutputTensorsFromCache(cacheHeader);
            assert(output.size() == numOutputs);
            for (uint64_t i = 0; i < numOutputs; ++i)
            {
                ifs.read(reinterpret_cast<char*>(output[i]->GetByteData()), output[i]->GetNumBytes());
            }
        }
        else
        {
            std::cout << "Failed to open Arm NN cache file: " << cacheFilename << std::endl;
        }
    }

    // Run the Arm NN network, but only if we didn't load a cached result above
    if (output.size() == 0)
    {
        output = runNetworkFunc();
    }
    else
    {
        std::cout << "Using cached Arm NN output from " << cacheFilename << ". Beware this may be stale." << std::endl;
    }

    // Save the cached data for next time if path is provided and supported
    if (cacheFilename.size() > 0)
    {
        bool isLittleEndian = IsLittleEndian();
        ETHOSN_UNUSED(isLittleEndian);
        assert(isLittleEndian);
        size_t lastDirSepIdx    = cacheFilename.find_last_of("/\\");
        std::string cacheFolder = cacheFilename.substr(0, lastDirSepIdx + 1);
#if defined(__unix__)
        // Create the cache folder if it doesn't exist, otherwise saving to it will fail.
        mkdir(cacheFolder.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
        std::ofstream ofs(cacheFilename, std::ios_base::binary | std::ios_base::trunc);
        if (ofs.is_open())
        {
            std::vector<char> header = CreateCacheHeader(output);
            ofs.write(header.data(), header.size());
            for (uint32_t i = 0; i < output.size(); ++i)
            {
                const uint8_t* data = output[i]->GetByteData();
                ofs.write(reinterpret_cast<const char*>(data), output[i]->GetNumBytes());
            }
        }
        else
        {
            std::cout << "Failed to write Arm NN cache file: " << cacheFilename << std::endl;
        }
    }

    return output;
}

bool IsDataTypeSigned(DataType dataType)
{
    switch (dataType)
    {
        case DataType::S8:
            return true;
        case DataType::U8:
            return false;
        case DataType::S32:
            return true;
        default:
        {
            std::string errorMessage = "Error in " + std::string(__func__) + ": DataType is not supported";
            throw std::invalid_argument(errorMessage);
        }
    }
}

uint32_t CalcUpsampleOutputSize(const ScaleParams& params, const uint32_t inputSize)
{
    if (params.m_Size != 0U)
    {
        return params.m_Size;
    }
    else
    {
        uint32_t size = static_cast<uint32_t>(params.m_Ratio * static_cast<float>(inputSize));
        if (size == 0U)
        {
            throw std::invalid_argument("Upsample output size is zero.");
        }
        return (params.m_Mode == ResizeMode::DROP) ? --size : size;
    }
}

std::vector<char> CreateCacheHeader(const InferenceOutputs& outputs)
{
    std::vector<char> ret;
    // 64 bits for the number of outputs, 64 bits for the size of each output, 8 bits for the type of each output
    size_t size = sizeof(uint64_t) + outputs.size() * (sizeof(uint64_t) + sizeof(uint8_t));
    ret.resize(size);
    // Write a header taining the number of outputs, the size and the data type of each output
    // e.g. For a network with the following 3 outputs with size and datatype:
    // (1x1x1x16, U8), (1x1x1x2, S8), (1x1x1x1, F32)
    // The header contains the following bytes in little endian format:
    // 03 00 00 00 00 00 00 00 (3 outputs encoded in 64 bits)
    // 10 00 00 00 00 00 00 00 (16 bytes size encoded in 64 bits)
    // 00                      (U8 type encoded as 0 in 8 bits)
    // 02 00 00 00 00 00 00 00 (2 byte size encoded in 64 bits)
    // 01                      (S8 type encoded as 1 in 8 bits)
    // 04 00 00 00 00 00 00 00 (4 byte size encoded in 64 bits)
    // 03                      (F32 type encoded as 3 in 8 bits)
    uint64_t outputSize = outputs.size();
    std::copy(reinterpret_cast<char*>(&outputSize), reinterpret_cast<char*>(&outputSize) + sizeof(uint64_t),
              ret.data());

    char* endPos = ret.data() + sizeof(uint64_t);
    for (uint32_t i = 0; i < outputs.size(); ++i)
    {
        uint64_t numBytes = outputs[i]->GetNumBytes();
        std::copy(reinterpret_cast<char*>(&numBytes), reinterpret_cast<char*>(&numBytes) + sizeof(uint64_t), endPos);
        endPos += sizeof(uint64_t);
        uint8_t type = static_cast<uint8_t>(outputs[i]->GetDataType());
        std::copy(reinterpret_cast<char*>(&type), reinterpret_cast<char*>(&type) + sizeof(uint8_t), endPos);
        endPos += sizeof(uint8_t);
    }
    return ret;
}

InferenceOutputs GetOutputTensorsFromCache(std::vector<char>& cacheHeader)
{
    InferenceOutputs ret;

    uint64_t numOutputs;
    std::copy_n(cacheHeader.data(), sizeof(uint64_t), reinterpret_cast<char*>(&numOutputs));
    ret.resize(numOutputs);
    size_t offset = sizeof(uint64_t);
    for (size_t i = 0; i < numOutputs; ++i)
    {
        uint64_t numBytes;
        std::copy_n(cacheHeader.data() + offset, sizeof(uint64_t), reinterpret_cast<char*>(&numBytes));
        offset += sizeof(uint64_t);
        uint8_t type;
        std::copy_n(cacheHeader.data() + offset, sizeof(uint8_t), reinterpret_cast<char*>(&type));
        offset += sizeof(uint8_t);
        DataType dataType = static_cast<DataType>(type);
        ret[i]            = MakeTensor(dataType, numBytes / static_cast<uint64_t>(GetNumBytes(dataType)));
    }
    return ret;
}

// float specialization of GetAbsoluteDifferences (see header file).
template <>
std::vector<float> GetAbsoluteDifferences<float>(const std::vector<float>& a, const std::vector<float>& b)
{
    size_t size = std::min(a.size(), b.size());
    std::vector<float> differences(size, 0);
    for (uint32_t i = 0; i < size; ++i)
    {
        differences[i] = std::abs(a[i] - b[i]);
    }
    return differences;
}

template <>
bool CompareArrays<float>(const std::vector<float>& a, const std::vector<float>& b, float tolerance)
{
    if (a.size() != b.size())
    {
        return false;
    }
    std::vector<float> differences = GetAbsoluteDifferences(a, b);
    float maxDifference            = *std::max_element(differences.begin(), differences.end());
    return (maxDifference <= tolerance);

    return false;
}

void BlockInferenceTest()
{
#if defined __unix__
    int ethosnTest = open("/dev/ethosn-tests", O_RDONLY);

    if (ethosnTest >= 0)
    {
        ioctl(ethosnTest, ETHOS_N_TEST_IOCTL_BLOCK_INFERENCES, NULL);
    }
#endif
}

std::vector<uint32_t> GetBinaryDataFromHexFile(std::istream& input, uint32_t startAddress, uint32_t lengthBytes)
{
    input.clear();
    input.seekg(0);
    std::vector<uint32_t> out;
    assert(startAddress % 4 == 0);
    assert(lengthBytes % 4 == 0);
    out.reserve(lengthBytes / 4);
    uint32_t endAddress = startAddress + lengthBytes;

    // Get the addresses of the lines which contains the start and end addresses
    uint32_t startLine = startAddress & ~uint32_t{ 16 - 1 };
    uint32_t endLine   = endAddress & ~uint32_t{ 16 - 1 };

    for (std::string line; std::getline(input, line);)
    {
        uint32_t addr;
        std::array<uint32_t, 4> words;
        // Format of Combined Memory Map hex file lines
        const char* formatString = "%" SCNx32 ": %8" SCNx32 " %8" SCNx32 " %8" SCNx32 " %8" SCNx32;
        // cppcheck-suppress wrongPrintfScanfArgNum
        if (sscanf(line.c_str(), formatString, &addr, &words[0], &words[1], &words[2], &words[3]) != 5)
        {
            throw std::runtime_error("Unable to parse data field in Memory Map file");
        }
        if (addr < startLine)
        {
            continue;
        }
        if (addr > endLine)
        {
            break;
        }
        for (uint32_t i = 0; i < 4; ++i)
        {
            uint32_t currentAddr = addr + i * static_cast<uint32_t>(sizeof(uint32_t));
            if (currentAddr < startAddress || currentAddr >= endAddress)
            {
                continue;
            }
            out.push_back(words[i]);
        }
    }
    return out;
}

OwnedTensor LoadTensorFromHexStream(std::istream& input, DataType dataType, size_t numElements)
{
    OwnedTensor result      = MakeTensor(dataType, numElements);
    std::vector<uint32_t> d = GetBinaryDataFromHexFile(input, 0, result->GetNumBytes());
    assert(d.size() == DivRoundUp(result->GetNumBytes(), 4));

    for (uint32_t i = 0; i < DivRoundUp(result->GetNumBytes(), 4); ++i)
    {
        reinterpret_cast<uint32_t*>(result->GetByteData())[i] = d[i];
    }

    return result;
}

OwnedTensor LoadTensorFromBinaryStream(std::istream& input, DataType dataType, size_t numElements)
{
    OwnedTensor result = MakeTensor(dataType, numElements);
    input.read(reinterpret_cast<char*>(result->GetByteData()), result->GetNumBytes());

    if (input.eof())
    {
        g_Logger.Error("Input image is smaller than tensor size");
        return nullptr;
    }

    return result;
}

float GetReferenceComparisonTolerance(const std::map<std::string, float>& referenceComparisonTolerances,
                                      const std::string& outputName)
{
    // First lookup using the exact name
    auto it = referenceComparisonTolerances.find(outputName);
    if (it == referenceComparisonTolerances.end())
    {
        // If that failed, try the special name "*"
        it = referenceComparisonTolerances.find("*");
        if (it == referenceComparisonTolerances.end())
        {
            throw std::runtime_error("No reference comparison tolerance provided for output " + outputName);
        }
    }
    return it->second;
}

DmaBufferDevice::DmaBufferDevice(DmaBufferDevice&& otherDmaBufferDevice)
    : m_DevFd(std::exchange(otherDmaBufferDevice.m_DevFd, -EINVAL))
{}

DmaBufferDevice::DmaBufferDevice(const char* dmaBufferDeviceFile)
    : m_DevFd(0)
{
#if defined(TARGET_KMOD)
    m_DevFd = open(dmaBufferDeviceFile, O_RDONLY | O_CLOEXEC);
    if (m_DevFd < 0)
    {
        throw std::runtime_error(std::string("Failed to open ") + dmaBufferDeviceFile +
                                 " to get dma_buf memory. You need to have access!");
    }
#elif defined(TARGET_MODEL)
    ETHOSN_UNUSED(dmaBufferDeviceFile);
    // Set the heap fd to max for debugging. We don't need it if we don't have a dma heap buffer
    m_DevFd         = std::numeric_limits<int>::max();
#endif
}

DmaBufferDevice::~DmaBufferDevice()
{
    if (m_DevFd >= 0)
    {
#if defined(TARGET_KMOD)
        close(m_DevFd);
#endif
        m_DevFd = -EINVAL;
    }
}

int DmaBufferDevice::GetFd() const
{
    if (m_DevFd < 0)
    {
        throw std::runtime_error("File descriptor doesn't exist");
    }
    return m_DevFd;
}

DmaBuffer::DmaBuffer()
    : m_DmaBufFd(-EINVAL)
    , m_Size(0)
{}

DmaBuffer::DmaBuffer(DmaBuffer&& otherDmaBuffer)
    : m_DmaBufFd(std::exchange(otherDmaBuffer.m_DmaBufFd, -EINVAL))
    , m_Size(std::exchange(otherDmaBuffer.m_Size, 0))
{}

DmaBuffer::DmaBuffer(const std::unique_ptr<DmaBufferDevice>& dmaBufHeap, size_t len)
    : m_DmaBufFd(-EINVAL)
    , m_Size(0)
{
#if defined(TARGET_KMOD) && LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
    if (dmaBufHeap == nullptr)
    {
        throw std::runtime_error("Supplied dmaBufHeap object is a nullptr");
    }

    int heapFd = dmaBufHeap->GetFd();

    struct dma_heap_allocation_data heapData = {
        .len        = len,
        .fd         = 0,
        .fd_flags   = O_RDWR | O_CLOEXEC,
        .heap_flags = 0,
    };

    int ret = ioctl(heapFd, DMA_HEAP_IOCTL_ALLOC, &heapData);
    if (ret < 0)
    {
        throw std::runtime_error("Failed to allocate dma_buf from DMA heap");
    }

    m_DmaBufFd = heapData.fd;
    m_Size     = len;
#elif defined(TARGET_MODEL)
    std::FILE* file = std::tmpfile();
    m_DmaBufFd      = fileno(file);
    m_Size          = len;
    // Initialize the buffer with all zeroes for ease of use.
    std::vector<uint8_t> inputData(len, 0);
    PopulateData(inputData.data(), len);
    ETHOSN_UNUSED(dmaBufHeap);
#else
    ETHOSN_UNUSED(dmaBufHeap);
    ETHOSN_UNUSED(len);
    throw std::runtime_error(
        "dma heap needs either Linux kernel version >= 5.6 when targetting the kmod backend, or the model backend");
#endif
}

DmaBuffer::~DmaBuffer()
{
    if (m_DmaBufFd >= 0)
    {
#if defined(TARGET_KMOD)
        close(m_DmaBufFd);
#endif
        m_DmaBufFd = -EINVAL;
    }
}

DmaBuffer& DmaBuffer::operator=(DmaBuffer&& otherDmaBuffer)
{
    if (m_DmaBufFd >= 0)
    {
#if defined(TARGET_KMOD)
        close(m_DmaBufFd);
#endif
        m_DmaBufFd = -EINVAL;
    }
    m_DmaBufFd                = otherDmaBuffer.GetFd();
    m_Size                    = otherDmaBuffer.m_Size;
    otherDmaBuffer.m_DmaBufFd = -EINVAL;
    otherDmaBuffer.m_Size     = 0;

    return *this;
}

int DmaBuffer::GetFd() const
{
    if (m_DmaBufFd < 0)
    {
        throw std::runtime_error("File descriptor for dma_buf heap area was not correct when DmaBuffer::GetFd()");
    }
    return m_DmaBufFd;
}

size_t DmaBuffer::GetSize() const
{
    if (m_DmaBufFd < 0)
    {
        throw std::runtime_error("File descriptor for dma_buf heap area was not correct when DmaBuffer::GetSize()");
    }
    return m_Size;
}

void DmaBuffer::PopulateData(const uint8_t* inData, size_t len)
{
#if defined(TARGET_KMOD)
    if (m_DmaBufFd < 0)
    {
        throw std::runtime_error(
            "File descriptor for dma_buf heap area was not correct when DmaBuffer::PopulateData()");
    }

    if (len > m_Size)
    {
        throw std::runtime_error("Supplied len is greater then size of the buffer when DmaBuffer::PopulateData()");
    }

    uint8_t* inputDmaBufData = static_cast<uint8_t*>(mmap(NULL, m_Size, PROT_WRITE, MAP_SHARED, m_DmaBufFd, 0));
    if (inputDmaBufData == MAP_FAILED)
    {
        throw std::runtime_error("Failed to mmap dma_buf");
    }

    dma_buf_sync syncStruct = { DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE };
    int result              = ioctl(m_DmaBufFd, DMA_BUF_IOCTL_SYNC, &syncStruct);
    if (result < 0)
    {
        throw std::runtime_error("Failed DMA_BUF_IOCTL_SYNC");
    }

    size_t sizeToCopy = std::min(len, m_Size);
    std::memcpy(inputDmaBufData, inData, sizeToCopy);

    syncStruct = { DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE };
    result     = ioctl(m_DmaBufFd, DMA_BUF_IOCTL_SYNC, &syncStruct);
    if (result < 0)
    {
        throw std::runtime_error("Failed DMA_BUF_IOCTL_SYNC");
    }

    munmap(inputDmaBufData, m_Size);
#elif defined(TARGET_MODEL)
    if (lseek(m_DmaBufFd, 0, SEEK_SET) < 0)
    {
        int err = errno;
        throw std::runtime_error("DmaBuffer lseek failed. errno = " + std::to_string(err) + ": " + std::strerror(err));
    }
    ssize_t numBytesWritten = write(m_DmaBufFd, inData, len);
    if (numBytesWritten < 0)
    {
        int err = errno;
        throw std::runtime_error("DmaBuffer write failed. errno = " + std::to_string(err) + ": " + std::strerror(err));
    }
    if (static_cast<size_t>(numBytesWritten) != len)
    {
        throw std::runtime_error("DmaBuffer asked to write " + std::to_string(len) + " but only wrote " +
                                 std::to_string(numBytesWritten));
    }
    if (lseek(m_DmaBufFd, 0, SEEK_SET) < 0)
    {
        int err = errno;
        throw std::runtime_error("DmaBuffer lseek failed. errno = " + std::to_string(err) + ": " + std::strerror(err));
    }
#endif
}

void DmaBuffer::RetrieveData(uint8_t* outData, size_t len)
{
#if defined(TARGET_KMOD)
    if (m_DmaBufFd < 0)
    {
        throw std::runtime_error(
            "File descriptor for dma_buf heap area was not correct when DmaBuffer::RetrieveData()");
    }

    if (len > m_Size)
    {
        throw std::runtime_error("Supplied len is greater then size of the buffer when DmaBuffer::RetrieveData()");
    }

    uint8_t* mappedBuffer = static_cast<uint8_t*>(mmap(NULL, m_Size, PROT_READ, MAP_SHARED, m_DmaBufFd, 0));
    if (mappedBuffer == MAP_FAILED)
    {
        throw std::runtime_error("Failed to mmap dma_buf");
    }

    dma_buf_sync syncStruct = { DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
    int result              = ioctl(m_DmaBufFd, DMA_BUF_IOCTL_SYNC, &syncStruct);
    if (result < 0)
    {
        throw std::runtime_error("Failed DMA_BUF_IOCTL_SYNC");
    }

    size_t sizeToCopy = std::min(len, m_Size);
    std::memcpy(outData, mappedBuffer, sizeToCopy);

    syncStruct = { DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ };
    result     = ioctl(m_DmaBufFd, DMA_BUF_IOCTL_SYNC, &syncStruct);
    if (result < 0)
    {
        throw std::runtime_error("Failed DMA_BUF_IOCTL_SYNC");
    }

    munmap(mappedBuffer, m_Size);
#elif defined(TARGET_MODEL)
    if (lseek(m_DmaBufFd, 0, SEEK_SET) < 0)
    {
        int err = errno;
        throw std::runtime_error("DmaBuffer lseek failed. errno = " + std::to_string(err) + ": " + std::strerror(err));
    }
    ssize_t numBytesRead = read(m_DmaBufFd, outData, len);
    if (numBytesRead < 0)
    {
        int err = errno;
        throw std::runtime_error("DmaBuffer read failed. errno = " + std::to_string(err) + ": " + std::strerror(err));
    }
    if (static_cast<size_t>(numBytesRead) != len)
    {
        throw std::runtime_error("DmaBuffer asked to read " + std::to_string(len) + " but only read " +
                                 std::to_string(numBytesRead));
    }
    if (lseek(m_DmaBufFd, 0, SEEK_SET) < 0)
    {
        int err = errno;
        throw std::runtime_error("DmaBuffer lseek failed. errno = " + std::to_string(err) + ": " + std::strerror(err));
    }
    ETHOSN_UNUSED(len);
    ETHOSN_UNUSED(outData);
#endif
}

}    // namespace system_tests
}    // namespace ethosn
