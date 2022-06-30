//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "WeightEncoderCache.hpp"

#include <fstream>

namespace ethosn
{
namespace support_library
{
namespace
{

int32_t ReadInt32(std::istream& s)
{
    int32_t i;
    s.read(reinterpret_cast<char*>(&i), sizeof(i));
    return i;
}

void WriteInt32(std::ostream& s, int32_t i)
{
    s.write(reinterpret_cast<const char*>(&i), sizeof(i));
}

uint32_t ReadUInt32(std::istream& s)
{
    uint32_t i;
    s.read(reinterpret_cast<char*>(&i), sizeof(i));
    return i;
}

void WriteUInt32(std::ostream& s, uint32_t i)
{
    s.write(reinterpret_cast<const char*>(&i), sizeof(i));
}

std::vector<uint8_t> ReadVectorUInt8(std::istream& s)
{
    std::vector<uint8_t> v;
    uint32_t c = ReadUInt32(s);
    v.resize(c);
    if (c > 0)
    {
        s.read(reinterpret_cast<char*>(v.data()), c * sizeof(uint8_t));
    }
    return v;
}

void WriteVectorUInt8(std::ostream& s, const std::vector<uint8_t>& v)
{
    WriteUInt32(s, static_cast<uint32_t>(v.size()));
    if (v.size() > 0)
    {
        s.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(uint8_t));
    }
}

std::vector<int32_t> ReadVectorInt32(std::istream& s)
{
    std::vector<int32_t> v;
    uint32_t c = ReadUInt32(s);
    v.resize(c);
    if (c > 0)
    {
        s.read(reinterpret_cast<char*>(v.data()), c * sizeof(int32_t));
    }
    return v;
}

void WriteVectorInt32(std::ostream& s, const std::vector<int32_t>& v)
{
    WriteInt32(s, static_cast<uint32_t>(v.size()));
    if (v.size() > 0)
    {
        s.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(int32_t));
    }
}

QuantizationInfo ReadQuantizationInfo(std::istream& s)
{
    QuantizationInfo q;

    q.SetZeroPoint(ReadInt32(s));

    uint32_t numScales = ReadUInt32(s);
    QuantizationScales scales(numScales);
    if (numScales > 0)
    {
        s.read(reinterpret_cast<char*>(&scales[0]), numScales * sizeof(float));
    }
    q.SetScales(scales);

    int32_t d = ReadInt32(s);
    if (d < 0)
    {
        assert(!q.GetQuantizationDim().has_value());
    }
    else
    {
        q.SetQuantizationDim(d);
    }

    return q;
}

void WriteQuantizationInfo(std::ostream& s, const QuantizationInfo& q)
{
    WriteInt32(s, q.GetZeroPoint());

    const QuantizationScales& scales = q.GetScales();
    WriteUInt32(s, static_cast<uint32_t>(scales.size()));
    if (scales.size() > 0)
    {
        s.write(reinterpret_cast<const char*>(&scales[0]), scales.size() * sizeof(float));
    }

    WriteInt32(s, q.GetQuantizationDim().has_value() ? q.GetQuantizationDim().value() : -1);
}

TensorInfo ReadTensorInfo(std::istream& s)
{
    TensorInfo t;
    t.m_DataFormat = static_cast<DataFormat>(ReadUInt32(s));
    t.m_DataType   = static_cast<DataType>(ReadUInt32(s));
    s.read(reinterpret_cast<char*>(t.m_Dimensions.data()), sizeof(t.m_Dimensions));
    t.m_QuantizationInfo = ReadQuantizationInfo(s);
    return t;
}

void WriteTensorInfo(std::ostream& s, const TensorInfo& t)
{
    WriteUInt32(s, static_cast<uint32_t>(t.m_DataFormat));
    WriteUInt32(s, static_cast<uint32_t>(t.m_DataType));
    s.write(reinterpret_cast<const char*>(t.m_Dimensions.data()), sizeof(t.m_Dimensions));
    WriteQuantizationInfo(s, t.m_QuantizationInfo);
}

WeightEncoderCache::Params ReadParams(std::istream& s)
{
    WeightEncoderCache::Params p;
    p.weightsTensorInfo      = ReadTensorInfo(s);
    p.weightsData            = std::make_shared<std::vector<uint8_t>>(ReadVectorUInt8(s));
    p.biasTensorInfo         = ReadTensorInfo(s);
    p.biasData               = ReadVectorInt32(s);
    p.inputQuantizationInfo  = ReadQuantizationInfo(s);
    p.outputQuantizationInfo = ReadQuantizationInfo(s);
    p.stripeDepth            = ReadUInt32(s);
    p.strideY                = ReadUInt32(s);
    p.strideX                = ReadUInt32(s);
    p.paddingTop             = ReadUInt32(s);
    p.paddingLeft            = ReadUInt32(s);
    p.iterationSize          = ReadUInt32(s);
    p.operation              = static_cast<ethosn::command_stream::MceOperation>(ReadUInt32(s));
    p.algorithm              = static_cast<CompilerMceAlgorithm>(ReadUInt32(s));
    return p;
}

void WriteParams(std::ostream& s, const WeightEncoderCache::Params& p)
{
    WriteTensorInfo(s, p.weightsTensorInfo);
    WriteVectorUInt8(s, *p.weightsData);
    WriteTensorInfo(s, p.biasTensorInfo);
    WriteVectorInt32(s, p.biasData);
    WriteQuantizationInfo(s, p.inputQuantizationInfo);
    WriteQuantizationInfo(s, p.outputQuantizationInfo);
    WriteUInt32(s, p.stripeDepth);
    WriteUInt32(s, p.strideY);
    WriteUInt32(s, p.strideX);
    WriteUInt32(s, p.paddingTop);
    WriteUInt32(s, p.paddingLeft);
    WriteUInt32(s, p.iterationSize);
    WriteUInt32(s, static_cast<uint32_t>(p.operation));
    WriteUInt32(s, static_cast<uint32_t>(p.algorithm));
}

EncodedWeights ReadEncodedWeights(std::istream& s)
{
    EncodedWeights w;

    uint32_t numMetadata = ReadUInt32(s);
    w.m_Metadata.resize(numMetadata);
    if (numMetadata > 0)
    {
        s.read(reinterpret_cast<char*>(w.m_Metadata.data()), numMetadata * sizeof(WeightsMetadata));
    }

    w.m_MaxSize = ReadUInt32(s);

    w.m_Data = ReadVectorUInt8(s);

    return w;
}

void WriteEncodedWeights(std::ostream& s, const EncodedWeights& w)
{
    WriteUInt32(s, static_cast<uint32_t>(w.m_Metadata.size()));
    if (w.m_Metadata.size() > 0)
    {
        s.write(reinterpret_cast<const char*>(w.m_Metadata.data()), w.m_Metadata.size() * sizeof(WeightsMetadata));
    }

    WriteUInt32(s, w.m_MaxSize);

    WriteVectorUInt8(s, w.m_Data);
}

}    // namespace

WeightEncoderCache::WeightEncoderCache(const HardwareCapabilities& caps, const char* id)
    : m_Encoder(caps)
{
    // Load cache entries from file, if specified. This can make network compilation _much_ faster and
    // can very useful for debugging/development.
    // Use a unique ID to create a separate cache file for each cache, to avoid slowing down all caches
    // by including irrelevant entries in them all.
    const char* env = std::getenv("ETHOSN_SUPPORT_LIBRARY_WEIGHT_ENCODER_CACHE");
    if (env && strlen(env) > 0)
    {
        m_PeristentFilename = std::string(env) + id + ".bin";
        g_Logger.Warning("Weight encoder cache is being loaded from %s. Beware this may be stale!",
                         m_PeristentFilename.c_str());

        std::ifstream f(m_PeristentFilename, std::ios::binary);
        while (f.good() && f.peek() != EOF)
        {
            Params p         = ReadParams(f);
            EncodedWeights w = ReadEncodedWeights(f);
            m_Entries[p]     = std::make_shared<EncodedWeights>(std::move(w));
        }
    }
}

bool WeightEncoderCache::Params::operator==(const Params& r) const
{
    return weightsTensorInfo == r.weightsTensorInfo && *weightsData == *r.weightsData &&
           biasTensorInfo == r.biasTensorInfo && biasData == r.biasData &&
           inputQuantizationInfo == r.inputQuantizationInfo && outputQuantizationInfo == r.outputQuantizationInfo &&
           stripeDepth == r.stripeDepth && strideY == r.strideY && strideX == r.strideX && paddingTop == r.paddingTop &&
           paddingLeft == r.paddingLeft && iterationSize == r.iterationSize && operation == r.operation &&
           algorithm == r.algorithm;
}

std::shared_ptr<ethosn::support_library::EncodedWeights> WeightEncoderCache::Encode(const Params& params)
{
    auto it = m_Entries.find(params);
    if (it == m_Entries.end())
    {
        EncodedWeights w =
            m_Encoder.Encode(params.weightsTensorInfo, params.weightsData->data(), params.biasTensorInfo,
                             params.biasData.data(), params.inputQuantizationInfo, params.outputQuantizationInfo,
                             params.stripeDepth, params.strideY, params.strideX, params.paddingTop, params.paddingLeft,
                             params.iterationSize, params.operation, params.algorithm);
        it = m_Entries.insert({ params, std::make_shared<EncodedWeights>(w) }).first;

        // Save this entry to the persistent file if enabled.
        if (!m_PeristentFilename.empty())
        {
            std::ofstream f(m_PeristentFilename, std::ios::app | std::ios::binary | std::ios::out);
            WriteParams(f, params);
            WriteEncodedWeights(f, w);
        }
    }
    return it->second;
}

size_t WeightEncoderCache::Hasher::operator()(const Params& p) const
{
    // This hash function is deliberately very simple and therefore you might think would lead to lots of
    // collisions. We may now get more hash collisions, which could be an issue as the equality comparison
    // is very expensive. However, so far we've noticed that comparing weights is less expensive than
    // copying them around for each part so this function is good enough.
    size_t h = 17;
    h        = h * 37 + std::hash<size_t>()(p.weightsData->size());
    h        = h * 37 + std::hash<size_t>()(p.biasData.size());
    h        = h * 37 + std::hash<uint32_t>()(p.stripeDepth);
    h        = h * 37 + std::hash<uint32_t>()(p.iterationSize);
    // Note we cast the enum to an integral type, as some compilers (e.g. aarch64-linux-gnu-g++ 5.3.1)
    // don't support using the enum type directly, even though the spec indicates that they should.
    h = h * 37 + std::hash<uint32_t>()(static_cast<uint32_t>(p.algorithm));
    return h;
}

}    // namespace support_library

}    // namespace ethosn
