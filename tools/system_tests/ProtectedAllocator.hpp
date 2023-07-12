//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SystemTestsUtils.hpp"

#include <armnn/backends/ICustomAllocator.hpp>

#include <cstdint>
#include <map>
#include <memory>

namespace ethosn
{
namespace system_tests
{

class ProtectedAllocator : public armnn::ICustomAllocator
{
public:
    ProtectedAllocator();

    void* allocate(size_t size, size_t alignment);

    void free(void* ptr);

    armnn::MemorySource GetMemorySourceType();

    // Note: Protected buffer populating and reading is an internal testing
    // feature and will not be possible in production setups.
    void PopulateData(void* ptr, const uint8_t* inData, size_t len);
    void RetrieveData(void* ptr, uint8_t* outData, size_t len);

private:
    std::unique_ptr<DmaBufferDevice> m_DmaBufDev;
    struct Allocation
    {
        std::unique_ptr<DmaBuffer> m_DmaBuf;
        int m_Fd;
    };
    std::map<int, Allocation> m_Allocations;
};

}    // namespace system_tests
}    // namespace ethosn
