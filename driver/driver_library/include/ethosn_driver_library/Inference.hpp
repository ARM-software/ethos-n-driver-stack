//
// Copyright © 2018-2020,2023 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

namespace ethosn
{
namespace driver_library
{

/// Possible results from reading the file descriptor returned by Inference::GetFileDescriptor().
/// Note this must be kept in-sync with the kernel driver's definitions.
enum class InferenceResult
{
    Scheduled = 0,
    Running   = 1,
    Completed = 2,
    Error     = 3,
};

class Inference
{
public:
    Inference(int fileDescriptor);
    ~Inference();

    /// Get a file descriptor which can be used to interact with this inference. The file descriptor supports the
    /// following operations:
    ///    * poll - Can be used to wait until the inference is complete
    ///    * read - Can be used to retrieve the status of the inference. Reading will always return a value of type
    ///             InferenceResult.
    ///    * release - Can be used to abort the inference.
    int GetFileDescriptor() const;

    /// Once an inference is complete, this can be used to get the number of NPU cycles
    /// that the inference took to complete, as measured by the NPU firmware.
    /// Note that this only includes time spent actually running the inference on the NPU, and not any
    /// time spent waiting for previously scheduled inferences to finish.
    /// If this is called before an inference has finished running, this will return zero.
    uint64_t GetCycleCount() const;

private:
    class InferenceImpl;
    std::unique_ptr<InferenceImpl> inferenceImpl;
};
}    // namespace driver_library
}    // namespace ethosn
