//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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
    int GetFileDescriptor();

private:
    class InferenceImpl;
    std::unique_ptr<InferenceImpl> inferenceImpl;
};
}    // namespace driver_library
}    // namespace ethosn
