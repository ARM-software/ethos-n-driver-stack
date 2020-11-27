//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define ETHOSN_STRINGIZE(x) #x
#define ETHOSN_STRINGIZE_VALUE_OF(x) ETHOSN_STRINGIZE(x)

// To prevent the warning that we aren't using a variable, use this macro.
#define ETHOSN_UNUSED(x) (void)(x)

#define ETHOSN_ARRAY_SIZE(X) (sizeof(X) / sizeof(X[0]))

#if __GNUC__
#define ETHOSN_FUNCTION_SIGNATURE __PRETTY_FUNCTION__
#elif _MSC_VER
#define ETHOSN_FUNCTION_SIGNATURE __FUNCSIG__
#else
#define ETHOSN_FUNCTION_SIGNATURE "Please add the compiler in Macros.hpp"
#endif
