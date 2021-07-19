//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define ETHOSN_CAT(a, b) a##b
#define ETHOSN_CAT_VALUES_OF(a, b) ETHOSN_CAT(a, b)

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
