//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

// To prevent the warning that we arent using a variable, use this macro.
#define UNUSED(x) (void)(x)
