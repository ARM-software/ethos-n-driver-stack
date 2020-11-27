//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Some compilers (e.g. MSVC) treat __VA_ARGS__ as a single token rather than automatically
// expanding it to each argument.
// Wrapping the entire macro definition in this EXPAND(...) macro works around this quirk.
#define EXPAND(x) x

// Indirect concatenation/token paste (a and b are expanded)
#define CAT(a, b) RAW_CAT(a, b)
// Raw concatenation/token paste
#define RAW_CAT(a, b) a##b

// Expands to the number of variadic arguments in the call. Add elements to the descending
// integer sequence to increase the maximum number of variadic arguments supported
#define N_ARGS(...)                                                                                                    \
    EXPAND(N_ARGS_IMPL(__VA_ARGS__, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,    \
                       14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
// Helper macro for N_ARGS(). Add placeholder arguments to increase the maximum
// number of variadic arguments supported
#define N_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21,    \
                    _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, n, ...)                           \
    n

// Expand to the comma-separated sequence 0, ..., n-1
#define SEQ(n) CAT(SEQ_, n)

// Fixed-maximum-depth macro recursion for SEQ
#define SEQ_1 0
#define SEQ_2 SEQ_1, 1
#define SEQ_3 SEQ_2, 2
#define SEQ_4 SEQ_3, 3
#define SEQ_5 SEQ_4, 4
#define SEQ_6 SEQ_5, 5
#define SEQ_7 SEQ_6, 6
#define SEQ_8 SEQ_7, 7
#define SEQ_9 SEQ_8, 8
#define SEQ_10 SEQ_9, 9
#define SEQ_11 SEQ_10, 10
#define SEQ_12 SEQ_11, 11
#define SEQ_13 SEQ_12, 12
#define SEQ_14 SEQ_13, 13
#define SEQ_15 SEQ_14, 14
#define SEQ_16 SEQ_15, 15
#define SEQ_17 SEQ_16, 16
// Add lines here to increase the maximum supported n for SEQ

// Expand f(arg) for each arg in the varadic call.
#define FOREACH_N(f, ...) RAW_FOREACH_N(f, f, __VA_ARGS__)

// Expand f(arg) for each arg in the varadic call except for the last.
// Expand fN(last) on the last arg instead.
#define RAW_FOREACH_N(f, fN, ...) EXPAND(CAT(FOREACH_N_, N_ARGS(__VA_ARGS__))(f, fN, __VA_ARGS__))

// Fixed-maximum-depth macro recursion for FOREACH_N
#define FOREACH_N_1(f, fN, _1, ...) fN(_1)
#define FOREACH_N_2(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_1(f, fN, __VA_ARGS__))
#define FOREACH_N_3(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_2(f, fN, __VA_ARGS__))
#define FOREACH_N_4(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_3(f, fN, __VA_ARGS__))
#define FOREACH_N_5(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_4(f, fN, __VA_ARGS__))
#define FOREACH_N_6(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_5(f, fN, __VA_ARGS__))
#define FOREACH_N_7(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_6(f, fN, __VA_ARGS__))
#define FOREACH_N_8(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_7(f, fN, __VA_ARGS__))
#define FOREACH_N_9(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_8(f, fN, __VA_ARGS__))
#define FOREACH_N_10(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_9(f, fN, __VA_ARGS__))
#define FOREACH_N_11(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_10(f, fN, __VA_ARGS__))
#define FOREACH_N_12(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_11(f, fN, __VA_ARGS__))
#define FOREACH_N_13(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_12(f, fN, __VA_ARGS__))
#define FOREACH_N_14(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_13(f, fN, __VA_ARGS__))
#define FOREACH_N_15(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_14(f, fN, __VA_ARGS__))
#define FOREACH_N_16(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_15(f, fN, __VA_ARGS__))
#define FOREACH_N_17(f, fN, _1, ...) f(_1) EXPAND(FOREACH_N_16(f, fN, __VA_ARGS__))
// Add lines here to increase the maximum number of variadic arguments supported

// Expand f(arg1, arg2) for each arg1, arg2 pair in the variadic call.
#define FOREACH_2N(f, ...) RAW_FOREACH_2N(f, f, __VA_ARGS__)

// Expand f(arg1, arg2) for each arg1, arg2 pair in the variadic call except for the last.
// Expand fN(last1, last2) on the last arg1, arg2 pair instead.
#define RAW_FOREACH_2N(f, fN, ...) EXPAND(CAT(FOREACH_2N_, N_ARGS(__VA_ARGS__))(f, fN, __VA_ARGS__))

// Fixed-maximum-depth macro recursion for FOREACH_2N
#define FOREACH_2N_2(f, fN, _1, _2, ...) fN(_1, _2)
#define FOREACH_2N_4(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_2(f, fN, __VA_ARGS__))
#define FOREACH_2N_6(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_4(f, fN, __VA_ARGS__))
#define FOREACH_2N_8(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_6(f, fN, __VA_ARGS__))
#define FOREACH_2N_10(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_8(f, fN, __VA_ARGS__))
#define FOREACH_2N_12(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_10(f, fN, __VA_ARGS__))
#define FOREACH_2N_14(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_12(f, fN, __VA_ARGS__))
#define FOREACH_2N_16(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_14(f, fN, __VA_ARGS__))
#define FOREACH_2N_18(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_16(f, fN, __VA_ARGS__))
#define FOREACH_2N_20(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_18(f, fN, __VA_ARGS__))
#define FOREACH_2N_22(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_20(f, fN, __VA_ARGS__))
#define FOREACH_2N_24(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_22(f, fN, __VA_ARGS__))
#define FOREACH_2N_26(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_24(f, fN, __VA_ARGS__))
#define FOREACH_2N_28(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_26(f, fN, __VA_ARGS__))
#define FOREACH_2N_30(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_28(f, fN, __VA_ARGS__))
#define FOREACH_2N_32(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_30(f, fN, __VA_ARGS__))
#define FOREACH_2N_34(f, fN, _1, _2, ...) f(_1, _2) EXPAND(FOREACH_2N_32(f, fN, __VA_ARGS__))
// Add lines here to increase the maximum number of variadic arguments supported

// Expands to the comma-separated list of odd-index variadic arguments
#define ODD_ARGS(...) RAW_FOREACH_2N(ODD_ARG_F, ODD_ARG_FN, __VA_ARGS__)

#define ODD_ARG_F(_1, _2) _1,
#define ODD_ARG_FN(_1, _2) _1

// Expands to the comma-separated list of odd-index variadic arguments
#define EVEN_ARGS(...) RAW_FOREACH_2N(EVEN_ARG_F, EVEN_ARG_FN, __VA_ARGS__)

#define EVEN_ARG_F(_1, _2) _2,
#define EVEN_ARG_FN(_1, _2) _2
