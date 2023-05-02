//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

// Performs max pooling in one dimension (X or Y, specified via compile-time define)
// as a standalone PLE kernel (i.e. loads/saves to SRAM and doesn't use the MCEIF).
// This is done by loading groups of patches (8x8 elements) from SRAM along each
// row (or column for pooling in Y) and bufferring them in the VE registers.
// Once we load the next group, we have enough data to calculate the result for the
// previous group and we write this out to SRAM. The actual max pooling is done by
// "offsetting" the patches (using swizzle instructions) and then performing an
// elementwise max between the original patch and the offset patch, and repeating
// multiple times with different offsets to cover the whole pooling 'window'.
//
// The following example is for pooling of size 3 in X with 1 padding on left and right,
// but everything is equivalent for pooling in Y and other pooling sizes and paddings.
// This diagram shows the registers used to store each patch as we are partway through
// processing a row:
//
//
//       ===============================================================================================
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||      0       |      2       ||      4       |      6       ||      8       |     10       ||
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       -----------------------------------------------------------------------------------------------
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||      1       |      3       ||      5       |      7       ||      9       |     11       ||
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ===============================================================================================
//
//
//
// We loop over each row of groups in the input tensor, and load one group at a time
// into registers 8-11. For the first group loaded in a row, we can't do any processing
// so we move straight to the next group. Before loading the next group though we "shuffle down"
// the groups already loaded, so in this case the data we loaded into 8-11 gets shuffled down
// to registers 4-7. We then load the next group into 8-11 and we now have enough data
// to do some pooling! We always calculate the output values for the pixels corresponding
// to registers 4-7, which in this case is the first 8x8 group of the tensor.
//
// There are four output patches to calculate, which are independent, but we interleave
// the calculations to avoid having to insert NOPs. To get the result for the top-left
// patch (4), we take the elementwise max between this patch and an "offset" version of
// 4 which contains the elements offset by one to the left. This is shown in the diagram
// below with the "dashed" vertical lines which show the offset version of patch 4 (and the
// same for 5, 6 and 7). The offset version of patch 4 is stored in patch 12 as it is labelled
// here.
//
//                                    <----- 12 ----> <---- 14 ---->
//       ===============================================================================================
//       ||              |           |  ||           |  |           |  ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |           |  ||           |  |           |  ||              |              ||
//       ||              |      2       ||      4       |      6       ||      8       |              ||
//       ||              |           |  ||           |  |           |  ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |           |  ||           |  |           |  ||              |              ||
//       -----------------------------------------------------------------------------------------------
//       ||              |           |  ||           |  |           |  ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |           |  ||           |  |           |  ||              |              ||
//       ||              |      3       ||      5       |      7       ||      9       |              ||
//       ||              |           |  ||           |  |           |  ||              |              ||
//       ||              |              ||              |              ||              |              ||
//       ||              |           |  ||           |  |           |  ||              |              ||
//       ===============================================================================================
//                                    <----- 13 ----> <---- 15 ---->
// We also do the same but offseting one element to the right, and again do an element-wise max.
// This gives us the result for patch 4, where each pixel is calculated as the maximum of itself
// and the pixels to the left and right.
//
// The same happens for patches 5, 6 and 7 and that gives us an entire group (8x8 elements) of results
// which we can write out to SRAM. Note that calculating the result for patch 4 requires data from patch 2,
// and calculating the result for patch 6 requires data from patch 8, so we need data on both sides,
// which is why we need to keep 3 groups in our rolling buffer. Each group is shuffled down as we load
// new data in at the right hand side. Using this method we can look at neighbouring data up to 8 pixels
// away from the element being calculated, which limits the maximum pooling/padding size that we support.
// It could be possible to support larger sizes by keeping more groups loaded so that we have access
// to data further away.

#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/DfcSramTraversal.hpp"
#include "../include/ethosn_ple/SignedSupport.hpp"

namespace
{

// Swizzle registers - see SetupSwizzles() for an explanation.
#if defined(IS_DIRECTION_X)
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_LEFT_1_EVEN_EVEN = 0;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_LEFT_1_ODD_ODD   = 1;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_LEFT_2_EVEN_EVEN = 2;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_LEFT_2_ODD_ODD   = 3;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_LEFT_3_EVEN_EVEN = 4;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_LEFT_3_ODD_ODD   = 5;
#elif defined(IS_DIRECTION_Y)
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_UP_1_EVEN_ODD = 0;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_UP_1_ODD_EVEN = 1;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_UP_2_EVEN_ODD = 2;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_UP_2_ODD_EVEN = 3;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_UP_3_EVEN_ODD = 4;
constexpr uint32_t SWIZZLE_REG_OFFSET_PATCH_UP_3_ODD_EVEN = 5;
#endif

/// Fills the swizzle registers with the patterns that we need for offsetting patches left/right/up/down
/// by various amounts.
/// The offsetting functions take as input two adjacent patches, L and R where L is to the left of R
/// spatially and outputs a new patch which contains some columns from L and the rest of the columns
/// from R. For example, shifting one element to the left:
///
///         L       R
///
///      a b c d|q r s t           d q r s
///      e f g h|u v w x      =>   h u v w
///      i j k l|y z 0 1           l y z 0
///      m n o p|2 3 4 5           p 2 3 4
///
/// An example for shifting one element to the right:
///
///         L       R
///
///      a b c d|q r s t              b c d q
///      e f g h|u v w x         =>   f g h u
///      i j k l|y z 0 1              j k l y
///      m n o p|2 3 4 5              n o p 2
///
void SetupSwizzles()
{
    // Unfortunately we can't use odd numbered registers for swizzle inputs,
    // so we need to have two cases based on whether the input registers are odd or even.

#if defined(IS_DIRECTION_X)
    // 2-bits per output element, selecting one of the four input registers
    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_1_EVEN_EVEN>(0b10'10'10'00'10'10'10'00'10'10'10'00'10'10'10'00);
    // 4-bits per output element, selecting one of the 16 elements within the input registers that was selected (by the above mask)
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_1_EVEN_EVEN>(0x65472103, 0xEDCFA98B);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_1_ODD_ODD>(0b11'11'11'01'11'11'11'01'11'11'11'01'11'11'11'01);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_1_ODD_ODD>(0x65472103, 0xEDCFA98B);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_2_EVEN_EVEN>(0b10'10'00'00'10'10'00'00'10'10'00'00'10'10'00'00);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_2_EVEN_EVEN>(0x54761032, 0x54761032 + 0x88888888);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_2_ODD_ODD>(0b11'11'01'01'11'11'01'01'11'11'01'01'11'11'01'01);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_2_ODD_ODD>(0x54761032, 0x54761032 + 0x88888888);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_3_EVEN_EVEN>(0b10'00'00'00'10'00'00'00'10'00'00'00'10'00'00'00);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_3_EVEN_EVEN>(0x47650321, 0x47650321 + 0x88888888);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_3_ODD_ODD>(0b11'01'01'01'11'01'01'01'11'01'01'01'11'01'01'01);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_LEFT_3_ODD_ODD>(0x47650321, 0x47650321 + 0x88888888);
#elif defined(IS_DIRECTION_Y)
    // For pooling in Y, the register odd/even-ness is slightly different, as we always have one odd and one
    // even input register, but they can be either way round (odd+even or even+odd)
    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_1_EVEN_ODD>(0b11'11'11'11'11'11'11'11'11'11'11'11'00'00'00'00);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_1_EVEN_ODD>(0x3210FEDC, 0xBA987654);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_1_ODD_EVEN>(0b10'10'10'10'10'10'10'10'10'10'10'10'01'01'01'01);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_1_ODD_EVEN>(0x3210FEDC, 0xBA987654);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_2_EVEN_ODD>(0b11'11'11'11'11'11'11'11'00'00'00'00'00'00'00'00);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_2_EVEN_ODD>(0xFEDCBA98, 0x76543210);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_2_ODD_EVEN>(0b10'10'10'10'10'10'10'10'01'01'01'01'01'01'01'01);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_2_ODD_EVEN>(0xFEDCBA98, 0x76543210);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_3_EVEN_ODD>(0b11'11'11'11'00'00'00'00'00'00'00'00'00'00'00'00);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_3_EVEN_ODD>(0xBA987654, 0x3210FEDC);

    ve_set_swzsel_reg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_3_ODD_EVEN>(0b10'10'10'10'01'01'01'01'01'01'01'01'01'01'01'01);
    ve_set_swzsel_subreg_sel<SWIZZLE_REG_OFFSET_PATCH_UP_3_ODD_EVEN>(0xBA987654, 0x3210FEDC);
#endif
}

#if defined(IS_DIRECTION_X)

/// See SetupSwizzles for an explanation for these offsetting functions
/// @{
template <int O, int L, int R>
void OffsetPatchLeft1()
{
    static_assert((L % 2 == 0 && R % 2 == 0) || (L % 2 == 1 && R % 2 == 1), "Must be both odd or both even");
    if constexpr (L % 2 == 0 && R % 2 == 0)
    {
        ve_swz_8<O, L, R, SWIZZLE_REG_OFFSET_PATCH_LEFT_1_EVEN_EVEN>();
    }
    else if constexpr (L % 2 == 1 && R % 2 == 1)
    {
        ve_swz_8<O, L - 1, R - 1, SWIZZLE_REG_OFFSET_PATCH_LEFT_1_ODD_ODD>();
    }
}

template <int O, int L, int R>
void OffsetPatchLeft2()
{
    static_assert((L % 2 == 0 && R % 2 == 0) || (L % 2 == 1 && R % 2 == 1), "Must be both odd or both even");
    if constexpr (L % 2 == 0 && R % 2 == 0)
    {
        ve_swz_8<O, L, R, SWIZZLE_REG_OFFSET_PATCH_LEFT_2_EVEN_EVEN>();
    }
    else if constexpr (L % 2 == 1 && R % 2 == 1)
    {
        ve_swz_8<O, L - 1, R - 1, SWIZZLE_REG_OFFSET_PATCH_LEFT_2_ODD_ODD>();
    }
}

template <int O, int L, int R>
void OffsetPatchLeft3()
{
    static_assert((L % 2 == 0 && R % 2 == 0) || (L % 2 == 1 && R % 2 == 1), "Must be both odd or both even");
    if constexpr (L % 2 == 0 && R % 2 == 0)
    {
        ve_swz_8<O, L, R, SWIZZLE_REG_OFFSET_PATCH_LEFT_3_EVEN_EVEN>();
    }
    else if constexpr (L % 2 == 1 && R % 2 == 1)
    {
        ve_swz_8<O, L - 1, R - 1, SWIZZLE_REG_OFFSET_PATCH_LEFT_3_ODD_ODD>();
    }
}

template <int O, int L, int R>
void OffsetPatchRight1()
{
    // Ofsetting right by 1 is the same as offsetting left by 3, as long as the L and R registers contain the correct data.
    return OffsetPatchLeft3<O, L, R>();
}

template <int O, int L, int R>
void OffsetPatchRight2()
{
    // Ofsetting right by 2 is the same as offsetting left by 2, as long as the L and R registers contain the correct data.
    return OffsetPatchLeft2<O, L, R>();
}

template <int O, int L, int R>
void OffsetPatchRight3()
{
    // Ofsetting right by 3 is the same as offsetting left by 1, as long as the L and R registers contain the correct data.
    return OffsetPatchLeft1<O, L, R>();
}

#elif defined(IS_DIRECTION_Y)

template <int O, int U, int D>
void OffsetPatchUp1()
{
    static_assert((U % 2 == 0 && D % 2 == 1) || (U % 2 == 1 && D % 2 == 0), "Must be odd and even but not both");
    if constexpr (U % 2 == 0 && D % 2 == 1)
    {
        ve_swz_8<O, U, D - 1, SWIZZLE_REG_OFFSET_PATCH_UP_1_EVEN_ODD>();
    }
    else if constexpr (U % 2 == 1 && D % 2 == 0)
    {
        ve_swz_8<O, U - 1, D, SWIZZLE_REG_OFFSET_PATCH_UP_1_ODD_EVEN>();
    }
}

template <int O, int U, int D>
void OffsetPatchUp2()
{
    static_assert((U % 2 == 0 && D % 2 == 1) || (U % 2 == 1 && D % 2 == 0), "Must be odd and even but not both");
    if constexpr (U % 2 == 0 && D % 2 == 1)
    {
        ve_swz_8<O, U, D - 1, SWIZZLE_REG_OFFSET_PATCH_UP_2_EVEN_ODD>();
    }
    else if constexpr (U % 2 == 1 && D % 2 == 0)
    {
        ve_swz_8<O, U - 1, D, SWIZZLE_REG_OFFSET_PATCH_UP_2_ODD_EVEN>();
    }
}

template <int O, int U, int D>
void OffsetPatchUp3()
{
    static_assert((U % 2 == 0 && D % 2 == 1) || (U % 2 == 1 && D % 2 == 0), "Must be odd and even but not both");
    if constexpr (U % 2 == 0 && D % 2 == 1)
    {
        ve_swz_8<O, U, D - 1, SWIZZLE_REG_OFFSET_PATCH_UP_3_EVEN_ODD>();
    }
    else if constexpr (U % 2 == 1 && D % 2 == 0)
    {
        ve_swz_8<O, U - 1, D, SWIZZLE_REG_OFFSET_PATCH_UP_3_ODD_EVEN>();
    }
}

template <int O, int U, int D>
void OffsetPatchDown1()
{
    // Ofsetting down by 1 is the same as offsetting up by 3, as long as the U and D registers contain the correct data.
    return OffsetPatchUp3<O, U, D>();
}

template <int O, int U, int D>
void OffsetPatchDown2()
{
    // Ofsetting down by 2 is the same as offsetting up by 2, as long as the U and D registers contain the correct data.
    return OffsetPatchUp2<O, U, D>();
}

template <int O, int U, int D>
void OffsetPatchDown3()
{
    // Ofsetting down by 3 is the same as offsetting up by 1, as long as the U and D registers contain the correct data.
    return OffsetPatchUp1<O, U, D>();
}
/// @}

#endif

void CalculateAndSaveOneGroup(
    uint32_t outDfcAddr, unsigned dfc, udma::UdmaStorer& udmaStorer, uint32_t padBefore, uint32_t poolingSize)
{
    // Accumulate max results in registers 16-19, starting with the original data (offset 0)
    // It seems the compiler is interleaving some CPU instructions between these ve instructions
    // So we do not need a nop after these ve instructions before the result is read further
    // If this fails in the further please consider adding nops.
    ve_mov_16<16, 4>();
    ve_mov_16<18, 6>();

#if defined(IS_DIRECTION_X)
    // We need to take the elementwise max between a range of offset patches, based on the pooling
    // size and padding.
    uint32_t leftmostOffset  = padBefore;
    uint32_t rightmostOffset = poolingSize - padBefore - 1;

    if (leftmostOffset >= 8)
    {
        // This one is simple because an offset of 8 is just two patches along
        Max8<16, 16, 0>();
        Max8<17, 17, 1>();
        Max8<18, 18, 2>();
        Max8<19, 19, 3>();
    }
    if (leftmostOffset >= 7)
    {
        // An offset of 7 is a whole patch plus an offset of 3.
        OffsetPatchLeft3<12, 0, 2>();
        OffsetPatchLeft3<13, 1, 3>();
        OffsetPatchLeft3<14, 2, 4>();
        OffsetPatchLeft3<15, 3, 5>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (leftmostOffset >= 6)
    {
        // An offset of 6 is a whole patch plus an offset of 2.
        OffsetPatchLeft2<12, 0, 2>();
        OffsetPatchLeft2<13, 1, 3>();
        OffsetPatchLeft2<14, 2, 4>();
        OffsetPatchLeft2<15, 3, 5>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (leftmostOffset >= 5)
    {
        // An offset of 5 is a whole patch plus an offset of 1.
        OffsetPatchLeft1<12, 0, 2>();
        OffsetPatchLeft1<13, 1, 3>();
        OffsetPatchLeft1<14, 2, 4>();
        OffsetPatchLeft1<15, 3, 5>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (leftmostOffset >= 4)
    {
        // This one is simple because an offset of 4 is just one patch along
        Max8<16, 16, 2>();
        Max8<17, 17, 3>();
        Max8<18, 18, 4>();
        Max8<19, 19, 5>();
    }
    if (leftmostOffset >= 3)
    {
        OffsetPatchLeft3<12, 2, 4>();
        OffsetPatchLeft3<13, 3, 5>();
        OffsetPatchLeft3<14, 4, 6>();
        OffsetPatchLeft3<15, 5, 7>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (leftmostOffset >= 2)
    {
        OffsetPatchLeft2<12, 2, 4>();
        OffsetPatchLeft2<13, 3, 5>();
        OffsetPatchLeft2<14, 4, 6>();
        OffsetPatchLeft2<15, 5, 7>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (leftmostOffset >= 1)
    {
        OffsetPatchLeft1<12, 2, 4>();
        OffsetPatchLeft1<13, 3, 5>();
        OffsetPatchLeft1<14, 4, 6>();
        OffsetPatchLeft1<15, 5, 7>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    // Offset of zero is already handled as we initialize 16-19 with this value at the top of this function
    if (rightmostOffset >= 1)
    {
        OffsetPatchRight1<12, 4, 6>();
        OffsetPatchRight1<13, 5, 7>();
        OffsetPatchRight1<14, 6, 8>();
        OffsetPatchRight1<15, 7, 9>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (rightmostOffset >= 2)
    {
        OffsetPatchRight2<12, 4, 6>();
        OffsetPatchRight2<13, 5, 7>();
        OffsetPatchRight2<14, 6, 8>();
        OffsetPatchRight2<15, 7, 9>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (rightmostOffset >= 3)
    {
        OffsetPatchRight3<12, 4, 6>();
        OffsetPatchRight3<13, 5, 7>();
        OffsetPatchRight3<14, 6, 8>();
        OffsetPatchRight3<15, 7, 9>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (rightmostOffset >= 4)
    {
        // This one is simple because an offset of 4 is just one patch along
        Max8<16, 16, 6>();
        Max8<17, 17, 7>();
        Max8<18, 18, 8>();
        Max8<19, 19, 9>();
    }
    if (rightmostOffset >= 5)
    {
        // An offset of 5 is a whole patch plus an offset of 1.
        OffsetPatchRight1<12, 6, 8>();
        OffsetPatchRight1<13, 7, 9>();
        OffsetPatchRight1<14, 8, 10>();
        OffsetPatchRight1<15, 9, 11>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (rightmostOffset >= 6)
    {
        // An offset of 6 is a whole patch plus an offset of 2.
        OffsetPatchRight2<12, 6, 8>();
        OffsetPatchRight2<13, 7, 9>();
        OffsetPatchRight2<14, 8, 10>();
        OffsetPatchRight2<15, 9, 11>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (rightmostOffset >= 7)
    {
        // An offset of 7 is a whole patch plus an offset of 2.
        OffsetPatchRight3<12, 6, 8>();
        OffsetPatchRight3<13, 7, 9>();
        OffsetPatchRight3<14, 8, 10>();
        OffsetPatchRight3<15, 9, 11>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (rightmostOffset >= 8)
    {
        // This one is simple because an offset of 8 is just two patches along
        Max8<16, 16, 8>();
        Max8<17, 17, 9>();
        Max8<18, 18, 10>();
        Max8<19, 19, 11>();
    }
    nop<1>();
#elif defined(IS_DIRECTION_Y)
    // We need to take the elementwise max between a range of offset patches, based on the pooling
    // size and padding.
    uint32_t topmostOffset    = padBefore;
    uint32_t bottommostOffset = poolingSize - padBefore - 1;

    if (topmostOffset >= 8)
    {
        // This one is simple because an offset of 8 is just two patches along
        Max8<16, 16, 0>();
        Max8<17, 17, 1>();
        Max8<18, 18, 2>();
        Max8<19, 19, 3>();
    }
    if (topmostOffset >= 7)
    {
        // An offset of 7 is a whole patch plus an offset of 3.
        OffsetPatchUp3<12, 0, 1>();
        OffsetPatchUp3<13, 1, 4>();
        OffsetPatchUp3<14, 2, 3>();
        OffsetPatchUp3<15, 3, 6>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (topmostOffset >= 6)
    {
        // An offset of 6 is a whole patch plus an offset of 2.
        OffsetPatchUp2<12, 0, 1>();
        OffsetPatchUp2<13, 1, 4>();
        OffsetPatchUp2<14, 2, 3>();
        OffsetPatchUp2<15, 3, 6>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (topmostOffset >= 5)
    {
        // An offset of 5 is a whole patch plus an offset of 1.
        OffsetPatchUp1<12, 0, 1>();
        OffsetPatchUp1<13, 1, 4>();
        OffsetPatchUp1<14, 2, 3>();
        OffsetPatchUp1<15, 3, 6>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (topmostOffset >= 4)
    {
        // This one is simple because an offset of 4 is just one patch along
        Max8<16, 16, 1>();
        Max8<17, 17, 4>();
        Max8<18, 18, 3>();
        Max8<19, 19, 6>();
    }
    if (topmostOffset >= 3)
    {
        OffsetPatchUp3<12, 1, 4>();
        OffsetPatchUp3<13, 4, 5>();
        OffsetPatchUp3<14, 3, 6>();
        OffsetPatchUp3<15, 6, 7>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (topmostOffset >= 2)
    {
        OffsetPatchUp2<12, 1, 4>();
        OffsetPatchUp2<13, 4, 5>();
        OffsetPatchUp2<14, 3, 6>();
        OffsetPatchUp2<15, 6, 7>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (topmostOffset >= 1)
    {
        OffsetPatchUp1<12, 1, 4>();
        OffsetPatchUp1<13, 4, 5>();
        OffsetPatchUp1<14, 3, 6>();
        OffsetPatchUp1<15, 6, 7>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    // Offset of zero is already handled as we initialize 16-19 with this value at the top of this function
    if (bottommostOffset >= 1)
    {
        OffsetPatchDown1<12, 4, 5>();
        OffsetPatchDown1<13, 5, 8>();
        OffsetPatchDown1<14, 6, 7>();
        OffsetPatchDown1<15, 7, 10>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (bottommostOffset >= 2)
    {
        OffsetPatchDown2<12, 4, 5>();
        OffsetPatchDown2<13, 5, 8>();
        OffsetPatchDown2<14, 6, 7>();
        OffsetPatchDown2<15, 7, 10>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (bottommostOffset >= 3)
    {
        OffsetPatchDown3<12, 4, 5>();
        OffsetPatchDown3<13, 5, 8>();
        OffsetPatchDown3<14, 6, 7>();
        OffsetPatchDown3<15, 7, 10>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (bottommostOffset >= 4)
    {
        // This one is simple because an offset of 4 is just one patch along
        Max8<16, 16, 5>();
        Max8<17, 17, 8>();
        Max8<18, 18, 7>();
        Max8<19, 19, 10>();
    }
    if (bottommostOffset >= 5)
    {
        // An offset of 5 is a whole patch plus an offset of 1.
        OffsetPatchDown1<12, 5, 8>();
        OffsetPatchDown1<13, 8, 9>();
        OffsetPatchDown1<14, 7, 10>();
        OffsetPatchDown1<15, 10, 11>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (bottommostOffset >= 6)
    {
        // An offset of 6 is a whole patch plus an offset of 2.
        OffsetPatchDown2<12, 5, 8>();
        OffsetPatchDown2<13, 8, 9>();
        OffsetPatchDown2<14, 7, 10>();
        OffsetPatchDown2<15, 10, 11>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (bottommostOffset >= 7)
    {
        // An offset of 7 is a whole patch plus an offset of 2.
        OffsetPatchDown3<12, 5, 8>();
        OffsetPatchDown3<13, 8, 9>();
        OffsetPatchDown3<14, 7, 10>();
        OffsetPatchDown3<15, 10, 11>();

        Max8<16, 16, 12>();
        Max8<17, 17, 13>();
        Max8<18, 18, 14>();
        Max8<19, 19, 15>();
    }
    if (bottommostOffset >= 8)
    {
        // This one is simple because an offset of 8 is just two patches along
        Max8<16, 16, 8>();
        Max8<17, 17, 9>();
        Max8<18, 18, 10>();
        Max8<19, 19, 11>();
    }
    nop<1>();
#endif

    // Save from VE registers 16-19 to PLE output SRAM (at address 0)
    lsu::StoreRfOutram<0>({ 0U, 16U * WORDS_PER_REGISTER });
    lsu::StoreRfOutram<2>({ 0U, 16U * WORDS_PER_REGISTER });

    // Store one group from PLE output SRAM to regular SRAM
    udma::Address udmaOutAddr;
    udmaOutAddr.dfcAddrWords = outDfcAddr / 4;
    udmaOutAddr.pleAddr      = 0;
    udmaStorer.Store(dfc, udmaOutAddr);
    udmaStorer.WaitForUdma();
}

ncu_ple_interface::PleMsg::StripeDone ProcessStripe(EnumBitset<Event>& activeEvents)
{
    // Read stripe parameters from scratch registers
    Xyz outputSizeInElements = {
        *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0)),
        *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH1)),
        *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH2)),
    };
    // For valid padding cases, the input size can be larger than the output size in the direction
    // of the pooling, so we get this value separately.
    Xyz inputSizeInElements = outputSizeInElements;
#if defined(IS_DIRECTION_X)
    inputSizeInElements.x = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH3));
#elif defined(IS_DIRECTION_Y)
    inputSizeInElements.y = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH3));
#endif
    // Number of channels to be processed by this PLE, with Z including all SRAMS and lanes.
    uint32_t numChannels  = DivRoundUp(std::max(outputSizeInElements.z, g_CeId) - g_CeId, NUM_CES);
    Xy outputSizeInGroups = DivRoundUp(Xy(outputSizeInElements), Xy::Dup(ELEMENTS_PER_GROUP_1D));
    Xy inputSizeInGroups  = DivRoundUp(Xy(inputSizeInElements), Xy::Dup(ELEMENTS_PER_GROUP_1D));

    uint32_t inDfcAddrBase  = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH4));
    uint32_t outDfcAddrBase = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH5));
    uint32_t padBefore      = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH6));
    uint32_t poolingSize    = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH7));

    udma::UdmaLoader udmaLoader(activeEvents);
    udma::UdmaStorer udmaStorer(activeEvents);

    // Set UDMA parameters which we never need to change. We load/store a single group (2x2 patches) at a time.
    udma::Params udmaParams;
    udmaParams.colGrpCountMinusOne = 0U;
    udmaParams.rowGrpCountMinusOne = 0U;
    udmaParams.colGrpStride        = 0U;    // Irrelevant as we're only copying one group at a time
    udmaParams.rowGrpStride        = 0U;    // Irrelevant as we're only copying one group at a time

    SetStoreParams<PATCHES_PER_GROUP>(udmaParams);
    SetLoadParams<PATCHES_PER_GROUP>(udmaParams);

    SetupSwizzles();

    // The distance between spatially adjacent groups in bytes
    Xyz outputGroupStrideBytes =
        Xyz(dfcsram::GetNhwcbGroupStride(outputSizeInElements) * ELEMENTS_PER_PATCH, ELEMENTS_PER_GROUP);
    Xyz inputGroupStrideBytes =
        Xyz(dfcsram::GetNhwcbGroupStride(inputSizeInElements) * ELEMENTS_PER_PATCH, ELEMENTS_PER_GROUP);

    // Process each SRAM in turn
    // Each PLE lane automatically processes a separate SRAM. We only need to program the first lane and
    // the other follows, so we skip some SRAMs
    for (unsigned dfc = 0; dfc < NUM_SRAMS; dfc += NUM_PLE_LANES)
    {
        // Default to both lanes being used
        SetPleLanesInUse(NUM_PLE_LANES);

        // Process each depth for this SRAM in turn
        unsigned depthForThisSram = DivRoundUp(std::max(numChannels, dfc) - dfc, NUM_SRAMS);
        unsigned depthForNextSram = DivRoundUp(std::max(numChannels, (dfc + 1)) - (dfc + 1), NUM_SRAMS);
        for (unsigned z = 0; z < depthForThisSram; ++z)
        {
            // If there is a second lane, but it isn't needed because this is the last pair of channels but there is an odd number, disable it
            if (z >= depthForNextSram)
            {
                SetPleLanesInUse(1U);
            }

#if defined(IS_DIRECTION_X)
            // Loop over each row
            for (unsigned y = 0; y < inputSizeInGroups.y; ++y)
            {
                uint32_t inDfcAddr  = inDfcAddrBase + z * inputGroupStrideBytes.z + y * inputGroupStrideBytes.y;
                uint32_t outDfcAddr = outDfcAddrBase + z * outputGroupStrideBytes.z + y * outputGroupStrideBytes.y;
#elif defined(IS_DIRECTION_Y)
            // Loop over each column
            for (unsigned x = 0; x < inputSizeInGroups.x; ++x)
            {
                uint32_t inDfcAddr  = inDfcAddrBase + z * inputGroupStrideBytes.z + x * inputGroupStrideBytes.x;
                uint32_t outDfcAddr = outDfcAddrBase + z * outputGroupStrideBytes.z + x * outputGroupStrideBytes.x;
#endif

                // Clear the padding area on the left side. These registers will have stale values
                // from the previous row but we need them to not contribute to the max value, so we
                // clear them to the smallest value.
                // It's quicker to set 2 adjacent 8 bit registers using 16 bit instructions
                ve_regrep_16<4>(static_cast<uint32_t>(k_SmallestValue << 8 | k_SmallestValue));
                ve_regrep_16<6>(static_cast<uint32_t>(k_SmallestValue << 8 | k_SmallestValue));

#if defined(IS_DIRECTION_X)
                // Loop over each group in the row
                for (unsigned x = 0; x < inputSizeInGroups.x; ++x)
#elif defined(IS_DIRECTION_Y)
                // Loop over each group in the column
                for (unsigned y = 0; y < inputSizeInGroups.y; ++y)
#endif
                {
                    // Load one group from regular SRAM into PLE input SRAM (at address 0)
                    udma::Address udmaInAddr;
                    udmaInAddr.dfcAddrWords = inDfcAddr / 4;
                    udmaInAddr.pleAddr      = 0;
                    udmaLoader.Load(dfc, udmaInAddr);
                    udmaLoader.WaitForUdma();

                    // Load into VE registers 8-11 (previous groups are in 0-3 and 4-7)
                    lsu::LoadInramRf<0>(dfc, { 0U, 8U * WORDS_PER_REGISTER });
                    lsu::LoadInramRf<2>(dfc, { 0U, 8U * WORDS_PER_REGISTER });

                    // Calculate the result for the previous group, now that we have the next group
                    // loaded. We skip this for the first group in the row as there is no previous group
                    // in this case.
#if defined(IS_DIRECTION_X)
                    if (x > 0)
                    {
                        CalculateAndSaveOneGroup(outDfcAddr, dfc, udmaStorer, padBefore, poolingSize);
                        outDfcAddr += outputGroupStrideBytes.x;
                    }
#elif defined(IS_DIRECTION_Y)
                    if (y > 0)
                    {
                        CalculateAndSaveOneGroup(outDfcAddr, dfc, udmaStorer, padBefore, poolingSize);
                        outDfcAddr += outputGroupStrideBytes.y;
                    }
#endif

                    // Shuffle up groups for next time
                    // It's quicker to set 2 adjacent 8 bit registers using 16 bit instructions
                    ve_mov_16<0, 4>();
                    ve_mov_16<2, 6>();

                    ve_mov_16<4, 8>();
                    ve_mov_16<6, 10>();

                    // Move to next group in regular SRAM
#if defined(IS_DIRECTION_X)
                    inDfcAddr += inputGroupStrideBytes.x;
#elif defined(IS_DIRECTION_Y)
                    inDfcAddr += inputGroupStrideBytes.y;
#endif
                }

                // After finishing looping over a row, the final group needs calculating.
                // There might not be a final group though if this is a valid padding case and
                // the IFM is larger than the OFM.
#if defined(IS_DIRECTION_X)
                if (inputSizeInGroups.x == outputSizeInGroups.x)
#elif defined(IS_DIRECTION_Y)
                if (inputSizeInGroups.y == outputSizeInGroups.y)
#endif
                {
                    // There is no "next" group for this case, so we clear those registers.
                    // It's quicker to set 2 adjacent 8 bit registers using 16 bit instructions
                    ve_regrep_16<8>(static_cast<uint32_t>(k_SmallestValue << 8 | k_SmallestValue));
                    ve_regrep_16<10>(static_cast<uint32_t>(k_SmallestValue << 8 | k_SmallestValue));
                    CalculateAndSaveOneGroup(outDfcAddr, dfc, udmaStorer, padBefore, poolingSize);
                }
            }
        }
    }

    return ncu_ple_interface::PleMsg::StripeDone{};
}

}    // namespace

extern "C" __attribute__((noreturn)) void main()
{
    EnumBitset<Event> activeEvents;
    Main([&]() { WaitForEvent<Event::SETIRQ_EVENT>(activeEvents); }, [&]() { return ProcessStripe(activeEvents); });
}
