//
// Copyright © 2017-2019,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

// These register definitions are for NPU HW version 1.4.13

namespace ethosn
{
namespace support_library
{
namespace registers
{

#if __GNUC__
// Ignore warnings about placing values into bitfields, which we cover with asserts.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

enum class dma_format_read_t
{
    NHWC      = 0,
    NHWCB     = 2,
    WEIGHTS   = 4,
    BROADCAST = 5,
    FCAF_DEEP = 6,
    FCAF_WIDE = 7,
};

enum class dma_format_write_t
{
    NHWC                   = 0,
    NHWCB                  = 2,
    NHWCB_WEIGHT_STREAMING = 3,
    FCAF_DEEP              = 6,
    FCAF_WIDE              = 7,
};

struct sram_addr_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t address : 15;
            uint32_t reserved0 : 17;
        } bits;
    };
    constexpr sram_addr_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_address() const
    {
        uint32_t value = static_cast<uint32_t>(bits.address);
        return (value << 4);
    }
    constexpr void set_address(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.address = static_cast<uint32_t>((value >> 4));
    }
};

struct dma_emcs_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t emcs : 32;
        } bits;
    };
    constexpr dma_emcs_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_emcs() const
    {
        uint32_t value = static_cast<uint32_t>(bits.emcs);
        return value;
    }
    constexpr void set_emcs(uint32_t value)
    {
        bits.emcs = static_cast<uint32_t>(value);
    }
};

struct dma_channels_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t channels : 16;
            uint32_t reserved0 : 16;
        } bits;
    };
    constexpr dma_channels_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_channels() const
    {
        uint32_t value = static_cast<uint32_t>(bits.channels);
        return (value + 1);
    }
    constexpr void set_channels(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 65536u);
        bits.channels = static_cast<uint32_t>((value - 1));
    }
};

struct dma_rd_cmd_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t rd_id : 3;
            uint32_t format : 3;
            uint32_t reserved0 : 26;
        } bits;
    };
    constexpr dma_rd_cmd_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_rd_id() const
    {
        uint32_t value = static_cast<uint32_t>(bits.rd_id);
        return value;
    }
    constexpr void set_rd_id(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 8u);
        bits.rd_id = static_cast<uint32_t>(value);
    }
    constexpr dma_format_read_t get_format() const
    {
        dma_format_read_t value = static_cast<dma_format_read_t>(bits.format);
        return value;
    }
    constexpr void set_format(dma_format_read_t value)
    {
        assert(static_cast<uint32_t>(value) < 8u);
        bits.format = static_cast<uint32_t>(value);
    }
};

struct dma_wr_cmd_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t wr_id : 3;
            uint32_t format : 3;
            uint32_t reserved0 : 26;
        } bits;
    };
    constexpr dma_wr_cmd_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_wr_id() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wr_id);
        return value;
    }
    constexpr void set_wr_id(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 8u);
        bits.wr_id = static_cast<uint32_t>(value);
    }
    constexpr dma_format_write_t get_format() const
    {
        dma_format_write_t value = static_cast<dma_format_write_t>(bits.format);
        return value;
    }
    constexpr void set_format(dma_format_write_t value)
    {
        assert(static_cast<uint32_t>(value) < 8u);
        bits.format = static_cast<uint32_t>(value);
    }
};

struct dma_stride0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t inner_stride : 32;
        } bits;
    };
    constexpr dma_stride0_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_inner_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.inner_stride);
        return (value + 1);
    }
    constexpr void set_inner_stride(uint32_t value)
    {
        bits.inner_stride = static_cast<uint32_t>((value - 1));
    }
};

struct dma_stride1_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t outer_stride : 32;
        } bits;
    };
    constexpr dma_stride1_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_outer_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.outer_stride);
        return (value + 1);
    }
    constexpr void set_outer_stride(uint32_t value)
    {
        bits.outer_stride = static_cast<uint32_t>((value - 1));
    }
};

struct dma_stride2_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t extra_stride : 32;
        } bits;
    };
    constexpr dma_stride2_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_extra_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.extra_stride);
        return (value + 1);
    }
    constexpr void set_extra_stride(uint32_t value)
    {
        bits.extra_stride = static_cast<uint32_t>((value - 1));
    }
};

struct dma_stride3_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t stride3 : 32;
        } bits;
    };
    constexpr dma_stride3_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_stride3() const
    {
        uint32_t value = static_cast<uint32_t>(bits.stride3);
        return (value + 1);
    }
    constexpr void set_stride3(uint32_t value)
    {
        bits.stride3 = static_cast<uint32_t>((value - 1));
    }
};

struct dma_sram_stride_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t sram_group_stride : 15;
            uint32_t reserved0 : 1;
            uint32_t sram_row_stride : 15;
            uint32_t reserved1 : 1;
        } bits;
    };
    constexpr dma_sram_stride_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_sram_group_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.sram_group_stride);
        return (value + 1);
    }
    constexpr void set_sram_group_stride(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 32768u);
        bits.sram_group_stride = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_sram_row_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.sram_row_stride);
        return (value + 1);
    }
    constexpr void set_sram_row_stride(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 32768u);
        bits.sram_row_stride = static_cast<uint32_t>((value - 1));
    }
};

struct dma_total_bytes_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t total_bytes : 32;
        } bits;
    };
    constexpr dma_total_bytes_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_total_bytes() const
    {
        uint32_t value = static_cast<uint32_t>(bits.total_bytes);
        return (value + 1);
    }
    constexpr void set_total_bytes(uint32_t value)
    {
        bits.total_bytes = static_cast<uint32_t>((value - 1));
    }
};

struct dma_comp_config0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t reserved0 : 23;
            uint32_t signed_activations : 1;
            uint32_t zero_point : 8;
        } bits;
    };
    constexpr dma_comp_config0_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_signed_activations() const
    {
        uint32_t value = static_cast<uint32_t>(bits.signed_activations);
        return value;
    }
    constexpr void set_signed_activations(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.signed_activations = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_zero_point() const
    {
        uint32_t value = static_cast<uint32_t>(bits.zero_point);
        return value;
    }
    constexpr void set_zero_point(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.zero_point = static_cast<uint32_t>(value);
    }
};

enum class wit_resampling_mode_t
{
    NONE             = 0,
    NEAREST_NEIGHBOR = 1,
    TRANSPOSE        = 2,
    BILINEAR         = 3,
};

enum class filter_mode_t
{
    DEPTHWISE_SEPARABLE = 0,
    FILTER_NXM          = 3,
    VECTOR_PRODUCT      = 4,
};

enum class wide_mul_mode_t
{
    WEIGHT_8_IFM_8 = 0,
};

enum class horiz_reinterleave_enable_t
{
    DISABLE = 0,
    ENABLE  = 1,
};

enum class vert_reinterleave_enable_t
{
    DISABLE = 0,
    ENABLE  = 1,
};

enum class wit_upscale_odd_height_enable_t
{
    DISABLE = 0,
    ENABLE  = 1,
};

enum class wit_upscale_odd_width_enable_t
{
    DISABLE = 0,
    ENABLE  = 1,
};

enum class wit_broadcast_mode_t
{
    ALL   = 0,
    LOCAL = 1,
};

enum class signed_ifm_mode_t
{
    DISABLE = 0,
    ENABLE  = 1,
};

enum class output_ofm_data_type_t
{
    UINT8 = 0,
    INT8  = 1,
};

enum class mceif_shuffle_pattern_t
{
    FLIPPED_N = 0,
    X_THEN_Y  = 1,
    Y_THEN_X  = 2,
};

struct ce_control_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ifm_pad_n_active : 4;
            uint32_t wide_mul_mode : 2;
            uint32_t resampling_mode : 2;
            uint32_t horiz_reinterleave_enable : 1;
            uint32_t vert_reinterleave_enable : 1;
            uint32_t upsample_2x_odd_width_enable : 1;
            uint32_t upsample_2x_odd_height_enable : 1;
            uint32_t reserved0 : 1;
            uint32_t wit_broadcast_mode : 2;
            uint32_t signed_ifm_mode : 1;
            uint32_t winograd_enable : 1;
            uint32_t relu_enable : 1;
            uint32_t ofm_bypass_enable : 1;
            uint32_t mac_acc_clr_disable : 1;
            uint32_t mac_acc_out_dis : 1;
            uint32_t output_ofm_data_type : 2;
            uint32_t reserved1 : 9;
        } bits;
    };
    constexpr ce_control_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ifm_pad_n_active() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_pad_n_active);
        return (value + 1);
    }
    constexpr void set_ifm_pad_n_active(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 16u);
        bits.ifm_pad_n_active = static_cast<uint32_t>((value - 1));
    }
    constexpr wide_mul_mode_t get_wide_mul_mode() const
    {
        wide_mul_mode_t value = static_cast<wide_mul_mode_t>(bits.wide_mul_mode);
        return value;
    }
    constexpr void set_wide_mul_mode(wide_mul_mode_t value)
    {
        assert(static_cast<uint32_t>(value) < 4u);
        bits.wide_mul_mode = static_cast<uint32_t>(value);
    }
    constexpr wit_resampling_mode_t get_resampling_mode() const
    {
        wit_resampling_mode_t value = static_cast<wit_resampling_mode_t>(bits.resampling_mode);
        return value;
    }
    constexpr void set_resampling_mode(wit_resampling_mode_t value)
    {
        assert(static_cast<uint32_t>(value) < 4u);
        bits.resampling_mode = static_cast<uint32_t>(value);
    }
    constexpr horiz_reinterleave_enable_t get_horiz_reinterleave_enable() const
    {
        horiz_reinterleave_enable_t value = static_cast<horiz_reinterleave_enable_t>(bits.horiz_reinterleave_enable);
        return value;
    }
    constexpr void set_horiz_reinterleave_enable(horiz_reinterleave_enable_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.horiz_reinterleave_enable = static_cast<uint32_t>(value);
    }
    constexpr vert_reinterleave_enable_t get_vert_reinterleave_enable() const
    {
        vert_reinterleave_enable_t value = static_cast<vert_reinterleave_enable_t>(bits.vert_reinterleave_enable);
        return value;
    }
    constexpr void set_vert_reinterleave_enable(vert_reinterleave_enable_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.vert_reinterleave_enable = static_cast<uint32_t>(value);
    }
    constexpr wit_upscale_odd_width_enable_t get_upsample_2x_odd_width_enable() const
    {
        wit_upscale_odd_width_enable_t value =
            static_cast<wit_upscale_odd_width_enable_t>(bits.upsample_2x_odd_width_enable);
        return value;
    }
    constexpr void set_upsample_2x_odd_width_enable(wit_upscale_odd_width_enable_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.upsample_2x_odd_width_enable = static_cast<uint32_t>(value);
    }
    constexpr wit_upscale_odd_height_enable_t get_upsample_2x_odd_height_enable() const
    {
        wit_upscale_odd_height_enable_t value =
            static_cast<wit_upscale_odd_height_enable_t>(bits.upsample_2x_odd_height_enable);
        return value;
    }
    constexpr void set_upsample_2x_odd_height_enable(wit_upscale_odd_height_enable_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.upsample_2x_odd_height_enable = static_cast<uint32_t>(value);
    }
    constexpr wit_broadcast_mode_t get_wit_broadcast_mode() const
    {
        wit_broadcast_mode_t value = static_cast<wit_broadcast_mode_t>(bits.wit_broadcast_mode);
        return value;
    }
    constexpr void set_wit_broadcast_mode(wit_broadcast_mode_t value)
    {
        assert(static_cast<uint32_t>(value) < 4u);
        bits.wit_broadcast_mode = static_cast<uint32_t>(value);
    }
    constexpr signed_ifm_mode_t get_signed_ifm_mode() const
    {
        signed_ifm_mode_t value = static_cast<signed_ifm_mode_t>(bits.signed_ifm_mode);
        return value;
    }
    constexpr void set_signed_ifm_mode(signed_ifm_mode_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.signed_ifm_mode = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_winograd_enable() const
    {
        uint32_t value = static_cast<uint32_t>(bits.winograd_enable);
        return value;
    }
    constexpr void set_winograd_enable(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.winograd_enable = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_relu_enable() const
    {
        uint32_t value = static_cast<uint32_t>(bits.relu_enable);
        return value;
    }
    constexpr void set_relu_enable(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.relu_enable = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_ofm_bypass_enable() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ofm_bypass_enable);
        return value;
    }
    constexpr void set_ofm_bypass_enable(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.ofm_bypass_enable = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_mac_acc_clr_disable() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mac_acc_clr_disable);
        return value;
    }
    constexpr void set_mac_acc_clr_disable(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.mac_acc_clr_disable = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_mac_acc_out_dis() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mac_acc_out_dis);
        return value;
    }
    constexpr void set_mac_acc_out_dis(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.mac_acc_out_dis = static_cast<uint32_t>(value);
    }
    constexpr output_ofm_data_type_t get_output_ofm_data_type() const
    {
        output_ofm_data_type_t value = static_cast<output_ofm_data_type_t>(bits.output_ofm_data_type);
        return value;
    }
    constexpr void set_output_ofm_data_type(output_ofm_data_type_t value)
    {
        assert(static_cast<uint32_t>(value) < 4u);
        bits.output_ofm_data_type = static_cast<uint32_t>(value);
    }
};

struct wide_kernel_control_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t wide_kernel_enable : 1;
            uint32_t wide_filter_width : 8;
            uint32_t wide_filter_height : 8;
            uint32_t reserved0 : 15;
        } bits;
    };
    constexpr wide_kernel_control_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_wide_kernel_enable() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wide_kernel_enable);
        return value;
    }
    constexpr void set_wide_kernel_enable(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.wide_kernel_enable = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_wide_filter_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wide_filter_width);
        return (value + 1);
    }
    constexpr void set_wide_filter_width(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 256u);
        bits.wide_filter_width = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_wide_filter_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wide_filter_height);
        return (value + 1);
    }
    constexpr void set_wide_filter_height(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 256u);
        bits.wide_filter_height = static_cast<uint32_t>((value - 1));
    }
};

struct wide_kernel_offset_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t wide_filter_offset_w : 8;
            uint32_t wide_filter_offset_h : 8;
            uint32_t wide_delta_width : 8;
            uint32_t wide_delta_height : 8;
        } bits;
    };
    constexpr wide_kernel_offset_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_wide_filter_offset_w() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wide_filter_offset_w);
        return value;
    }
    constexpr void set_wide_filter_offset_w(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.wide_filter_offset_w = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_wide_filter_offset_h() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wide_filter_offset_h);
        return value;
    }
    constexpr void set_wide_filter_offset_h(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.wide_filter_offset_h = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_wide_delta_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wide_delta_width);
        return value;
    }
    constexpr void set_wide_delta_width(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.wide_delta_width = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_wide_delta_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.wide_delta_height);
        return value;
    }
    constexpr void set_wide_delta_height(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.wide_delta_height = static_cast<uint32_t>(value);
    }
};

struct ifm_zero_point_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t zero_point : 8;
            uint32_t reserved0 : 24;
        } bits;
    };
    constexpr ifm_zero_point_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_zero_point() const
    {
        uint32_t value = static_cast<uint32_t>(bits.zero_point);
        return value;
    }
    constexpr void set_zero_point(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.zero_point = static_cast<uint32_t>(value);
    }
};

struct ifm_default_slot_size_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ifm_default_slot_width : 16;
            uint32_t ifm_default_slot_height : 16;
        } bits;
    };
    constexpr ifm_default_slot_size_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ifm_default_slot_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_default_slot_width);
        return value;
    }
    constexpr void set_ifm_default_slot_width(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.ifm_default_slot_width = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_ifm_default_slot_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_default_slot_height);
        return value;
    }
    constexpr void set_ifm_default_slot_height(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.ifm_default_slot_height = static_cast<uint32_t>(value);
    }
};

struct ifm_slot_stride_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ifm_default_slot_stride : 15;
            uint32_t reserved0 : 1;
            uint32_t ifm_boundary_slot_stride : 15;
            uint32_t reserved1 : 1;
        } bits;
    };
    constexpr ifm_slot_stride_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ifm_default_slot_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_default_slot_stride);
        return (value << 4);
    }
    constexpr void set_ifm_default_slot_stride(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.ifm_default_slot_stride = static_cast<uint32_t>((value >> 4));
    }
    constexpr uint32_t get_ifm_boundary_slot_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_boundary_slot_stride);
        return (value << 4);
    }
    constexpr void set_ifm_boundary_slot_stride(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.ifm_boundary_slot_stride = static_cast<uint32_t>((value >> 4));
    }
};

struct ifm_row_stride_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ifm_default_row_stride : 15;
            uint32_t reserved0 : 1;
            uint32_t ifm_residual_row_stride : 15;
            uint32_t reserved1 : 1;
        } bits;
    };
    constexpr ifm_row_stride_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ifm_default_row_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_default_row_stride);
        return (value << 4);
    }
    constexpr void set_ifm_default_row_stride(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.ifm_default_row_stride = static_cast<uint32_t>((value >> 4));
    }
    constexpr uint32_t get_ifm_residual_row_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_residual_row_stride);
        return (value << 4);
    }
    constexpr void set_ifm_residual_row_stride(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.ifm_residual_row_stride = static_cast<uint32_t>((value >> 4));
    }
};

struct ifm_config1_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ifm_group_stride : 15;
            uint32_t reserved0 : 1;
            uint32_t num_ifm_global : 16;
        } bits;
    };
    constexpr ifm_config1_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ifm_group_stride() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_group_stride);
        return (value << 4);
    }
    constexpr void set_ifm_group_stride(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.ifm_group_stride = static_cast<uint32_t>((value >> 4));
    }
    constexpr uint32_t get_num_ifm_global() const
    {
        uint32_t value = static_cast<uint32_t>(bits.num_ifm_global);
        return value;
    }
    constexpr void set_num_ifm_global(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.num_ifm_global = static_cast<uint32_t>(value);
    }
};

struct ifm_top_slots_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t top_left_slot : 4;
            uint32_t top_left_residual : 1;
            uint32_t reserved0 : 3;
            uint32_t top_center_slot : 4;
            uint32_t top_center_residual : 1;
            uint32_t reserved1 : 3;
            uint32_t top_right_slot : 4;
            uint32_t top_right_residual : 1;
            uint32_t reserved2 : 11;
        } bits;
    };
    constexpr ifm_top_slots_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_top_left_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_left_slot);
        return value;
    }
    constexpr void set_top_left_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.top_left_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_top_left_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_left_residual);
        return value;
    }
    constexpr void set_top_left_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.top_left_residual = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_top_center_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_center_slot);
        return value;
    }
    constexpr void set_top_center_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.top_center_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_top_center_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_center_residual);
        return value;
    }
    constexpr void set_top_center_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.top_center_residual = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_top_right_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_right_slot);
        return value;
    }
    constexpr void set_top_right_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.top_right_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_top_right_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_right_residual);
        return value;
    }
    constexpr void set_top_right_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.top_right_residual = static_cast<uint32_t>(value);
    }
};

struct ifm_mid_slots_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t mid_left_slot : 4;
            uint32_t mid_left_residual : 1;
            uint32_t reserved0 : 3;
            uint32_t mid_center_slot : 4;
            uint32_t mid_center_residual : 1;
            uint32_t reserved1 : 3;
            uint32_t mid_right_slot : 4;
            uint32_t mid_right_residual : 1;
            uint32_t reserved2 : 11;
        } bits;
    };
    constexpr ifm_mid_slots_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_mid_left_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mid_left_slot);
        return value;
    }
    constexpr void set_mid_left_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.mid_left_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_mid_left_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mid_left_residual);
        return value;
    }
    constexpr void set_mid_left_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.mid_left_residual = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_mid_center_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mid_center_slot);
        return value;
    }
    constexpr void set_mid_center_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.mid_center_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_mid_center_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mid_center_residual);
        return value;
    }
    constexpr void set_mid_center_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.mid_center_residual = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_mid_right_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mid_right_slot);
        return value;
    }
    constexpr void set_mid_right_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.mid_right_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_mid_right_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mid_right_residual);
        return value;
    }
    constexpr void set_mid_right_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.mid_right_residual = static_cast<uint32_t>(value);
    }
};

struct ifm_bottom_slots_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t bottom_left_slot : 4;
            uint32_t bottom_left_residual : 1;
            uint32_t reserved0 : 3;
            uint32_t bottom_center_slot : 4;
            uint32_t bottom_center_residual : 1;
            uint32_t reserved1 : 3;
            uint32_t bottom_right_slot : 4;
            uint32_t bottom_right_residual : 1;
            uint32_t reserved2 : 11;
        } bits;
    };
    constexpr ifm_bottom_slots_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_bottom_left_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.bottom_left_slot);
        return value;
    }
    constexpr void set_bottom_left_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.bottom_left_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_bottom_left_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.bottom_left_residual);
        return value;
    }
    constexpr void set_bottom_left_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.bottom_left_residual = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_bottom_center_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.bottom_center_slot);
        return value;
    }
    constexpr void set_bottom_center_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.bottom_center_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_bottom_center_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.bottom_center_residual);
        return value;
    }
    constexpr void set_bottom_center_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.bottom_center_residual = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_bottom_right_slot() const
    {
        uint32_t value = static_cast<uint32_t>(bits.bottom_right_slot);
        return value;
    }
    constexpr void set_bottom_right_slot(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.bottom_right_slot = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_bottom_right_residual() const
    {
        uint32_t value = static_cast<uint32_t>(bits.bottom_right_residual);
        return value;
    }
    constexpr void set_bottom_right_residual(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.bottom_right_residual = static_cast<uint32_t>(value);
    }
};

struct ifm_slot_pad_config_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t left_data : 1;
            uint32_t right_data : 1;
            uint32_t top_data : 1;
            uint32_t bottom_data : 1;
            uint32_t reserved0 : 28;
        } bits;
    };
    constexpr ifm_slot_pad_config_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_left_data() const
    {
        uint32_t value = static_cast<uint32_t>(bits.left_data);
        return value;
    }
    constexpr void set_left_data(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.left_data = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_right_data() const
    {
        uint32_t value = static_cast<uint32_t>(bits.right_data);
        return value;
    }
    constexpr void set_right_data(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.right_data = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_top_data() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_data);
        return value;
    }
    constexpr void set_top_data(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.top_data = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_bottom_data() const
    {
        uint32_t value = static_cast<uint32_t>(bits.bottom_data);
        return value;
    }
    constexpr void set_bottom_data(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.bottom_data = static_cast<uint32_t>(value);
    }
};

struct depthwise_control_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t num_ifms_per_ofm : 8;
            uint32_t reserved0 : 24;
        } bits;
    };
    constexpr depthwise_control_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_num_ifms_per_ofm() const
    {
        uint32_t value = static_cast<uint32_t>(bits.num_ifms_per_ofm);
        return value;
    }
    constexpr void set_num_ifms_per_ofm(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.num_ifms_per_ofm = static_cast<uint32_t>(value);
    }
};

struct ifm_config2_ig0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t num_ifm_local : 16;
            uint32_t reserved0 : 16;
        } bits;
    };
    constexpr ifm_config2_ig0_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_num_ifm_local() const
    {
        uint32_t value = static_cast<uint32_t>(bits.num_ifm_local);
        return value;
    }
    constexpr void set_num_ifm_local(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.num_ifm_local = static_cast<uint32_t>(value);
    }
};

struct ifm_slot_base_address_ig0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ifm_slot_base_addr : 15;
            uint32_t reserved0 : 1;
            uint32_t ifm_slot_base_addr_hi : 15;
            uint32_t reserved1 : 1;
        } bits;
    };
    constexpr ifm_slot_base_address_ig0_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ifm_slot_base_addr() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_slot_base_addr);
        return (value << 4);
    }
    constexpr void set_ifm_slot_base_addr(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.ifm_slot_base_addr = static_cast<uint32_t>((value >> 4));
    }
    constexpr uint32_t get_ifm_slot_base_addr_hi() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ifm_slot_base_addr_hi);
        return (value << 4);
    }
    constexpr void set_ifm_slot_base_addr_hi(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.ifm_slot_base_addr_hi = static_cast<uint32_t>((value >> 4));
    }
};

struct ifm_pad0_ig0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t left_pad : 3;
            uint32_t top_pad : 3;
            int32_t ifm_stripe_width_delta : 5;
            int32_t ifm_stripe_height_delta : 5;
            uint32_t reserved0 : 16;
        } bits;
    };
    constexpr ifm_pad0_ig0_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_left_pad() const
    {
        uint32_t value = static_cast<uint32_t>(bits.left_pad);
        return value;
    }
    constexpr void set_left_pad(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 8u);
        bits.left_pad = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_top_pad() const
    {
        uint32_t value = static_cast<uint32_t>(bits.top_pad);
        return value;
    }
    constexpr void set_top_pad(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 8u);
        bits.top_pad = static_cast<uint32_t>(value);
    }
    constexpr int32_t get_ifm_stripe_width_delta() const
    {
        int32_t value = static_cast<int32_t>(bits.ifm_stripe_width_delta);
        return value;
    }
    constexpr void set_ifm_stripe_width_delta(int32_t value)
    {
        assert(static_cast<int32_t>(value) >= -16 && static_cast<int32_t>(value) <= 15);
        bits.ifm_stripe_width_delta = static_cast<int32_t>(value);
    }
    constexpr int32_t get_ifm_stripe_height_delta() const
    {
        int32_t value = static_cast<int32_t>(bits.ifm_stripe_height_delta);
        return value;
    }
    constexpr void set_ifm_stripe_height_delta(int32_t value)
    {
        assert(static_cast<int32_t>(value) >= -16 && static_cast<int32_t>(value) <= 15);
        bits.ifm_stripe_height_delta = static_cast<int32_t>(value);
    }
};

struct activation_config_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t relu_min : 16;
            uint32_t relu_max : 16;
        } bits;
    };
    constexpr activation_config_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_relu_min() const
    {
        uint32_t value = static_cast<uint32_t>(bits.relu_min);
        return value;
    }
    constexpr void set_relu_min(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.relu_min = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_relu_max() const
    {
        uint32_t value = static_cast<uint32_t>(bits.relu_max);
        return value;
    }
    constexpr void set_relu_max(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.relu_max = static_cast<uint32_t>(value);
    }
};

struct stripe_block_config_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ofm_default_block_width : 6;
            uint32_t ofm_default_block_height : 6;
            uint32_t ofm_bypass_half_patch_output_type : 1;
            uint32_t reserved0 : 11;
            uint32_t mceif_shuffle_pattern : 4;
            uint32_t reserved1 : 4;
        } bits;
    };
    constexpr stripe_block_config_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ofm_default_block_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ofm_default_block_width);
        return value;
    }
    constexpr void set_ofm_default_block_width(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 64u);
        bits.ofm_default_block_width = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_ofm_default_block_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ofm_default_block_height);
        return value;
    }
    constexpr void set_ofm_default_block_height(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 64u);
        bits.ofm_default_block_height = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_ofm_bypass_half_patch_output_type() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ofm_bypass_half_patch_output_type);
        return value;
    }
    constexpr void set_ofm_bypass_half_patch_output_type(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 2u);
        bits.ofm_bypass_half_patch_output_type = static_cast<uint32_t>(value);
    }
    constexpr mceif_shuffle_pattern_t get_mceif_shuffle_pattern() const
    {
        mceif_shuffle_pattern_t value = static_cast<mceif_shuffle_pattern_t>(bits.mceif_shuffle_pattern);
        return value;
    }
    constexpr void set_mceif_shuffle_pattern(mceif_shuffle_pattern_t value)
    {
        assert(static_cast<uint32_t>(value) < 16u);
        bits.mceif_shuffle_pattern = static_cast<uint32_t>(value);
    }
};

struct ofm_stripe_size_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ofm_stripe_width : 16;
            uint32_t ofm_stripe_height : 16;
        } bits;
    };
    constexpr ofm_stripe_size_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_ofm_stripe_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ofm_stripe_width);
        return value;
    }
    constexpr void set_ofm_stripe_width(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.ofm_stripe_width = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_ofm_stripe_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.ofm_stripe_height);
        return value;
    }
    constexpr void set_ofm_stripe_height(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.ofm_stripe_height = static_cast<uint32_t>(value);
    }
};

struct ofm_config_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t num_ofm : 16;
            uint32_t reserved0 : 16;
        } bits;
    };
    constexpr ofm_config_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_num_ofm() const
    {
        uint32_t value = static_cast<uint32_t>(bits.num_ofm);
        return value;
    }
    constexpr void set_num_ofm(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 65536u);
        bits.num_ofm = static_cast<uint32_t>(value);
    }
};

struct filter_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t filter_mode : 3;
            uint32_t filter0_width : 3;
            uint32_t filter0_height : 3;
            uint32_t filter1_width : 3;
            uint32_t filter1_height : 3;
            uint32_t filter2_width : 3;
            uint32_t filter2_height : 3;
            uint32_t filter3_width : 3;
            uint32_t filter3_height : 3;
            uint32_t reserved0 : 5;
        } bits;
    };
    constexpr filter_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr filter_mode_t get_filter_mode() const
    {
        filter_mode_t value = static_cast<filter_mode_t>(bits.filter_mode);
        return value;
    }
    constexpr void set_filter_mode(filter_mode_t value)
    {
        assert(static_cast<uint32_t>(value) < 8u);
        bits.filter_mode = static_cast<uint32_t>(value);
    }
    constexpr uint32_t get_filter0_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter0_width);
        return (value + 1);
    }
    constexpr void set_filter0_width(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter0_width = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_filter0_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter0_height);
        return (value + 1);
    }
    constexpr void set_filter0_height(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter0_height = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_filter1_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter1_width);
        return (value + 1);
    }
    constexpr void set_filter1_width(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter1_width = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_filter1_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter1_height);
        return (value + 1);
    }
    constexpr void set_filter1_height(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter1_height = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_filter2_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter2_width);
        return (value + 1);
    }
    constexpr void set_filter2_width(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter2_width = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_filter2_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter2_height);
        return (value + 1);
    }
    constexpr void set_filter2_height(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter2_height = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_filter3_width() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter3_width);
        return (value + 1);
    }
    constexpr void set_filter3_width(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter3_width = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_filter3_height() const
    {
        uint32_t value = static_cast<uint32_t>(bits.filter3_height);
        return (value + 1);
    }
    constexpr void set_filter3_height(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 8u);
        bits.filter3_height = static_cast<uint32_t>((value - 1));
    }
};

struct mul_enable_og0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t mul_enable : 32;
        } bits;
    };
    constexpr mul_enable_og0_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_mul_enable() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mul_enable);
        return value;
    }
    constexpr void set_mul_enable(uint32_t value)
    {
        bits.mul_enable = static_cast<uint32_t>(value);
    }
};

struct weight_base_addr_og0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t address : 15;
            uint32_t reserved0 : 17;
        } bits;
    };
    constexpr weight_base_addr_og0_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_address() const
    {
        uint32_t value = static_cast<uint32_t>(bits.address);
        return (value << 4);
    }
    constexpr void set_address(uint32_t value)
    {
        assert(static_cast<uint32_t>((value >> 4)) < 32768u);
        bits.address = static_cast<uint32_t>((value >> 4));
    }
};

struct ple_mceif_config_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t mceif_num_bufs : 4;
            uint32_t mceif_buf_size : 8;
            uint32_t mceif_buf_base : 8;
            uint32_t reserved0 : 12;
        } bits;
    };
    constexpr ple_mceif_config_r(uint32_t init = 0)
        : word(init)
    {}
    constexpr uint32_t get_mceif_num_bufs() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mceif_num_bufs);
        return (value + 1);
    }
    constexpr void set_mceif_num_bufs(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 16u);
        bits.mceif_num_bufs = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_mceif_buf_size() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mceif_buf_size);
        return (value + 1);
    }
    constexpr void set_mceif_buf_size(uint32_t value)
    {
        assert(static_cast<uint32_t>((value - 1)) < 256u);
        bits.mceif_buf_size = static_cast<uint32_t>((value - 1));
    }
    constexpr uint32_t get_mceif_buf_base() const
    {
        uint32_t value = static_cast<uint32_t>(bits.mceif_buf_base);
        return value;
    }
    constexpr void set_mceif_buf_base(uint32_t value)
    {
        assert(static_cast<uint32_t>(value) < 256u);
        bits.mceif_buf_base = static_cast<uint32_t>(value);
    }
};

#if __GNUC__
#pragma GCC diagnostic pop
#endif

}    // namespace registers
}    // namespace support_library
}    // namespace ethosn
