// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifdef PPL_USE_ARM_SERVER_FP16

#include "ppl/kernel/arm_server/conv2d/neon/fp16/direct/conv2d_n8cx_direct_fp16.h"

#include <arm_neon.h>
#include <chrono>
#include <new>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#if defined PPL_USE_ARM_SERVER_OMP
#include <omp.h>
#endif

#include "ppl/common/arm/sysinfo.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

#include "ppl/kernel/arm_server/conv2d/neon/fp16/utils/conv2d_utils_fp16.h"

namespace ppl { namespace kernel { namespace arm_server {

#define CBLK() 8

#define OUT_TILE_W() 10
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 9
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 8
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 7
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 6
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 5
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 4
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 3
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 2
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 1
#include "conv_direct_kernel_fp16.inc"
#undef OUT_TILE_W

typedef void (*ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t)(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    int64_t ic_block_ceil8,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t flt_diff_w_x_icv_x_ocs_bytes,
    const int64_t flt_diff_h_x_flt_w_x_icv_x_ocs_bytes,
    const int64_t dst_h_x_dst_w_x_ocv_bytes,
    const int64_t src_h_x_src_w_x_icv_bytes,
    const int64_t dltn_h_x_src_w_x_icv_bytes,
    const int64_t dltn_w_x_icv_bytes,
    const int64_t strd_w_x_icv_bytes,
    const uint32_t fuse_flag);

#define OW_CASE() 10
static ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t ppl_arm_server_kernel_fp16_conv_gen_direct_kernels_oc16[OW_CASE() + 1] =
    {
        nullptr,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow1_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow2_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow3_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow4_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow5_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow6_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow7_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow8_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow9_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc16_oh1_ow10_ext_layout_asm,
};
#undef OW_CASE

#define OW_CASE() 10
static ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t ppl_arm_server_kernel_fp16_conv_gen_direct_kernels_oc8[OW_CASE() + 1] =
    {
        nullptr,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow1_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow2_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow3_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow4_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow5_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow6_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow7_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow8_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow9_ext_layout_asm,
        ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_oc8_oh1_ow10_ext_layout_asm,
};
#undef OW_CASE

static void load_n8cx_input_tile(
    const __fp16 *input,
    __fp16 *tmp_buffer,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t ih_start,
    const int64_t ih_end,
    const int64_t iw_start,
    const int64_t iw_end,
    const int64_t ic_start,
    const int64_t ic_end)
{
    (void)load_n8cx_input_tile;
    static const float16x8_t k_zeros = {(__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f};

    const int64_t src_h_x_src_w       = src_h * src_w;
    const int64_t src_w_x_8         = src_w * 8;
    const int64_t itile_h_x_itile_w = (ih_end - ih_start) * (iw_end - iw_start);
    const int64_t itile_w_x_8      = (iw_end - iw_start) * 8;

    const bool in_pad_right    = (iw_end > src_w);
    const int64_t in_valid_end = ppl::kernel::arm_server::min(src_w, iw_end);

    if (ic_start % 8 != 0) std::abort();

    for (int64_t ic = ic_start, itc = 0; ic < ic_end; ic += 8, itc += 8) {
        for (int64_t ih = ih_start, ith = 0; ih < ih_end; ih++, ith++) {
            __fp16 *const buffer_row_base_ptr = tmp_buffer + itc * itile_h_x_itile_w + ith * itile_w_x_8;
            if (ih < 0 || ih >= src_h) {
                // #pragma unroll 4
                for (int64_t iw = iw_start, itw_x_8 = 0; iw < iw_end; iw++, itw_x_8 += 8) {
                    vst1q_f16(buffer_row_base_ptr + itw_x_8, k_zeros);
                }
            } else {
                const __fp16 *const input_row_base_ptr = input + ic * src_h_x_src_w + ih * src_w_x_8;
                int64_t iw                             = iw_start;
                int64_t itw_x_8                        = 0;
                for (; iw < 0; iw++, itw_x_8 += 8) {
                    vst1q_f16(buffer_row_base_ptr + itw_x_8, k_zeros);
                }
                // #pragma unroll 4
                for (; iw < in_valid_end; iw++, itw_x_8 += 8) {
                    vst1q_f16(buffer_row_base_ptr + itw_x_8, vld1q_f16(input_row_base_ptr + iw * 8));
                }
                if (in_pad_right) {
                    for (; iw < iw_end; iw++, itw_x_8 += 8) {
                        vst1q_f16(buffer_row_base_ptr + itw_x_8, k_zeros);
                    }
                }
            }
        }
    }
}

uint64_t conv2d_n8cx_direct_fp16_runtime_executor::get_padding_buffer_size()
{
    const int64_t num_threads                        = PPL_OMP_MAX_THREADS();
    const conv2d_param &cp                           = *conv_param_;
    const conv2d_n8cx_direct_fp16_schedule_param &sp = sched_param_;
    const int64_t src_h                              = src_shape_->GetDim(2);
    const int64_t src_w                              = src_shape_->GetDim(3);
    const int64_t dst_w                              = dst_shape_->GetDim(3);
    if (cp.pad_h == 0 && cp.pad_w == 0) return 0;

    // Local padding buffer
    const int64_t oth = sp.oh_blk;
    const int64_t otw = sp.ow_blk;
    const int64_t ics = sp.ic_blk;

    const int64_t itile_h      = (oth - 1) * cp.stride_h + cp.dilation_h * (cp.kernel_h - 1) + 1;
    const int64_t itile_w      = (otw - 1) * cp.stride_w + cp.dilation_w * (cp.kernel_w - 1) + 1;
    const int64_t in_tile_num = ics * itile_h * itile_w;
    const size_t in_tile_size = in_tile_num * sizeof(__fp16);

    uint64_t local_padding_buffer_size  = CEIL128(in_tile_size) + 128;
    uint64_t global_padding_buffer_size = CEIL128(ics * (src_h + 2 * cp.pad_h) * (src_w + 2 * cp.pad_w) * sizeof(__fp16)) + 128;

    if (cp.pad_w > 0 && dst_w <= (otw + (otw + 1) / 2)) { // TODO: Check against the cache size after memory-footprint64_t analysis
        return global_padding_buffer_size * num_threads;
    }

    return local_padding_buffer_size * num_threads;
}

uint64_t conv2d_n8cx_direct_fp16_runtime_executor::cal_temp_buffer_size()
{
    const conv2d_param &cp = *conv_param_;
    const int64_t ic_g_pck = CEIL8(cp.channels / cp.group);
    const int64_t oc_g_pck = CEIL8(cp.num_output / cp.group);

    const int64_t src_h     = src_shape_->GetDim(2);
    const int64_t src_w     = src_shape_->GetDim(3);
    const int64_t dst_h     = dst_shape_->GetDim(2);
    const int64_t dst_w     = dst_shape_->GetDim(3);
    const int64_t num_batch = src_shape_->GetDim(0);

    uint64_t input_gbuf_size  = num_batch * ic_g_pck * src_h * src_w * sizeof(__fp16);
    uint64_t output_gbuf_size = num_batch * oc_g_pck * dst_h * dst_w * sizeof(__fp16);
    // uint64_t padding_buffer_size = get_padding_buffer_size();

    return input_gbuf_size + output_gbuf_size;
}

void conv2d_n8cx_direct_fp16_runtime_executor::adjust_schedule_param()
{
    const int64_t num_threads                   = PPL_OMP_MAX_THREADS();
    sched_param_.padding_buffer_size_per_thread = get_padding_buffer_size() / num_threads;
    return;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_runtime_executor::execute()
{
    const conv2d_param &cp                           = *conv_param_;
    const conv2d_n8cx_direct_fp16_schedule_param &sp = sched_param_;
    const __fp16 *input                              = (const __fp16 *)src_;
    const __fp16 *cvt_filter                         = (const __fp16 *)cvt_filter_;
    const __fp16 *bias                               = (const __fp16 *)cvt_bias_;
    __fp16 *output                                   = (__fp16 *)dst_;
    __fp16 *sum                                      = (__fp16 *)sum_;
    __fp16 *tmp_buffer                               = (__fp16 *)temp_buffer_;
    const int64_t src_h                                = src_shape_->GetDim(2);
    const int64_t src_w                                = src_shape_->GetDim(3);
    const int64_t inC                                = src_shape_->GetDim(1);
    const int64_t outC                               = cp.num_output;
    const int64_t dst_h                               = dst_shape_->GetDim(2);
    const int64_t dst_w                               = dst_shape_->GetDim(3);
    const int64_t flt_h                               = cp.kernel_h;
    const int64_t flt_w                               = cp.kernel_w;
    const int64_t padH                               = cp.pad_h;
    const int64_t padW                               = cp.pad_w;
    const int64_t strd_h                              = cp.stride_h;
    const int64_t strd_w                              = cp.stride_w;
    const int64_t dltn_h                              = cp.dilation_h;
    const int64_t dltn_w                              = cp.dilation_w;
    // const int64_t single_core_padding_buffer_offset = sp.padding_buffer_size_per_thread / sizeof(__fp16);
    const int64_t num_batch                          = src_shape_->GetDim(0);

    PRAGMA_OMP_PARALLEL()
    {
        const int64_t src_c_pck = CEIL8(inC);
        const int64_t dst_c_pck = CEIL8(outC);

        const int64_t ic_group = inC / cp.group;
        const int64_t oc_group = outC / cp.group;
        const int64_t ic_g_pck = CEIL8(ic_group);
        const int64_t oc_g_pck = CEIL8(oc_group);

        const int64_t oth = sp.oh_blk;
        const int64_t otw = sp.ow_blk;

        const int64_t ocs = sp.oc_blk;
        const int64_t ics = sp.ic_blk;

        const int64_t icv_bytes = 16; // 8 * sizeof(__fp16);
        const int64_t ocv_bytes = 16; // 8 * sizeof(__fp16);

        const int64_t src_h_x_src_w_x_icv_bytes   = src_h * src_w * icv_bytes;
        const int64_t dst_h_x_dst_w_x_ocv_bytes = dst_h * dst_w * ocv_bytes;
        const int64_t dltn_h_x_icv_bytes       = dltn_h * icv_bytes;
        const int64_t dltn_w_x_icv_bytes       = dltn_w * icv_bytes;
        const int64_t strd_w_x_icv_bytes       = strd_w * icv_bytes;
        const int64_t dltn_h_x_src_w_x_icv_bytes = src_w * dltn_h_x_icv_bytes;
        const int64_t icv_x_ocs_bytes         = ocs * icv_bytes;
        const int64_t flt_w_x_icv_x_ocs_bytes  = flt_w * ocs * icv_bytes;

        const int64_t single_batch_input_size  = src_c_pck * src_h * src_w;
        const int64_t single_batch_output_size = dst_c_pck * dst_h * dst_w;

        const bool use_in_gbuf                   = (cp.group > 1 && ic_g_pck != ic_group);
        const bool use_out_gbuf                  = (cp.group > 1 && oc_g_pck != oc_group);
        const int64_t input_group_buffer_offset  = num_batch * ic_g_pck * src_h * src_w;
        __fp16 *input_gbuf                       = tmp_buffer;
        __fp16 *output_gbuf                      = input_gbuf + input_group_buffer_offset;

        const int64_t oc_l1_s = 64;
        (void)oc_l1_s;
        const int64_t ic_l1_s = 128;
        (void)ic_l1_s;

        int64_t ow_inner_start = std::max((int64_t)0, DIV_CEIL((padW - 0 * dltn_w), strd_w)); // inclusive
        int64_t ow_inner_end   = std::min((int64_t)dst_w, DIV_CEIL((src_w + padW - (flt_w - 1) * dltn_w), strd_w)); // exclusive
        ow_inner_start         = std::min(ow_inner_start, dst_w);
        ow_inner_end           = std::max(ow_inner_end, ow_inner_start);

        uint32_t kernel_fuse_type = cp.fuse_flag;
        if (use_out_gbuf && (cp.fuse_flag & conv_fuse_flag::SUM)) {
            kernel_fuse_type = conv_fuse_flag::NONE;
        }

        for (int64_t g = 0; g < cp.group; g++) {
            int64_t in_b_stride  = single_batch_input_size;
            int64_t out_b_stride = single_batch_output_size;

            const __fp16 *cvt_filter_g_base = cvt_filter + g * CEIL(oc_group, ocs) * ic_g_pck * flt_h * flt_w;
            const __fp16 *bias_g_base       = bias + g * oc_group;

            const __fp16 *kernel_input = input + g * ic_group * src_h * src_w;
            __fp16 *kernel_output      = output + g * oc_group * dst_h * dst_w;
            if (use_in_gbuf) {
                in_b_stride  = ic_g_pck * src_h * src_w;
                kernel_input = input_gbuf;
                for (int64_t b = 0; b < num_batch; b++) {
                    conv2d_n8cx_load_group_fp16(
                        input + b * single_batch_input_size,
                        input_gbuf + b * in_b_stride,
                        src_h * src_w,
                        ic_group,
                        g,
                        0);
                }
                PRAGMA_OMP_BARRIER()
            }
            if (use_out_gbuf) {
                out_b_stride  = oc_g_pck * dst_h * dst_w;
                kernel_output = output_gbuf;
            }
#if not defined PPL_USE_ARM_SERVER_OMP
            for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
                const __fp16 *input_batch_base_ptr = kernel_input + batch_id * in_b_stride;
                __fp16 *output_batch_base_ptr      = kernel_output + batch_id * out_b_stride;
                __fp16 *sum_bg_base_ptr            = sum + batch_id * single_batch_output_size + g * oc_group * dst_h * dst_w; // CAVEATS: single_batch_output_size ?! out_b_stride
                for (int64_t ic_l1 = 0; ic_l1 < ic_g_pck; ic_l1 += ics) {
                    const int64_t ic_remain  = ppl::kernel::arm_server::min(ics, ic_g_pck - ic_l1);
                    const uint32_t fuse_flag = (ic_l1 + ics >= ic_g_pck) ? kernel_fuse_type : (const uint32_t)conv_fuse_flag::NONE;
                    for (int64_t oc_l1 = 0; oc_l1 < oc_g_pck; oc_l1 += ocs) {
                        const __fp16 *const bias_ptr = (ic_l1 == 0) ? (bias_g_base + oc_l1) : nullptr;
                        const int64_t oc_remains     = ppl::kernel::arm_server::min(ocs, oc_g_pck - oc_l1);
                        const ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t *const fp16_conv_gd_kernel =
                            (oc_remains > 8) ? ppl_arm_server_kernel_fp16_conv_gen_direct_kernels_oc16 : ppl_arm_server_kernel_fp16_conv_gen_direct_kernels_oc8;
                        for (int64_t oh = 0; oh < dst_h; oh += oth) {
#else
            for (int64_t ic_l1 = 0; ic_l1 < ic_g_pck; ic_l1 += ics) {
                const uint32_t fuse_flag = (ic_l1 + ics >= ic_g_pck) ? kernel_fuse_type : (const uint32_t)conv_fuse_flag::NONE;
                PRAGMA_OMP_FOR_COLLAPSE(3)
                for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
                    for (int64_t oc_l1 = 0; oc_l1 < oc_g_pck; oc_l1 += ocs) {
                        for (int64_t oh = 0; oh < dst_h; oh += oth) {
                            const __fp16 *input_batch_base_ptr = kernel_input + batch_id * in_b_stride;
                            __fp16 *output_batch_base_ptr      = kernel_output + batch_id * out_b_stride;
                            __fp16 *sum_bg_base_ptr            = sum + batch_id * single_batch_output_size + g * oc_group * dst_h * dst_w;
                            const __fp16 *const bias_ptr       = (ic_l1 == 0) ? (bias_g_base + oc_l1) : nullptr;
                            const int64_t ic_remain            = ppl::kernel::arm_server::min(ics, ic_g_pck - ic_l1);
                            const int64_t oc_remains           = ppl::kernel::arm_server::min(ocs, oc_g_pck - oc_l1);
                            const ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t *const fp16_conv_gd_kernel =
                                (oc_remains > 8) ? ppl_arm_server_kernel_fp16_conv_gen_direct_kernels_oc16 : ppl_arm_server_kernel_fp16_conv_gen_direct_kernels_oc8;
#endif
                            const int64_t ih           = -padH + oh * strd_h;
                            int64_t flt_h_start         = DIV_CEIL(std::max((int64_t)0, -ih), dltn_h);
                            int64_t flt_h_end           = std::min(flt_h, DIV_CEIL((src_h - ih), dltn_h));
                            flt_h_end                   = std::max(flt_h_end, flt_h_start);
                            const int64_t flt_h_skipped = flt_h - (flt_h_end - flt_h_start);

                            if (0 < ow_inner_start) {
                                int64_t prv_ow         = 0;
                                int64_t ow             = 0;
                                int64_t prv_flt_w_start = -1;
                                int64_t prv_flt_w_end   = -1;
                                for (; ow < ow_inner_start + 1; ow++) {
                                    const int64_t iw               = -padW + ow * strd_w;
                                    int64_t flt_w_start             = DIV_CEIL(std::max((int64_t)0, -iw), dltn_w);
                                    int64_t flt_w_end               = std::min(flt_w, DIV_CEIL((src_w - iw), dltn_w));
                                    flt_w_end                       = std::max(flt_w_end, flt_w_start);
                                    if (prv_flt_w_start != flt_w_start || prv_flt_w_end != flt_w_end || ow - prv_ow == 10 || ow == ow_inner_start) {
                                        const int64_t prv_flt_w_skipped = flt_w - (prv_flt_w_end - prv_flt_w_start);
                                        if (prv_flt_w_skipped < flt_w && ow > prv_ow) {
                                            const int64_t iw_iter = -padW + prv_ow * strd_w + prv_flt_w_start * dltn_w;
                                            fp16_conv_gd_kernel[ow - prv_ow](
                                                input_batch_base_ptr + ic_l1 * src_h * src_w + (ih + flt_h_start * dltn_h) * src_w * CBLK() + iw_iter * CBLK(),
                                                cvt_filter_g_base + oc_l1 * ic_g_pck * flt_h * flt_w + ic_l1 * flt_h * flt_w * ocs + flt_h_start * flt_w * CBLK() * ocs + prv_flt_w_start * CBLK() * ocs, //TODO
                                                bias_ptr,
                                                output_batch_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                sum_bg_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                ic_remain,
                                                flt_h_end - flt_h_start,
                                                prv_flt_w_end - prv_flt_w_start,
                                                prv_flt_w_skipped * icv_x_ocs_bytes,
                                                flt_h_skipped * flt_w_x_icv_x_ocs_bytes,
                                                dst_h_x_dst_w_x_ocv_bytes,
                                                src_h_x_src_w_x_icv_bytes,
                                                dltn_h_x_src_w_x_icv_bytes,
                                                dltn_w_x_icv_bytes,
                                                strd_w_x_icv_bytes,
                                                fuse_flag);
                                        }
                                        prv_ow         = ow;
                                        prv_flt_w_start = flt_w_start;
                                        prv_flt_w_end   = flt_w_end;
                                    }
                                }
                            }
                            for (int64_t ow = ow_inner_start; ow < ow_inner_end; ow += otw) {
                                const int64_t ow_len = std::min(otw, ow_inner_end - ow);
                                const int64_t iw     = -padW + ow * strd_w;

                                fp16_conv_gd_kernel[ow_len](
                                    input_batch_base_ptr + ic_l1 * src_h * src_w + (ih + flt_h_start * dltn_h) * src_w * CBLK() + iw * CBLK(),
                                    cvt_filter_g_base + oc_l1 * ic_g_pck * flt_h * flt_w + ic_l1 * flt_h * flt_w * ocs + flt_h_start * flt_w * CBLK() * ocs, //TODO
                                    bias_ptr,
                                    output_batch_base_ptr + oc_l1 * dst_w * dst_h + oh * dst_w * CBLK() + ow * CBLK(),
                                    sum_bg_base_ptr + oc_l1 * dst_w * dst_h + oh * dst_w * CBLK() + ow * CBLK(),
                                    ic_remain,
                                    flt_h_end - flt_h_start,
                                    flt_w,
                                    0,
                                    flt_h_skipped * flt_w_x_icv_x_ocs_bytes,
                                    dst_h_x_dst_w_x_ocv_bytes,
                                    src_h_x_src_w_x_icv_bytes,
                                    dltn_h_x_src_w_x_icv_bytes,
                                    dltn_w_x_icv_bytes,
                                    strd_w_x_icv_bytes,
                                    fuse_flag);
                            }
                            if (ow_inner_end < dst_w) {
                                int64_t prv_ow         = ow_inner_end;
                                int64_t ow             = ow_inner_end;
                                int64_t prv_flt_w_start = -1;
                                int64_t prv_flt_w_end   = -1;
                                for (; ow < dst_w + 1; ow++) {
                                    const int64_t iw   = -padW + ow * strd_w;
                                    int64_t flt_w_start = DIV_CEIL(std::max((int64_t)0, -iw), dltn_w);
                                    int64_t flt_w_end   = std::min(flt_w, DIV_CEIL((src_w - iw), dltn_w));
                                    flt_w_end           = std::max(flt_w_end, flt_w_start);
                                    if (prv_flt_w_start != flt_w_start || prv_flt_w_end != flt_w_end || ow - prv_ow == 10 || ow == dst_w) {
                                        const int64_t prv_flt_w_skipped = flt_w - (prv_flt_w_end - prv_flt_w_start);
                                        if (prv_flt_w_skipped < flt_w && ow > prv_ow) {
                                            const int64_t iw_iter = -padW + prv_ow * strd_w + prv_flt_w_start * dltn_w;
                                            fp16_conv_gd_kernel[ow - prv_ow](
                                                input_batch_base_ptr + ic_l1 * src_h * src_w + (ih + flt_h_start * dltn_h) * src_w * CBLK() + iw_iter * CBLK(),
                                                cvt_filter_g_base + oc_l1 * ic_g_pck * flt_h * flt_w + ic_l1 * flt_h * flt_w * ocs + flt_h_start * flt_w * CBLK() * ocs + prv_flt_w_start * CBLK() * ocs, //TODO
                                                bias_ptr,
                                                output_batch_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                sum_bg_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                ic_remain,
                                                flt_h_end - flt_h_start,
                                                prv_flt_w_end - prv_flt_w_start,
                                                prv_flt_w_skipped * icv_x_ocs_bytes,
                                                flt_h_skipped * flt_w_x_icv_x_ocs_bytes,
                                                dst_h_x_dst_w_x_ocv_bytes,
                                                src_h_x_src_w_x_icv_bytes,
                                                dltn_h_x_src_w_x_icv_bytes,
                                                dltn_w_x_icv_bytes,
                                                strd_w_x_icv_bytes,
                                                fuse_flag);
                                        }
                                        prv_ow         = ow;
                                        prv_flt_w_start = flt_w_start;
                                        prv_flt_w_end   = flt_w_end;
                                    }
                                }
                            }
                        } // close loop over oh
                    } // close loop over ic l1 section
                } // close loop over oc l1 section
            } // close loop over batch

            if (use_out_gbuf) {
                for (int64_t b = 0; b < num_batch; b++) {
                    conv2d_n8cx_store_group_fp16(
                        output_gbuf + b * out_b_stride,
                        output + b * single_batch_output_size,
                        sum + b * single_batch_output_size,
                        dst_h * dst_w,
                        oc_group,
                        g,
                        0,
                        cp.fuse_flag);
                }
                PRAGMA_OMP_BARRIER()
            }
        } // close loop over group
    }
    return ppl::common::RC_SUCCESS;
}

bool conv2d_n8cx_direct_fp16_offline_manager::is_supported()
{
    return true;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::fast_init_schedule_param()
{
    sched_param_.oh_blk = 1;
    sched_param_.ow_blk = 10;
    sched_param_.oc_blk = 16;
    sched_param_.ic_blk = 128;
    if (sched_param_.oc_blk != 16 && sched_param_.oc_blk != 8) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (sched_param_.ic_blk != 128) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

static inline int64_t ppl_arm_server_kernel_fp16_conv_direct_n8cx_get_converted_filter_size(
    const int64_t num_group,
    const int64_t in_c,
    const int64_t out_c,
    const int64_t ker_h,
    const int64_t ker_w)
{
    const int64_t ic_group  = in_c / num_group;
    const int64_t oc_group  = out_c / num_group;
    const int64_t ic_g_pck  = CEIL8(ic_group);
    const int64_t oc_g_pck2 = CEIL16(oc_group);
    return CEIL128(num_group * oc_g_pck2 * ic_g_pck * ker_h * ker_w * sizeof(__fp16)) + 128;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::pick_best_schedule_param(const ppl::nn::TensorShape &src_shape, double &run_time, bool tune_blocksize)
{
    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    if (src_shape.GetDimCount() < 4) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int64_t num_batch = src_shape.GetDim(0);
    const int64_t src_h     = src_shape.GetDim(2);
    const int64_t src_w     = src_shape.GetDim(3);
    const int64_t dst_h     = ((src_h + 2 * param_.pad_h - param_.dilation_h * (param_.kernel_h - 1) - 1) / param_.stride_h + 1);
    const int64_t dst_w     = ((src_w + 2 * param_.pad_w - param_.dilation_w * (param_.kernel_w - 1) - 1) / param_.stride_w + 1);
    ppl::nn::TensorShape dst_shape;
    dst_shape.Reshape({num_batch, num_output, dst_h, dst_w});

    uint64_t cvt_filter_size = ppl_arm_server_kernel_fp16_conv_direct_n8cx_get_converted_filter_size(
        param_.group, channels, num_output, kernel_h, kernel_w);
    uint64_t cvt_bias_size = CEIL8(num_output) * sizeof(__fp16);
    uint64_t src_size      = num_batch * CEIL8(channels) * src_h * src_w * sizeof(__fp16);
    uint64_t dst_size      = num_batch * CEIL8(num_output) * dst_h * dst_w * sizeof(__fp16);
    __fp16 *cvt_filter     = (__fp16 *)allocator_->Alloc(cvt_filter_size);
    __fp16 *cvt_bias       = (__fp16 *)allocator_->Alloc(cvt_bias_size);
    __fp16 *src            = (__fp16 *)allocator_->Alloc(src_size);
    __fp16 *dst            = (__fp16 *)allocator_->Alloc(dst_size);

    for (uint64_t idx = 0; idx < cvt_filter_size / sizeof(__fp16); idx++) {
        cvt_filter[idx] = float(rand()) / float((RAND_MAX)) - 0.5;
    }
    for (uint64_t idx = 0; idx < cvt_bias_size / sizeof(__fp16); idx++) {
        cvt_bias[idx] = float(rand()) / float((RAND_MAX)) - 0.5;
    }
    for (uint64_t idx = 0; idx < src_size / sizeof(__fp16); idx++) {
        src[idx] = float(rand()) / float((RAND_MAX)) - 0.5;
    }
    for (uint64_t idx = 0; idx < dst_size / sizeof(__fp16); idx++) {
        dst[idx] = float(rand()) / float((RAND_MAX)) - 0.5;
    }

    std::vector<int64_t> candidate_oc_blk_list = {16};
    std::vector<int64_t> candidate_ic_blk_list = {128};

    if (tune_blocksize) {
        candidate_oc_blk_list = {16};
        candidate_ic_blk_list = {32, /*48,*/ 64, /*72, 96, 112,*/ 128, /*160,*/ 192, /*224,*/ 256};
    }

    int64_t best_oc_blk   = 16;
    int64_t best_ic_blk   = 128;
    int64_t best_run_time = std::numeric_limits<int64_t>::max();

    const int num_warmup_iter    = 1;
    const int num_benchmark_iter = 5;
    for (auto oc_blk : candidate_oc_blk_list) {
        for (auto ic_blk : candidate_ic_blk_list) {
            sched_param_.oc_blk = oc_blk;
            sched_param_.ic_blk = ic_blk;
            sched_param_.oh_blk = 1;
            sched_param_.ow_blk = 10;

            auto conv_exe = gen_executor();
            conv_exe->set_cvt_filter(cvt_filter);
            conv_exe->set_cvt_bias(cvt_bias);
            conv_exe->set_src(src);
            conv_exe->set_src_shape(&src_shape);
            conv_exe->set_dst(dst);
            conv_exe->set_dst_shape(&dst_shape);
            conv_exe->prepare();
            uint64_t tmp_buf_size = conv_exe->cal_temp_buffer_size();
            __fp16 *tmp_buffer    = (__fp16 *)allocator_->Alloc(tmp_buf_size);
            conv_exe->set_temp_buffer(tmp_buffer);

            for (int i = 0; i < num_warmup_iter; i++) {
                conv_exe->execute();
            }

            auto begin_ts = std::chrono::system_clock::now();
            for (int i = 0; i < num_benchmark_iter; i++) {
                conv_exe->execute();
            }
            auto end_ts = std::chrono::system_clock::now();

            int64_t elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - begin_ts).count();
            if (elapsed_time < best_run_time) {
                best_oc_blk   = oc_blk;
                best_ic_blk   = ic_blk;
                best_run_time = elapsed_time;
            }

            allocator_->Free(tmp_buffer);
            delete conv_exe;

            if (ic_blk >= channels / param_.group) break;
        }
        if (oc_blk >= num_output / param_.group) break;
    }

    cvt_filter_ = nullptr;
    cvt_bias_   = nullptr;
    allocator_->Free(cvt_filter);
    allocator_->Free(cvt_bias);
    allocator_->Free(src);
    allocator_->Free(dst);

    sched_param_.oc_blk = best_oc_blk;
    sched_param_.ic_blk = best_ic_blk;
    sched_param_.oh_blk = 1;
    sched_param_.ow_blk = 10;
    LOG(INFO) << "choose sp param oc: " << sched_param_.oc_blk;
    LOG(INFO) << "choose sp param ic: " << sched_param_.ic_blk;
    LOG(INFO) << "best run time: " << best_run_time / num_benchmark_iter / 1000 << " ms";
    run_time = (double)best_run_time / (double)num_benchmark_iter;
    return ppl::common::RC_SUCCESS;
}

#define ICBLK() 8
// NOTE: (oc, ic, kh, kw) -> (oc/16, ic/8, kh, kw, 8ic, 16oc)
static void ppl_arm_server_kernel_fp16_conv_direct_n8cx_convert_filter(
    const __fp16 *filter,
    __fp16 *converted_filter,
    const int64_t num_group,
    const int64_t in_c,
    const int64_t out_c,
    const int64_t flt_h,
    const int64_t flt_w)
{
    const int64_t ocs       = 16;
    const int64_t ic_group  = in_c / num_group;
    const int64_t oc_group  = out_c / num_group;
    const int64_t ic_g_pck  = CEIL8(ic_group);
    const int64_t oc_g_pck2 = CEIL16(oc_group);

    for (int64_t g = 0; g < num_group; g++) {
        const __fp16 *filter_g_base = filter + g * oc_group * ic_group * flt_h * flt_w;
        __fp16 *cvt_filter_g_base   = converted_filter + g * oc_g_pck2 * ic_g_pck * flt_h * flt_w;

        for (int64_t oc = 0; oc < oc_group; oc++) {
            for (int64_t ic = 0; ic < ic_group; ic++) {
                for (int64_t kh = 0; kh < flt_h; kh++) {
                    for (int64_t kw = 0; kw < flt_w; kw++) {
                        const int64_t cvt_index = (oc / ocs) * DIV_CEIL(ic_group, ICBLK()) * flt_h * flt_w * ICBLK() * ocs +
                                                  (ic / ICBLK()) * flt_h * flt_w * ICBLK() * ocs +
                                                  kh * flt_w * ICBLK() * ocs +
                                                  kw * ICBLK() * ocs +
                                                  (ic % ICBLK()) * ocs +
                                                  oc % ocs;
                        cvt_filter_g_base[cvt_index] = filter_g_base[oc * ic_group * flt_h * flt_w + ic * flt_h * flt_w + kh * flt_w + kw];
                    }
                }
            }
            for (int64_t ic = ic_group; ic < ic_g_pck; ic++) {
                for (int64_t kh = 0; kh < flt_h; kh++) {
                    for (int64_t kw = 0; kw < flt_w; kw++) {
                        const int64_t cvt_index = (oc / ocs) * DIV_CEIL(ic_group, ICBLK()) * flt_h * flt_w * ICBLK() * ocs +
                                                  (ic / ICBLK()) * flt_h * flt_w * ICBLK() * ocs +
                                                  kh * flt_w * ICBLK() * ocs +
                                                  kw * ICBLK() * ocs +
                                                  (ic % ICBLK()) * ocs +
                                                  oc % ocs;
                        cvt_filter_g_base[cvt_index] = 0.0f;
                    }
                }
            }
        }

        for (int64_t oc = oc_group; oc < oc_g_pck2; oc++) {
            for (int64_t ic = 0; ic < ic_g_pck; ic++) {
                for (int64_t kh = 0; kh < flt_h; kh++) {
                    for (int64_t kw = 0; kw < flt_w; kw++) {
                        const int64_t cvt_index = (oc / ocs) * DIV_CEIL(ic_group, ICBLK()) * flt_h * flt_w * ICBLK() * ocs +
                                                  (ic / ICBLK()) * flt_h * flt_w * ICBLK() * ocs +
                                                  kh * flt_w * ICBLK() * ocs +
                                                  kw * ICBLK() * ocs +
                                                  (ic % ICBLK()) * ocs +
                                                  oc % ocs;
                        cvt_filter_g_base[cvt_index] = 0.0f;
                    }
                }
            }
        }
    }
}
#undef ICBLK

// should be called after init_schedule_param
ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::gen_cvt_weights(const void *filter, const void *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_group  = param_.group;
    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    cvt_bias_size_               = CEIL8(num_output) * sizeof(__fp16);
    cvt_bias_                    = allocator_->Alloc(cvt_bias_size_);
    int64_t padding_offset_bytes = num_output * sizeof(__fp16);
    int64_t padding_bytes        = (CEIL8(num_output) - num_output) * sizeof(__fp16);
    memcpy(cvt_bias_, bias, padding_offset_bytes);
    memset((uint8_t *)cvt_bias_ + padding_offset_bytes, 0, padding_bytes);

    if (sched_param_.oc_blk == 16) {
        cvt_filter_size_  = ppl_arm_server_kernel_fp16_conv_direct_n8cx_get_converted_filter_size(
            num_group, channels, num_output, kernel_h, kernel_w);
        cvt_filter_ = (__fp16 *)allocator_->Alloc(cvt_filter_size_);
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_convert_filter(
            (const __fp16 *)filter,
            (__fp16 *)cvt_filter_,
            num_group,
            channels,
            num_output,
            kernel_h,
            kernel_w);
        return ppl::common::RC_SUCCESS;
    }
    return ppl::common::RC_INVALID_VALUE;
}

conv2d_runtime_executor *conv2d_n8cx_direct_fp16_offline_manager::gen_executor()
{
    return new conv2d_n8cx_direct_fp16_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}

}}}; // namespace ppl::kernel::arm_server

#endif
