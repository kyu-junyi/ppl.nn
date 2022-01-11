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

#include "ppl/kernel/arm_server/conv2d/neon/fp32/direct/conv2d_n4cx_direct_fp32.h"

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
#include "ppl/kernel/arm_server/conv2d/neon/fp32/utils/conv2d_utils_fp32.h"

namespace ppl { namespace kernel { namespace arm_server {

#define CBLK() 4

#define ICBLK() CBLK()
#define OCBLK() CBLK()

// #define CEIL4(val) (((val) + 3) & (~3))
// #define CEIL128(val) (((val) + 127) & (~127))
// #define DIV_CEIL(aval, bval) ( ((aval) + (bval) - 1) / (bval) )

#define OUT_TILE_W() 10
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 9
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 8
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 7
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 6
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 5
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 4
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 3
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 2
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 1
    #include "conv_direct_kernel.inc"
#undef OUT_TILE_W

#define OUT_TILE_W() 10
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 9
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 8
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 7
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 6
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 5
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 4
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 3
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 2
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 1
    #include "conv_direct_kernel_f1.inc"
#undef OUT_TILE_W

typedef void(*ppl_arm_server_kernel_fp32_conv_direct_kernel_t)(
    const float *input_base,
    const float *filter_base,
    const float *bias_base,
    float *output_base,
    float *sum_base,
    int64_t inC_block_ceil4,
    const int64_t fltH,
    const int64_t fltW,
    const int64_t fltDiffW_x_icV_x_ocS_bytes,
    const int64_t fltDiffH_x_fltW_x_icV_x_ocS_bytes,
    const int64_t outH_x_outW_x_ocV_bytes,
    const int64_t inH_x_inW_x_icV_bytes,
    const int64_t dltnH_x_inW_x_icV_bytes,
    const int64_t dltnW_x_icV_bytes,
    const int64_t strdW_x_icV_bytes,
    const uint32_t fuse_flag);

#define OW_CASE() 10
static ppl_arm_server_kernel_fp32_conv_direct_kernel_t ppl_arm_server_kernel_fp32_conv_direct_oc8_kernel_func_table[OW_CASE()+1] =
{
    nullptr,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w2_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w3_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w4_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w5_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w6_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w7_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w8_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w9_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w10_func,
};
#undef OW_CASE

#define OW_CASE() 10
static ppl_arm_server_kernel_fp32_conv_direct_kernel_t ppl_arm_server_kernel_fp32_conv_direct_oc4_kernel_func_table[OW_CASE()+1] =
{
    nullptr,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w2_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w3_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w4_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w5_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w6_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w7_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w8_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w9_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w10_func,
};
#undef OW_CASE

#define OW_CASE() 10
static ppl_arm_server_kernel_fp32_conv_direct_kernel_t ppl_arm_server_kernel_fp32_conv_direct_oc8_kernel_f1_func_table[OW_CASE()+1] =
{
    nullptr,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w1_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w2_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w3_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w4_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w5_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w6_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w7_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w8_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w9_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc8_h1w10_f1_func,
};
#undef OW_CASE

#define OW_CASE() 10
static ppl_arm_server_kernel_fp32_conv_direct_kernel_t ppl_arm_server_kernel_fp32_conv_direct_oc4_kernel_f1_func_table[OW_CASE()+1] =
{
    nullptr,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w1_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w2_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w3_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w4_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w5_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w6_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w7_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w8_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w9_f1_func,
    ppl_arm_server_kernel_fp32_conv_direct_n4cx_oc4_h1w10_f1_func,
};
#undef OW_CASE

static void ppl_arm_server_kernel_fp32_conv_direct_n4cx_load_input_tile(
    const float *input,
    float *tmp_buffer,
    const int64_t inH,
    const int64_t inW,
    const int64_t ih_start,
    const int64_t ih_end,
    const int64_t iw_start,
    const int64_t iw_end,
    const int64_t ic_start,
    const int64_t ic_end)
{
    const float32x4_t vzeros = vdupq_n_f32(0.0f);

    const int64_t inH_x_inW = inH * inW;
    const int64_t inW_x_cblk = inW * ICBLK();
    const int64_t iTileH_x_iTileW = (ih_end-ih_start) * (iw_end-iw_start);
    const int64_t iTileW_x_cblk = (iw_end-iw_start) * ICBLK();

    const bool in_pad_right = (iw_end > inW);
    const int64_t in_valid_end = std::min(inW, iw_end);

    // if (ic_start % ICBLK() != 0) std::abort();

    for (int64_t ic = ic_start, itc = 0; ic < ic_end; ic += ICBLK(), itc += ICBLK()) {
        for (int64_t ih = ih_start, ith = 0; ih < ih_end; ih++, ith++) {
            float * const buffer_row_base_ptr = tmp_buffer + itc * iTileH_x_iTileW + ith * iTileW_x_cblk;
            if (ih < 0 || ih >= inH) {
// #pragma unroll 4
                for (int64_t iw = iw_start, itw_x_cblk = 0; iw < iw_end; iw++, itw_x_cblk += ICBLK()) {
                    vst1q_f32(buffer_row_base_ptr+itw_x_cblk, vzeros);
                }
            }
            else {
                const float * const input_row_base_ptr = input + ic * inH_x_inW + ih * inW_x_cblk;
                int64_t iw = iw_start;
                int64_t itw_x_cblk = 0;
                for (; iw < 0; iw++, itw_x_cblk += ICBLK()) {
                    vst1q_f32(buffer_row_base_ptr+itw_x_cblk, vzeros);
                }
// #pragma unroll 4
                for (; iw < in_valid_end; iw++, itw_x_cblk += ICBLK()) {
                    vst1q_f32(buffer_row_base_ptr+itw_x_cblk, vld1q_f32(input_row_base_ptr + iw * ICBLK()));
                }
                if (in_pad_right) {
                    for (; iw < iw_end; iw++, itw_x_cblk += ICBLK()) {
                        vst1q_f32(buffer_row_base_ptr+itw_x_cblk, vzeros);
                    }
                }
            }
        }
    }
}

#if 0
static void ppl_arm_server_kernel_fp32_conv_direct_ndarray_load_input_tile(
    const float *input,
    float *tmp_buffer,
    const int64_t inH,
    const int64_t inW,
    const int64_t ih_start,
    const int64_t ih_end,
    const int64_t iw_start,
    const int64_t iw_end,
    const int64_t ic_start,
    const int64_t ic_end)
{
    const float32x4_t vzeros = vdupq_n_f32(0.0f);

    const int64_t inH_x_inW = inH * inW;
    const int64_t inW_x_cblk = inW * ICBLK();
    const int64_t iTileH_x_iTileW = (ih_end-ih_start) * (iw_end-iw_start);
    const int64_t iTileW_x_cblk = (iw_end-iw_start) * ICBLK();

    const bool in_pad_right = (iw_end > inW);
    const int64_t in_valid_end = std::min(inW, iw_end);

    // if (ic_start % ICBLK() != 0) std::abort();

    for (int64_t ic = ic_start, itc = 0; ic < ic_end; ic += ICBLK(), itc += ICBLK()) {
        const int64_t ic_blk_valid = std::min((int64_t)ICBLK(), ic_end-ic);
            // std::cout << ih_start << " H " << ih_end << std::endl;
            // std::cout << iw_start << " W " << iw_end << std::endl;
        for (int64_t ih = ih_start, ith = 0; ih < ih_end; ih++, ith++) {
            float * const buffer_row_base_ptr = tmp_buffer + itc * iTileH_x_iTileW + ith * iTileW_x_cblk;
            if (ih < 0 || ih >= inH) {
// #pragma unroll 4
                for (int64_t iw = iw_start, itw_x_cblk = 0; iw < iw_end; iw++, itw_x_cblk += ICBLK()) {
                    vst1q_f32(buffer_row_base_ptr+itw_x_cblk, vzeros);
                }
            }
            else {
                const float * const input_c_h_base = input + ic * inH_x_inW + ih * inW;
                int64_t iw = iw_start;
                int64_t itw_x_cblk = 0;
                for (; iw < 0; iw++, itw_x_cblk += ICBLK()) {
                    vst1q_f32(buffer_row_base_ptr+itw_x_cblk, vzeros);
                }
// #pragma unroll 4
                for (; iw < in_valid_end; iw++, itw_x_cblk += ICBLK()) {
                    const float * const input_w_base = input_c_h_base + iw;
                    for (int64_t lane = 0; lane < ic_blk_valid; lane++) {
                        buffer_row_base_ptr[itw_x_cblk+lane] = input_w_base[lane * inH_x_inW];
                    }
                    for (int64_t lane = ic_blk_valid; lane < ICBLK(); lane++) {
                        buffer_row_base_ptr[itw_x_cblk+lane] = 0.0f;
                    }
                }
                if (in_pad_right) {
                    for (; iw < iw_end; iw++, itw_x_cblk += ICBLK()) {
                        vst1q_f32(buffer_row_base_ptr+itw_x_cblk, vzeros);
                    }
                }
            }
        }
    }
    // for (int64_t idx = 0; idx < 36; idx++) std::cout << tmp_buffer[idx] << " , ";
    // std::cout << std::endl;
}
#endif

uint64_t conv2d_n4cx_direct_fp32_runtime_executor::get_padding_buffer_size() {
    const int64_t num_threads = PPL_OMP_MAX_THREADS();
    const conv2d_param &cp = *conv_param_;
    const conv2d_n4cx_direct_fp32_schedule_param &sp = sched_param_;
    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);
    // if (cp.pad_h == 0 && cp.pad_w == 0) return 0;

    if (cp.kernel_h == 1 && cp.kernel_w == 1) {
        return (dst_h * dst_w * sp.ic_blk * sizeof(float) + 160 + 128) * num_threads;
    }

    // Local padding buffer
    const int64_t otH = sp.oh_blk;
    const int64_t otW = sp.ow_blk;
    const int64_t icS = sp.ic_blk;
    const int64_t ocS = sp.oc_blk; (void)ocS;

    const int64_t iTileH = (otH-1)*cp.stride_h + cp.dilation_h*(cp.kernel_h-1) + 1;
    const int64_t iTileW = (otW-1)*cp.stride_w + cp.dilation_w*(cp.kernel_w-1) + 1;
    const int64_t in_tile_num = icS * iTileH * iTileW;
    const size_t in_tile_size = in_tile_num * sizeof(float);
    // std::cerr << "Buffer size: " << CEIL128(in_tile_size) << std::endl;
    size_t local_padding_buffer_size = CEIL128(in_tile_size) + 128;

    size_t global_padding_buffer_size = CEIL128(icS * (src_h + 2 * cp.pad_h) * (src_w + 2 * cp.pad_w) * sizeof(float)) + 128;

    if (cp.pad_w > 0 && cp.pad_w <= (otW + (otW+1)/2)) {  // TODO: Check against the cache size after memory-footprint analysis
        return global_padding_buffer_size * num_threads;
    }
    
    return local_padding_buffer_size * num_threads;
}

uint64_t conv2d_n4cx_direct_fp32_runtime_executor::cal_temp_buffer_size() {
    const conv2d_param &cp = *conv_param_;
    const int64_t ic_g_pck = CEIL4(cp.channels / cp.group);
    const int64_t oc_g_pck = CEIL4(cp.num_output / cp.group);

    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);
    const int64_t num_batch = src_shape_->GetDim(0);

    uint64_t input_gbuf_size = num_batch * ic_g_pck * src_h * src_w * sizeof(float);
    uint64_t output_gbuf_size = num_batch * oc_g_pck * dst_h * dst_w * sizeof(float);
    uint64_t packing_buffer_size = PPL_OMP_MAX_THREADS() * dst_h * dst_w * sched_param_.ic_blk * sizeof(float);

    return input_gbuf_size + output_gbuf_size + packing_buffer_size;
}

void conv2d_n4cx_direct_fp32_runtime_executor::adjust_schedule_param() {
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);
    sched_param_.padding_buffer_size_per_thread = dst_h * dst_w * sched_param_.ic_blk * sizeof(float);
    return;
}

ppl::common::RetCode conv2d_n4cx_direct_fp32_runtime_executor::prepare() {
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_direct_fp32_runtime_executor::execute() {
    const conv2d_param &cp = *conv_param_;
    const conv2d_n4cx_direct_fp32_schedule_param &sp = sched_param_;

    const float *input = (const float *)src_;
    const float *cvt_filter = (const float *)cvt_filter_;
    const float *bias = (const float *)cvt_bias_;
    float *output = (float *)dst_;
    float *sum = (float *)sum_;
    float *tmp_buffer = (float*)temp_buffer_;
    const int64_t inH = src_shape_->GetDim(2);
    const int64_t inW = src_shape_->GetDim(3);
    const int64_t inC = src_shape_->GetDim(1);
    const int64_t outC = cp.num_output;
    const int64_t outH = dst_shape_->GetDim(2);
    const int64_t outW = dst_shape_->GetDim(3);
    const int64_t fltH = cp.kernel_h;
    const int64_t fltW = cp.kernel_w;
    const int64_t padH = cp.pad_h;
    const int64_t padW = cp.pad_w;
    const int64_t strdH = cp.stride_h;
    const int64_t strdW = cp.stride_w;
    const int64_t dltnH = cp.dilation_h;
    const int64_t dltnW = cp.dilation_w;
    const int64_t single_core_tmp_buffer_offset = sp.padding_buffer_size_per_thread / sizeof(float);
    const int64_t num_batch = src_shape_->GetDim(0);

PRAGMA_OMP_PARALLEL()
{
    const int64_t thread_id = PPL_OMP_THREAD_ID();
    // float *input_aux_buffer = tmp_buffer + thread_id * single_core_tmp_buffer_offset;
    // input_aux_buffer = (float*)(((uint64_t)input_aux_buffer + 63) / 64 * 64);
    
    const int64_t inC_pck = CEIL4(inC);
    const int64_t outC_pck = CEIL4(outC);

    const int64_t ic_group = inC / cp.group;
    const int64_t oc_group = outC / cp.group;
    const int64_t ic_g_pck = CEIL4(ic_group);
    const int64_t oc_g_pck = CEIL4(oc_group);
    
    const int64_t otH = sp.oh_blk;
    const int64_t otW = sp.ow_blk;
    
    const int64_t ocS = sp.oc_blk;
    const int64_t icS = sp.ic_blk;

    const int64_t icV_bytes = 16; // ICBLK() * sizeof(float);
    const int64_t ocV_bytes = 16; // OCBLK() * sizeof(float);
    
    const int64_t inH_x_inW_x_icV_bytes = inH * inW * icV_bytes;
    const int64_t outH_x_outW_x_ocV_bytes = outH * outW * ocV_bytes;
    const int64_t dltnH_x_icV_bytes = dltnH * icV_bytes;
    const int64_t dltnW_x_icV_bytes = dltnW * icV_bytes;
    const int64_t strdW_x_icV_bytes = strdW * icV_bytes;
    const int64_t dltnH_x_inW_x_icV_bytes = inW * dltnH_x_icV_bytes;
    const int64_t icV_x_ocS_bytes = ocS * icV_bytes;
    const int64_t fltW_x_icV_x_ocS_bytes = fltW * ocS * icV_bytes;

    const int64_t single_batch_input_size = inC_pck * inH * inW;
    const int64_t single_batch_output_size = outC_pck * outH * outW;

    const bool use_in_gbuf  = (cp.group > 1 && ic_g_pck != ic_group);
    const bool use_out_gbuf = (cp.group > 1 && oc_g_pck != oc_group);
    const int64_t input_group_buffer_offset = num_batch * ic_g_pck * inH * inW;
    const int64_t output_group_buffer_offset = num_batch * oc_g_pck * outH * outW;
    float *input_gbuf = tmp_buffer;
    float *output_gbuf = input_gbuf + input_group_buffer_offset;
    float *input_aux_buffer = output_gbuf + output_group_buffer_offset + thread_id * single_core_tmp_buffer_offset;
    // input_aux_buffer = (float*)(((uint64_t)input_aux_buffer + 63) / 64 * 64);

    // const int64_t ocL1S = 64; (void)ocL1S;
    // const int64_t icL1S = 128; (void)icL1S;

    const bool pre_padding = (outW <= (otW + (otW+1)/2)) && (padW > 0);
    const bool use_pack = (fltH == 1 && fltW == 1 && (strdH > 1 || strdW > 1));
    const bool use_f1s1 = (fltH == 1 && fltW == 1 && strdH == 1 && strdW == 1);
    int64_t packed_batch = -1;
    int64_t packed_ic = -1;

    int64_t ow_inner_start = std::max((int64_t)0, DIV_CEIL((padW - 0 * dltnW), strdW));  // inclusive
    int64_t ow_inner_end   = std::min((int64_t)outW, DIV_CEIL((inW + padW - (fltW-1) * dltnW), strdW));  // exclusive
    ow_inner_start = std::min(ow_inner_start, outW);
    ow_inner_end   = std::max(ow_inner_end, ow_inner_start);

    uint32_t kernel_fuse_type = cp.fuse_flag;
    if (use_out_gbuf && (cp.fuse_flag & conv_fuse_flag::SUM)) {
        kernel_fuse_type = conv_fuse_flag::NONE;
    }

    for (int64_t g = 0; g < cp.group; g++) {
        int64_t in_b_stride = single_batch_input_size;
        int64_t out_b_stride = single_batch_output_size;

        const float *cvt_filter_g_base = cvt_filter + g * CEIL(oc_group, ocS) * ic_g_pck * fltH * fltW;
        const float *bias_g_base = bias + g * oc_group;

        const float * kernel_input = input + g * ic_group * inH * inW;
        float * kernel_output = output + g * oc_group * outH * outW;
        if (use_in_gbuf) {
            in_b_stride = ic_g_pck * inH * inW;
            kernel_input = input_gbuf;
            for (int64_t b = 0; b < num_batch; b++) {
                conv2d_n4cx_load_group_fp32(
                    input + b * single_batch_input_size,
                    input_gbuf + b * in_b_stride,
                    inH * inW,
                    ic_group,
                    g,
                    0);
            }
            PRAGMA_OMP_BARRIER()
        }
        if (use_out_gbuf) {
            out_b_stride = oc_g_pck * outH * outW;
            kernel_output = output_gbuf;
        }

#if not defined PPL_USE_ARM_SERVER_OMP
        for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
            const float * input_batch_base_ptr = kernel_input + batch_id * in_b_stride;
            float * output_batch_base_ptr = kernel_output + batch_id * out_b_stride;
            float * sum_bg_base_ptr = sum + batch_id * single_batch_output_size + g * oc_group * outH * outW;
            for (int64_t ic_l1 = 0; ic_l1 < ic_g_pck; ic_l1 += icS) {
                const int64_t ic_remain = std::min(icS, ic_g_pck - ic_l1);
                const uint32_t fuse_flag = (ic_l1 + icS >= ic_g_pck) ? kernel_fuse_type : conv_fuse_flag::NONE;
                for (int64_t oc_l1 = 0; oc_l1 < oc_g_pck; oc_l1 += ocS) {
                    const float * const bias_ptr = (ic_l1 == 0) ? (bias_g_base + oc_l1) : nullptr;
                    const int64_t oc_remains = std::min(ocS, oc_g_pck - oc_l1);
                    const ppl_arm_server_kernel_fp32_conv_direct_kernel_t * conv_direct_kernel_func_table = 
                        (oc_remains > 4) ? ppl_arm_server_kernel_fp32_conv_direct_oc8_kernel_func_table :
                                           ppl_arm_server_kernel_fp32_conv_direct_oc4_kernel_func_table;
                    if (use_pack || use_f1s1) {
                        conv_direct_kernel_func_table = 
                            (oc_remains > 4) ? ppl_arm_server_kernel_fp32_conv_direct_oc8_kernel_f1_func_table :
                                               ppl_arm_server_kernel_fp32_conv_direct_oc4_kernel_f1_func_table;
                    }
                    for (int64_t oh = 0; oh < outH; oh += otH) {
#else
        for (int64_t ic_l1 = 0; ic_l1 < ic_g_pck; ic_l1 += icS) {
            const uint32_t fuse_flag = (ic_l1 + icS >= ic_g_pck) ? kernel_fuse_type : conv_fuse_flag::NONE;
            PRAGMA_OMP_FOR_COLLAPSE(3)
            for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
                for (int64_t oc_l1 = 0; oc_l1 < oc_g_pck; oc_l1 += ocS) {
                    for (int64_t oh = 0; oh < outH; oh += otH) {
                        const float * input_batch_base_ptr = kernel_input + batch_id * in_b_stride;
                        float * output_batch_base_ptr = kernel_output + batch_id * out_b_stride;
                        float * sum_bg_base_ptr = sum + batch_id * single_batch_output_size + g * oc_group * outH * outW;
                        const float * const bias_ptr = (ic_l1 == 0) ? (bias_g_base + oc_l1) : nullptr;
                        const int64_t ic_remain = std::min(icS, ic_g_pck - ic_l1);
                        const int64_t oc_remains = std::min(ocS, oc_g_pck - oc_l1);
                        const ppl_arm_server_kernel_fp32_conv_direct_kernel_t * conv_direct_kernel_func_table = 
                            (oc_remains > 4) ? ppl_arm_server_kernel_fp32_conv_direct_oc8_kernel_func_table :
                                                ppl_arm_server_kernel_fp32_conv_direct_oc4_kernel_func_table;
                        if (use_pack || use_f1s1) {
                            conv_direct_kernel_func_table = 
                                (oc_remains > 4) ? ppl_arm_server_kernel_fp32_conv_direct_oc8_kernel_f1_func_table :
                                                   ppl_arm_server_kernel_fp32_conv_direct_oc4_kernel_f1_func_table;
                        }
#endif
                        const int64_t ih = -padH + oh * strdH;
                        int64_t fltH_start = DIV_CEIL(std::max((int64_t)0, -ih), dltnH);
                        int64_t fltH_end   = std::min(fltH, DIV_CEIL((inH-ih), dltnH));
                        fltH_end = std::max(fltH_end, fltH_start);
                        const int64_t fltH_skipped = fltH - (fltH_end - fltH_start);
                        if (fltH_skipped == fltH) continue;

                        if (use_pack && (ic_l1 != packed_ic || batch_id != packed_batch)) {
                            for (int ic = 0; ic < ic_remain; ic += ICBLK()) {
                                for (int oh_iter = 0; oh_iter < outH; oh_iter++) {
                                    const int64_t ih = -padH + oh_iter * strdH;
                                    for (int ow = 0; ow < outW; ow++) {
                                        const int64_t iw = -padW + ow * strdW;
                                        vst1q_f32(input_aux_buffer + ic * outH * outW + oh_iter * outW * OCBLK() + ow * OCBLK(),
                                                  vld1q_f32(input_batch_base_ptr + (ic_l1 + ic) * inH * inW + ih * inW * ICBLK() + iw * ICBLK()));
                                    }
                                }
                            }
                            packed_batch = batch_id;
                            packed_ic = ic_l1;
                        }

                        const float * input_h_base = input_batch_base_ptr + ic_l1 * inW * inH + (ih + fltH_start * dltnH) * inW * CBLK();
                        int64_t local_strdW = strdW;
                        int64_t input_cblk_stride_bytes = inH_x_inW_x_icV_bytes;
                        int64_t input_filtered_row_stride_bytes = dltnH_x_inW_x_icV_bytes;
                        if (use_pack) {
                            local_strdW = 1;
                            input_h_base = input_aux_buffer + oh * outW * CBLK();
                            input_cblk_stride_bytes = outH * outW * icV_bytes;
                            input_filtered_row_stride_bytes = outW * icV_bytes;
                        }
                        if (0 < ow_inner_start) {
                            int64_t prv_ow = 0;
                            int64_t ow = 0;
                            int64_t prv_fltW_start = -1;
                            int64_t prv_fltW_end   = -1;
                            for (; ow < ow_inner_start + 1; ow++) {
                                const int64_t iw = -padW + ow * strdW;
                                int64_t fltW_start = DIV_CEIL(std::max((int64_t)0, -iw), dltnW);
                                int64_t fltW_end   = std::min(fltW, DIV_CEIL((inW-iw), dltnW));
                                fltW_end = std::max(fltW_end, fltW_start);
                                const int64_t prv_fltW_skipped = fltW - (prv_fltW_end - prv_fltW_start);
                                if (prv_fltW_start != fltW_start || prv_fltW_end != fltW_end || ow - prv_ow == 10 || ow == ow_inner_start) {
                                    if (prv_fltW_skipped < fltW && ow > prv_ow) {
                                        const int64_t iw_iter = -padW + prv_ow * local_strdW + prv_fltW_start * dltnW;
                                        conv_direct_kernel_func_table[ow-prv_ow](
                                            input_h_base + iw_iter * CBLK(),
                                            cvt_filter_g_base + oc_l1 * ic_g_pck * fltH * fltW + ic_l1 * fltH * fltW * ocS + fltH_start * fltW * CBLK() * ocS + prv_fltW_start * CBLK() * ocS, //TODO
                                            bias_ptr,
                                            output_batch_base_ptr + oc_l1 * outH * outW + oh * outW * CBLK() + prv_ow * CBLK(),
                                            sum_bg_base_ptr + oc_l1 * outH * outW + oh * outW * CBLK() + prv_ow * CBLK(),
                                            ic_remain,
                                            fltH_end - fltH_start,
                                            prv_fltW_end - prv_fltW_start,
                                            prv_fltW_skipped * icV_x_ocS_bytes,
                                            fltH_skipped * fltW_x_icV_x_ocS_bytes,
                                            outH_x_outW_x_ocV_bytes,
                                            input_cblk_stride_bytes,
                                            input_filtered_row_stride_bytes,
                                            dltnW_x_icV_bytes,
                                            use_pack ? icV_bytes : strdW_x_icV_bytes,
                                            fuse_flag);
                                    }
                                    prv_ow = ow;
                                    prv_fltW_start = fltW_start;
                                    prv_fltW_end   = fltW_end;
                                }
                            }
                        }
                        for (int64_t ow = ow_inner_start; ow < ow_inner_end; ow += otW) {
                            const int64_t ow_len = std::min(otW, ow_inner_end - ow);
                            const int64_t iw = -padW + ow * local_strdW;

                            conv_direct_kernel_func_table[ow_len](
                                input_h_base + iw * CBLK(),
                                cvt_filter_g_base + oc_l1 * ic_g_pck * fltH * fltW + ic_l1 * fltH * fltW * ocS + fltH_start * fltW * CBLK() * ocS, //TODO
                                bias_ptr,
                                output_batch_base_ptr + oc_l1 * outW * outH + oh * outW * CBLK() + ow * CBLK(),
                                sum_bg_base_ptr + oc_l1 * outW * outH + oh * outW * CBLK() + ow * CBLK(),
                                ic_remain,
                                fltH_end - fltH_start,
                                fltW,
                                0,
                                fltH_skipped * fltW_x_icV_x_ocS_bytes,
                                outH_x_outW_x_ocV_bytes,
                                input_cblk_stride_bytes,
                                input_filtered_row_stride_bytes,
                                dltnW_x_icV_bytes,
                                use_pack ? icV_bytes : strdW_x_icV_bytes,
                                fuse_flag);
                        }
                        if (ow_inner_end < outW) {
                            int64_t prv_ow = ow_inner_end;
                            int64_t ow = ow_inner_end;
                            int64_t prv_fltW_start = -1;
                            int64_t prv_fltW_end   = -1;
                            for (; ow < outW + 1; ow++) {
                                const int64_t iw = -padW + ow * strdW;
                                int64_t fltW_start = DIV_CEIL(std::max((int64_t)0, -iw), dltnW);
                                int64_t fltW_end   = std::min(fltW, DIV_CEIL((inW-iw), dltnW));
                                fltW_end = std::max(fltW_end, fltW_start);
                                if (prv_fltW_start != fltW_start || prv_fltW_end != fltW_end || ow - prv_ow == 10 || ow == outW) {
                                    const int64_t prv_fltW_skipped = fltW - (prv_fltW_end - prv_fltW_start);
                                    if (prv_fltW_skipped < fltW && ow > prv_ow) {
                                        const int64_t iw_iter = -padW + prv_ow * local_strdW + prv_fltW_start * dltnW;
                                        conv_direct_kernel_func_table[ow-prv_ow](
                                            input_h_base + iw_iter * CBLK(),
                                            cvt_filter_g_base + oc_l1 * ic_g_pck * fltH * fltW + ic_l1 * fltH * fltW * ocS + fltH_start * fltW * CBLK() * ocS + prv_fltW_start * CBLK() * ocS, //TODO
                                            bias_ptr,
                                            output_batch_base_ptr + oc_l1 * outH * outW + oh * outW * CBLK() + prv_ow * CBLK(),
                                            sum_bg_base_ptr + oc_l1 * outH * outW + oh * outW * CBLK() + prv_ow * CBLK(),
                                            ic_remain,
                                            fltH_end - fltH_start,
                                            prv_fltW_end - prv_fltW_start,
                                            prv_fltW_skipped * icV_x_ocS_bytes,
                                            fltH_skipped * fltW_x_icV_x_ocS_bytes,
                                            outH_x_outW_x_ocV_bytes,
                                            input_cblk_stride_bytes,
                                            input_filtered_row_stride_bytes,
                                            dltnW_x_icV_bytes,
                                            use_pack ? icV_bytes : strdW_x_icV_bytes,
                                            fuse_flag);
                                    }
                                    prv_ow = ow;
                                    prv_fltW_start = fltW_start;
                                    prv_fltW_end   = fltW_end;
                                }
                            }
                        }
                    }  // close loop over oh
                }  // close loop over ic l1 section
            }  // close loop over oc l1 section
        }  // close loop over batch

        if (use_out_gbuf) {
            for (int64_t b = 0; b < num_batch; b++) {
                conv2d_n4cx_store_group_fp32(
                    output_gbuf + b * out_b_stride,
                    output + b * single_batch_output_size,
                    sum + b * single_batch_output_size,
                    outH * outW,
                    oc_group,
                    g,
                    0,
                    cp.fuse_flag);
            }
            PRAGMA_OMP_BARRIER()
        }
    }  // close loop over group

}
    return ppl::common::RC_SUCCESS;
}

bool conv2d_n4cx_direct_fp32_offline_manager::is_supported() {
    return true;
}

ppl::common::RetCode conv2d_n4cx_direct_fp32_offline_manager::fast_init_schedule_param() {
    sched_param_.oh_blk = 1;
    sched_param_.ow_blk = 10;
    sched_param_.oc_blk = 8;
    sched_param_.ic_blk = 64;
    if (sched_param_.oc_blk != 8 && sched_param_.oc_blk != 4) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (sched_param_.ic_blk != 64) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}


// NOTE: (oc, ic, kh, kw) -> (oc/8, ic/4, kh, kw, 4ic, 8oc)
static inline int64_t ppl_arm_server_kernel_fp32_conv_direct_n4cx_get_converted_filter_size(
    const int64_t num_group,
    const int64_t in_c,
    const int64_t out_c,
    const int64_t ker_h,
    const int64_t ker_w)
{
    const int64_t ic_group  = in_c / num_group;
    const int64_t oc_group  = out_c / num_group;
    const int64_t ic_g_pck  = CEIL4(ic_group);
    const int64_t oc_g_pck2 = CEIL8(oc_group);
    return CEIL128( num_group * oc_g_pck2 * ic_g_pck * ker_h * ker_w * sizeof(float) ) + 128;
}

ppl::common::RetCode conv2d_n4cx_direct_fp32_offline_manager::pick_best_schedule_param(const ppl::nn::TensorShape &src_shape, double &run_time, bool tune_blocksize) {
    const int64_t num_output = param_.num_output;
    const int64_t channels = param_.channels;
    const int64_t kernel_h = param_.kernel_h;
    const int64_t kernel_w = param_.kernel_w;

    if (src_shape.GetDimCount() < 4) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int64_t num_batch = src_shape.GetDim(0);
    const int64_t src_h = src_shape.GetDim(2);
    const int64_t src_w = src_shape.GetDim(3);
    const int64_t dst_h = ((src_h + 2 * param_.pad_h - param_.dilation_h * (param_.kernel_h - 1) - 1) / param_.stride_h + 1);
    const int64_t dst_w = ((src_w + 2 * param_.pad_w - param_.dilation_w * (param_.kernel_w - 1) - 1) / param_.stride_w + 1);
    ppl::nn::TensorShape dst_shape;
    dst_shape.Reshape({num_batch, num_output, dst_h, dst_w});

    uint64_t cvt_filter_size = ppl_arm_server_kernel_fp32_conv_direct_n4cx_get_converted_filter_size(
        param_.group, channels, num_output, kernel_h, kernel_w);
    uint64_t cvt_bias_size  = CEIL4(num_output) * sizeof(float);
    uint64_t src_size = num_batch * CEIL4(channels)   * src_h * src_w * sizeof(float);
    uint64_t dst_size = num_batch * CEIL4(num_output) * dst_h * dst_w * sizeof(float);
    float *cvt_filter = (float*)allocator_->Alloc(cvt_filter_size);
    float *cvt_bias   = (float*)allocator_->Alloc(cvt_bias_size);
    float *src        = (float*)allocator_->Alloc(src_size);
    float *dst        = (float*)allocator_->Alloc(dst_size);

    for (int64_t idx = 0; idx < cvt_filter_size/sizeof(float); idx++) {
        cvt_filter[idx] = float(rand())/float((RAND_MAX)) - 0.5;
    }
    for (int64_t idx = 0; idx < cvt_bias_size/sizeof(float); idx++) {
        cvt_bias[idx] = float(rand())/float((RAND_MAX)) - 0.5;
    }
    for (int64_t idx = 0; idx < src_size/sizeof(float); idx++) {
        src[idx] = float(rand())/float((RAND_MAX)) - 0.5;
    }
    for (int64_t idx = 0; idx < dst_size/sizeof(float); idx++) {
        dst[idx] = float(rand())/float((RAND_MAX)) - 0.5;
    }

    std::vector<int64_t> candidate_oc_blk_list = {8};
    std::vector<int64_t> candidate_ic_blk_list = {64};

    if (tune_blocksize) {
        candidate_oc_blk_list = {8};
        // candidate_ic_blk_list = {32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 80, 96, 112, 128};
        candidate_ic_blk_list = {32, /*36, 40, 44,*/ 48, /*52, 56, 60,*/ 64, /*68, 72, 80, */96, 112, 128};       
    }

    int64_t best_oc_blk = 8;
    int64_t best_ic_blk = 64;
    int64_t best_run_time = std::numeric_limits<int64_t>::max();

    const int num_warmup_iter = 1;
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
            float *tmp_buffer = (float*)allocator_->Alloc(tmp_buf_size);
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
            // LOG(INFO) << "using time: " << elapsed_time / num_benchmark_iter / 1000 << " ms";
            if (elapsed_time < best_run_time) {
                best_oc_blk = oc_blk;
                best_ic_blk = ic_blk;
                best_run_time = elapsed_time;
            }

            allocator_->Free(tmp_buffer);
            delete conv_exe;

            if (ic_blk >= channels / param_.group) break;
        }
        if (oc_blk >= num_output / param_.group) break;
    }

    cvt_filter_ = nullptr;
    cvt_bias_ = nullptr;
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

// NOTE: (oc, ic, kh, kw) -> (oc/8, ic/4, kh, kw, 4ic, 8oc)
static void ppl_arm_server_kernel_fp32_conv_direct_n4cx_convert_filter(
    const float *filter,
    float *converted_filter,
    const int64_t num_group,
    const int64_t in_c,
    const int64_t out_c,
    const int64_t fltH,
    const int64_t fltW
) {
    const int64_t ocS = 8;
    const int64_t ic_group  = in_c / num_group;
    const int64_t oc_group  = out_c / num_group;
    const int64_t ic_g_pck  = CEIL4(ic_group);
    const int64_t oc_g_pck2 = CEIL8(oc_group);

    for (int64_t g = 0; g < num_group; g++) {
        const float *filter_g_base = filter + g * oc_group * ic_group * fltH * fltW;
        float *cvt_filter_g_base   = converted_filter + g * oc_g_pck2 * ic_g_pck * fltH * fltW;
        for (int64_t oc = 0; oc < oc_group; oc++) {
            for (int64_t ic = 0; ic < ic_group; ic++) {
                for (int64_t kh = 0; kh < fltH; kh++) {
                    for (int64_t kw = 0; kw < fltW; kw++) {
                        const int64_t cvt_index = (oc/ocS) * DIV_CEIL(ic_group, ICBLK()) * fltH * fltW * ICBLK()        * ocS      +
                                                             (ic/ICBLK())                * fltH * fltW * ICBLK()        * ocS      +
                                                                                           kh   * fltW * ICBLK()        * ocS      +
                                                                                                  kw   * ICBLK()        * ocS      +
                                                                                                         (ic % ICBLK()) * ocS      +
                                                                                                                          oc % ocS   ; 
                        cvt_filter_g_base[cvt_index] = filter_g_base[oc * ic_group*fltH*fltW + ic * fltH*fltW + kh * fltW + kw];
                    }
                }
            }
            for (int64_t ic = ic_group; ic < ic_g_pck; ic++) {
                for (int64_t kh = 0; kh < fltH; kh++) {
                    for (int64_t kw = 0; kw < fltW; kw++) {
                        const int64_t cvt_index = (oc/ocS) * DIV_CEIL(ic_group, ICBLK()) * fltH * fltW * ICBLK()        * ocS      +
                                                            (ic/ICBLK())                 * fltH * fltW * ICBLK()        * ocS      +
                                                                                           kh   * fltW * ICBLK()        * ocS      +
                                                                                                  kw   * ICBLK()        * ocS      +
                                                                                                         (ic % ICBLK()) * ocS      +
                                                                                                                          oc % ocS   ; 
                        cvt_filter_g_base[cvt_index] = 0.0f;
                    }
                }
            }
        }

        for (int64_t oc = oc_group; oc < oc_g_pck2; oc++) {
            for (int64_t ic = 0; ic < ic_g_pck; ic++) {
                for (int64_t kh = 0; kh < fltH; kh++) {
                    for (int64_t kw = 0; kw < fltW; kw++) {
                        const int64_t cvt_index = (oc/ocS) * DIV_CEIL(ic_group, ICBLK()) * fltH * fltW * ICBLK()        * ocS      +
                                                            (ic/ICBLK())                 * fltH * fltW * ICBLK()        * ocS      +
                                                                                           kh   * fltW * ICBLK()        * ocS      +
                                                                                                  kw   * ICBLK()        * ocS      +
                                                                                                         (ic % ICBLK()) * ocS      +
                                                                                                                          oc % ocS   ; 
                        cvt_filter_g_base[cvt_index] = 0.0f;
                    }
                }
            }
        }
    }
}

// should be called after init_schedule_param
ppl::common::RetCode conv2d_n4cx_direct_fp32_offline_manager::gen_cvt_weights(const void *filter, const void *bias) {
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_group = param_.group;
    const int64_t num_output = param_.num_output;
    const int64_t channels = param_.channels;
    const int64_t kernel_h = param_.kernel_h;
    const int64_t kernel_w = param_.kernel_w;
    
    cvt_bias_size_ = CEIL4(num_output) * sizeof(float);
    cvt_bias_      = allocator_->Alloc(cvt_bias_size_);
    int64_t padding_offset_bytes = num_output * sizeof(float);
    int64_t padding_bytes        = (CEIL4(num_output) - num_output) * sizeof(float);
    memcpy(cvt_bias_, bias, num_output * sizeof(float));
    memset(cvt_bias_ + padding_offset_bytes, 0, padding_bytes);
    
    if (sched_param_.oc_blk == 8) {
        cvt_filter_size_ = ppl_arm_server_kernel_fp32_conv_direct_n4cx_get_converted_filter_size(
                            num_group, channels, num_output, kernel_h, kernel_w);
        cvt_filter_      = allocator_->Alloc(cvt_filter_size_);
        ppl_arm_server_kernel_fp32_conv_direct_n4cx_convert_filter(
                (const float *)filter,
                (float *)cvt_filter_,
                num_group,
                channels,
                num_output,
                kernel_h,
                kernel_w);
        return ppl::common::RC_SUCCESS;
    }
    return ppl::common::RC_INVALID_VALUE;
}

conv2d_runtime_executor *conv2d_n4cx_direct_fp32_offline_manager::gen_executor() {
    return new conv2d_n4cx_direct_fp32_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}


#undef CBLK
#undef ICBLK
#undef OCBLK

}}}; // namespace ppl::kernel::arm_server