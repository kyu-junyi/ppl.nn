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

#include "ppl/kernel/arm_server/conv2d/neon/fp32/direct_ndarray/conv2d_direct_ndarray_fp32.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp32/direct_ndarray/conv_direct_ndarray_kernel.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp32/direct_ndarray/conv_direct_ndarray_h1w1_kernel.h"

#include <arm_neon.h>
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

namespace ppl { namespace kernel { namespace arm_server {

#define CBLK() 4

#define ICBLK() CBLK()
#define OCBLK() CBLK()

uint64_t conv2d_direct_ndarray_fp32_runtime_executor::cal_temp_buffer_size()
{
    return 0;
}

void conv2d_direct_ndarray_fp32_runtime_executor::adjust_schedule_param()
{
    const int64_t num_threads                = PPL_OMP_MAX_THREADS();
    sched_param_.temp_buffer_size_per_thread = cal_temp_buffer_size() / num_threads;
    return;
}

ppl::common::RetCode conv2d_direct_ndarray_fp32_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_direct_ndarray_fp32_runtime_executor::execute()
{
    PRAGMA_OMP_PARALLEL()
    {
        const conv2d_param &cp                              = *conv_param_;
        const conv2d_direct_ndarray_fp32_schedule_param &sp = sched_param_;

        const float *input                          = (const float *)src_;
        const float *cvt_filter                     = (const float *)cvt_filter_;
        const float *bias                           = (const float *)cvt_bias_;
        float *output                               = (float *)dst_;
        float *sum                                  = (float *)sum_;
        float *tmp_buffer                           = (float *)temp_buffer_;
        const int64_t inH                           = src_shape_->GetDim(2);
        const int64_t inW                           = src_shape_->GetDim(3);
        const int64_t inC                           = src_shape_->GetDim(1);
        const int64_t outC                          = cp.num_output;
        const int64_t outH                          = dst_shape_->GetDim(2);
        const int64_t outW                          = dst_shape_->GetDim(3);
        const int64_t fltH                          = cp.kernel_h;
        const int64_t fltW                          = cp.kernel_w;
        const int64_t padH                          = cp.pad_h;
        const int64_t padW                          = cp.pad_w;
        const int64_t strdH                         = cp.stride_h;
        const int64_t strdW                         = cp.stride_w;
        const int64_t dltnH                         = cp.dilation_h;
        const int64_t dltnW                         = cp.dilation_w;
        const int64_t single_core_tmp_buffer_offset = sp.temp_buffer_size_per_thread / sizeof(float);
        const int64_t num_batch                     = src_shape_->GetDim(0);

        int64_t ow_inner_start = std::max((int64_t)0, DIV_CEIL((padW - 0 * dltnW), strdW)); // inclusive
        int64_t ow_inner_end   = std::min((int64_t)outW, DIV_CEIL((inW + padW - (fltW - 1) * dltnW), strdW)); // exclusive
        ow_inner_start         = std::min(ow_inner_start, outW);
        ow_inner_end           = std::max(ow_inner_end, ow_inner_start);

        // const int64_t inC_pck = CEIL4(inC);
        const int64_t outC_pck = CEIL4(outC);

        const int64_t otH = 1;
        const int64_t otW = 14;

        const int64_t ocS = OCBLK() * 2; // 64
        const int64_t icS = 128;

        const int64_t icV_bytes = 16; // ICBLK() * sizeof(float);
        const int64_t ocV_bytes = 16; // OCBLK() * sizeof(float);

        const int64_t inH_x_inW_x_icV_bytes   = inH * inW * icV_bytes;
        const int64_t outH_x_outW_x_ocV_bytes = outH * outW * ocV_bytes;
        const int64_t dltnH_x_icV_bytes       = dltnH * icV_bytes;
        const int64_t dltnW_x_icV_bytes       = dltnW * icV_bytes;
        const int64_t strdW_x_icV_bytes       = strdW * icV_bytes;
        const int64_t dltnH_x_inW_x_icV_bytes = inW * dltnH_x_icV_bytes;
        const int64_t fltW_x_icV_x_ocS_bytes  = fltW * ocS * icV_bytes;

        const int64_t input_hw_num        = inH * inW;
        const int64_t input_chw_num       = inC * input_hw_num;
        const int64_t output_hw_num       = outH * outW;
        const int64_t output_batch_stride = outC_pck * output_hw_num;
        const int64_t output_hwcb_num     = output_hw_num * CBLK();
        const int64_t output_wcb_num      = outW * CBLK();
        const int64_t flt_ichw_num        = inC * fltH * fltW;
        const int64_t flt_ic_stride       = fltH * fltW * ocS;

        // const int64_t ocL1S = 64; (void)ocL1S;
        // const int64_t icL1S = 128; (void)icL1S;

#if not defined PPL_USE_ARM_SERVER_OMP
        for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
            const float *input_batch_base = input + batch_id * input_chw_num;
            float *output_batch_base_ptr  = output + batch_id * output_batch_stride;
            for (int64_t ic_l1 = 0; ic_l1 < inC; ic_l1 += icS) {
                const int64_t ic_remain    = std::min(icS, inC - ic_l1);
                const uint32_t fuse_flag   = (ic_l1 + icS >= inC) ? cp.fuse_flag : conv_fuse_flag::NONE;
                const float *input_ic_base = input_batch_base + ic_l1 * input_hw_num;
                for (int64_t oc_l1 = 0; oc_l1 < outC_pck; oc_l1 += ocS) {
                    const float *filter_cc_base     = cvt_filter + oc_l1 * flt_ichw_num + ic_l1 * flt_ic_stride;
                    const float *const bias_oc_base = (ic_l1 == 0) ? (bias + oc_l1) : nullptr;
                    const int64_t oc_remains        = std::min(ocS, outC_pck - oc_l1);
                    const ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel_func_t *const conv_direct_kernel_func_table =
                        (oc_remains > OCBLK()) ? ppl_arm_server_kernel_fp32_conv_direct_ndarray_oc8_kernel_func_table : ppl_arm_server_kernel_fp32_conv_direct_ndarray_oc4_kernel_func_table;
                    const ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_h1w1_kernel_func_t conv_direct_kernel_h1w1_func =
                        (oc_remains > OCBLK()) ? ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_h1w1_kernel<8> : ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_h1w1_kernel<4>;
                    for (int64_t oh = 0; oh < outH; oh += otH) {
                        float *output_h_base = output + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
                        float *sum_h_base    = sum + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
#else
        for (int64_t ic_l1 = 0; ic_l1 < inC; ic_l1 += icS) {
            const uint32_t fuse_flag = (ic_l1 + icS >= inC) ? cp.fuse_flag : 0;
            PRAGMA_OMP_FOR_COLLAPSE(3)
            for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
                for (int64_t oc_l1 = 0; oc_l1 < outC_pck; oc_l1 += ocS) {
                    for (int64_t oh = 0; oh < outH; oh += otH) {
                        const float *input_ic_base      = input + batch_id * input_chw_num + ic_l1 * input_hw_num;
                        float *output_h_base            = output + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
                        float *sum_h_base               = sum + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
                        const float *filter_cc_base     = cvt_filter + oc_l1 * flt_ichw_num + ic_l1 * flt_ic_stride;
                        const float *const bias_oc_base = (ic_l1 == 0) ? (bias + oc_l1) : nullptr;
                        const int64_t ic_remain         = std::min(icS, inC - ic_l1);
                        const int64_t oc_remains        = std::min(ocS, outC_pck - oc_l1);
                        const ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel_func_t *const conv_direct_kernel_func_table =
                            (oc_remains > OCBLK()) ? ppl_arm_server_kernel_fp32_conv_direct_ndarray_oc8_kernel_func_table : ppl_arm_server_kernel_fp32_conv_direct_ndarray_oc4_kernel_func_table;
                        const ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_h1w1_kernel_func_t conv_direct_kernel_h1w1_func =
                            (oc_remains > OCBLK()) ? ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_h1w1_kernel<8> : ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_h1w1_kernel<4>;
#endif
                        const int64_t ih          = -padH + oh * strdH;
                        int64_t fltH_start        = DIV_CEIL(std::max((int64_t)0, -ih), dltnH); // std::max((int64_t)0, DIV_CEIL((padH-oh*strdH), dltnH));
                        int64_t fltH_end          = std::min(fltH, DIV_CEIL((inH - ih), dltnH));
                        int64_t fltH_valid        = fltH_end - fltH_start;
                        const float *input_h_base = input_ic_base + ih * inW;

                        for (int64_t ow = 0; ow < ow_inner_start; ow++) {
                            const int64_t iw   = -padW + ow * strdW;
                            int64_t fltW_start = DIV_CEIL(std::max((int64_t)0, -iw), dltnW);
                            int64_t fltW_end   = std::min(fltW, DIV_CEIL((inW - iw), dltnW));
                            conv_direct_kernel_h1w1_func(
                                input_h_base + iw,
                                filter_cc_base,
                                bias_oc_base,
                                output_h_base + ow * OCBLK(),
                                sum_h_base + ow * OCBLK(),
                                input_hw_num,
                                ic_remain,
                                fltH_start,
                                fltH_end,
                                fltW_start,
                                fltW_end,
                                fltW,
                                flt_ic_stride,
                                dltnH * inW,
                                dltnW,
                                output_hwcb_num,
                                fuse_flag);
                        } // close loop over ow(1/3):head

                        const float *input_kh_base = input_h_base + fltH_start * dltnH * inW;
                        for (int64_t ow = ow_inner_start; ow < ow_inner_end; ow += otW) {
                            const int64_t ow_len = std::min(otW, ow_inner_end - ow);
                            const int64_t iw     = -padW + ow * strdW;
                            conv_direct_kernel_func_table[ow_len](
                                input_kh_base + iw,
                                filter_cc_base + (fltH_start * fltW * OCBLK() * 2),
                                bias_oc_base,
                                output_h_base + ow * OCBLK(),
                                sum_h_base + ow * OCBLK(),
                                inH,
                                inW,
                                ic_remain,
                                fltH_valid,
                                fltW,
                                strdW,
                                dltnH,
                                dltnW,
                                flt_ic_stride,
                                output_hwcb_num,
                                fuse_flag);

                        } // close loop over ow(2/3):body

                        for (int64_t ow = ow_inner_end; ow < outW; ow++) {
                            const int64_t iw   = -padW + ow * strdW;
                            int64_t fltW_start = DIV_CEIL(std::max((int64_t)0, -iw), dltnW);
                            int64_t fltW_end   = std::min(fltW, DIV_CEIL((inW - iw), dltnW));

                            conv_direct_kernel_h1w1_func(
                                input_h_base + iw,
                                filter_cc_base,
                                bias_oc_base,
                                output_h_base + ow * CBLK(),
                                sum_h_base + ow * OCBLK(),
                                input_hw_num,
                                ic_remain,
                                fltH_start,
                                fltH_end,
                                fltW_start,
                                fltW_end,
                                fltW,
                                flt_ic_stride,
                                dltnH * inW,
                                dltnW,
                                output_hwcb_num,
                                fuse_flag);
                        } // close loop over ow(3/3):tail

                    } // close loop over oh
                } // close loop over ic l1 section
            } // close loop over oc l1 section
        } // close loop over batch
    }
    return ppl::common::RC_SUCCESS;
}

bool conv2d_direct_ndarray_fp32_offline_manager::is_supported()
{
    return true;
}

ppl::common::RetCode conv2d_direct_ndarray_fp32_offline_manager::fast_init_schedule_param()
{
    sched_param_.oh_blk = 1;
    sched_param_.ow_blk = 14;
    sched_param_.oc_blk = 8;
    sched_param_.ic_blk = 128;
    if (sched_param_.oc_blk != 8) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (sched_param_.ic_blk != 128) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_direct_ndarray_fp32_offline_manager::pick_best_schedule_param(const ppl::nn::TensorShape &src_shape, double &run_time, bool tune_blocksize)
{
    return fast_init_schedule_param();
}

// NOTE: (oc, ic, kh, kw) -> (oc/8, ic, kh, kw, 8oc)
static inline int64_t ppl_arm_server_kernel_fp32_conv_direct_n4cx_get_converted_filter_size(
    const int64_t inC,
    const int64_t outC,
    const int64_t fltH,
    const int64_t fltW)
{
    return CEIL128(((outC + 7) & (~7)) * inC * fltH * fltW * sizeof(float)) + 128;
}

// NOTE: (oc, ic, kh, kw) -> (oc/8, ic, kh, kw, 8oc)
static void ppl_arm_server_kernel_fp32_conv_direct_n4cx_convert_filter(
    const float *filter,
    float *converted_filter,
    const int64_t inC,
    const int64_t outC,
    const int64_t fltH,
    const int64_t fltW)
{
    const int64_t ocS = OCBLK() * 2;
    for (int64_t oc = 0; oc < outC; oc++) {
        for (int64_t ic = 0; ic < inC; ic++) {
            for (int64_t kh = 0; kh < fltH; kh++) {
                for (int64_t kw = 0; kw < fltW; kw++) {
                    const int64_t cvt_index = (oc / ocS) * inC * fltH * fltW * ocS +
                                              ic * fltH * fltW * ocS +
                                              kh * fltW * ocS +
                                              kw * ocS +
                                              oc % ocS;
                    converted_filter[cvt_index] = filter[oc * inC * fltH * fltW + ic * fltH * fltW + kh * fltW + kw];
                }
            }
        }
    }

    for (int64_t oc = outC; oc < CEIL4(outC); oc++) {
        for (int64_t ic = 0; ic < inC; ic++) {
            for (int64_t kh = 0; kh < fltH; kh++) {
                for (int64_t kw = 0; kw < fltW; kw++) {
                    const int64_t cvt_index = (oc / ocS) * inC * fltH * fltW * ocS +
                                              ic * fltH * fltW * ocS +
                                              kh * fltW * ocS +
                                              kw * ocS +
                                              oc % ocS;
                    converted_filter[cvt_index] = 0.0f;
                }
            }
        }
    }
}

// should be called after init_schedule_param
ppl::common::RetCode conv2d_direct_ndarray_fp32_offline_manager::gen_cvt_weights(const void *filter, const void *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    cvt_bias_size_               = CEIL4(num_output) * sizeof(float);
    cvt_bias_                    = allocator_->Alloc(cvt_bias_size_);
    int64_t padding_offset_bytes = num_output * sizeof(float);
    int64_t padding_bytes        = (CEIL4(num_output) - num_output) * sizeof(float);
    memcpy(cvt_bias_, bias, num_output * sizeof(float));
    memset(cvt_bias_ + padding_offset_bytes, 0, padding_bytes);

    if (sched_param_.oc_blk == 8) {
        cvt_filter_size_ = ppl_arm_server_kernel_fp32_conv_direct_n4cx_get_converted_filter_size(
            channels, num_output, kernel_h, kernel_w);
        cvt_filter_ = allocator_->Alloc(cvt_filter_size_);
        ppl_arm_server_kernel_fp32_conv_direct_n4cx_convert_filter(
            (const float *)filter,
            (float *)cvt_filter_,
            channels,
            num_output,
            kernel_h,
            kernel_w);
        return ppl::common::RC_SUCCESS;
    }
    return ppl::common::RC_INVALID_VALUE;
}

conv2d_runtime_executor *conv2d_direct_ndarray_fp32_offline_manager::gen_executor()
{
    return new conv2d_direct_ndarray_fp32_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}

#undef CBLK
#undef ICBLK
#undef OCBLK

}}}; // namespace ppl::kernel::arm_server