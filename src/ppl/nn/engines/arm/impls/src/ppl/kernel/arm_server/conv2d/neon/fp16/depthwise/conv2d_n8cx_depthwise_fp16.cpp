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

#include "ppl/kernel/arm_server/conv2d/neon/fp16/depthwise/conv2d_n8cx_depthwise_fp16.h"

#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "ppl/common/arm/sysinfo.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

#define FLOOR8(val)   ((val) & (~7))
#define CEIL8(val)   (((val) + 7) & (~7))
#define CEIL128(val) (((val) + 127) & (~127))

#define CBLK() 8
#define ICBLK() CBLK()
#define OCBLK() CBLK()

namespace ppl { namespace kernel { namespace arm_server {

static inline void prefetch_l1(const void *ptr, size_t offset)
{
    asm volatile(
        "prfm pldl1keep, [%0, %1]\n\t"
        :
        :"r"(ptr), "r"(offset)
        :"cc","memory"
    );
}

static inline void conv_n8cx_depthwise_general_h1w1_kernel(
    const __fp16 * cvt_filter_ptr,
    const __fp16 * input_ptr,
          __fp16 * output_ptr,
          __fp16 * sum_ptr,
    const float16x8_t vbias,
    const int64_t inW,
    const int64_t fltW,
    const int64_t ih_base,
    const int64_t iw_base,
    const int64_t fltH_start,
    const int64_t fltH_end,
    const int64_t fltW_start,
    const int64_t fltW_end,
    const int64_t dltnH,
    const int64_t dltnW,
    const int64_t fuse_flag)
{
    float16x8_t vout = vbias;
    for (int64_t fh = fltH_start; fh < fltH_end; fh++) {
        const __fp16 *input_h_base = input_ptr + fh * dltnH * inW * ICBLK();
        const __fp16 *filter_h_base = cvt_filter_ptr + fh * fltW * CBLK();
        for (int64_t fw = fltW_start; fw < fltW_end; fw++) {
            float16x8_t vin = vld1q_f16(input_h_base + fw * dltnW * ICBLK());
            float16x8_t vflt = vld1q_f16(filter_h_base + fw * CBLK());
            vout = vfmaq_f16(vout, vin, vflt);
        }
    }
    if (fuse_flag & conv_fuse_flag::SUM) {
        vout = vaddq_f16(vout, vld1q_f16(sum_ptr));
    }
    if (fuse_flag & conv_fuse_flag::RELU) {
        vout = vmaxq_f16(vout, vdupq_n_f16(0.0f));
    }
    if (fuse_flag & conv_fuse_flag::RELU6) {
        vout = vminq_f16(vout, vdupq_n_f16(6.0f));
    }
    vst1q_f16(output_ptr, vout);
}

static inline void conv_n8cx_depthwise_general_h1w8_kernel(
    const __fp16 * cvt_filter_ptr,
    const __fp16 * input_ptr,
          __fp16 * output_ptr,
          __fp16 * sum_ptr,
    const float16x8_t vbias,
    const int64_t fltW,
    const int64_t strdW,
    const int64_t ih_base,
    const int64_t iw_base,
    const int64_t fltH_start,
    const int64_t fltH_end,
    const int64_t dltnH_x_inW,
    const int64_t dltnW,
    const int64_t fuse_flag)
{
    float16x8_t vout0 = vbias;
    float16x8_t vout1 = vbias;
    float16x8_t vout2 = vbias;
    float16x8_t vout3 = vbias;
    float16x8_t vout4 = vbias;
    float16x8_t vout5 = vbias;
    float16x8_t vout6 = vbias;
    float16x8_t vout7 = vbias;

    for (int64_t fh = fltH_start; fh < fltH_end; fh++) {
        const __fp16 *filter_h_base = cvt_filter_ptr + fh * fltW * CBLK();
        const __fp16 *input_h_base = input_ptr + fh * dltnH_x_inW * ICBLK();
        for (int64_t fw = 0; fw < fltW; fw++) {
            const __fp16* input_base = input_h_base + fw * dltnW * ICBLK();
            float16x8_t vflt = vld1q_f16(filter_h_base + fw * CBLK());

            float16x8_t vin0 = vld1q_f16(input_base                      );
            float16x8_t vin1 = vld1q_f16(input_base + strdW * OCBLK()    );
            float16x8_t vin2 = vld1q_f16(input_base + strdW * OCBLK() * 2);
            float16x8_t vin3 = vld1q_f16(input_base + strdW * OCBLK() * 3);
            float16x8_t vin4 = vld1q_f16(input_base + strdW * OCBLK() * 4);
            float16x8_t vin5 = vld1q_f16(input_base + strdW * OCBLK() * 5);
            float16x8_t vin6 = vld1q_f16(input_base + strdW * OCBLK() * 6);
            float16x8_t vin7 = vld1q_f16(input_base + strdW * OCBLK() * 7);

            vout0 = vfmaq_f16(vout0, vin0, vflt);
            vout1 = vfmaq_f16(vout1, vin1, vflt);
            vout2 = vfmaq_f16(vout2, vin2, vflt);
            vout3 = vfmaq_f16(vout3, vin3, vflt);
            vout4 = vfmaq_f16(vout4, vin4, vflt);
            vout5 = vfmaq_f16(vout5, vin5, vflt);
            vout6 = vfmaq_f16(vout6, vin6, vflt);
            vout7 = vfmaq_f16(vout7, vin7, vflt);
        }
    }
    if (fuse_flag & conv_fuse_flag::SUM) {
        vout0 = vaddq_f16(vout0, vld1q_f16(sum_ptr              ));
        vout1 = vaddq_f16(vout1, vld1q_f16(sum_ptr + OCBLK()    ));
        vout2 = vaddq_f16(vout2, vld1q_f16(sum_ptr + OCBLK() * 2));
        vout3 = vaddq_f16(vout3, vld1q_f16(sum_ptr + OCBLK() * 3));
        vout4 = vaddq_f16(vout4, vld1q_f16(sum_ptr + OCBLK() * 4));
        vout5 = vaddq_f16(vout5, vld1q_f16(sum_ptr + OCBLK() * 5));
        vout6 = vaddq_f16(vout6, vld1q_f16(sum_ptr + OCBLK() * 6));
        vout7 = vaddq_f16(vout7, vld1q_f16(sum_ptr + OCBLK() * 7));
    }
    if (fuse_flag & conv_fuse_flag::RELU) {
        float16x8_t vzero = vdupq_n_f16(0.0f);
        vout0 = vmaxq_f16(vout0, vzero);
        vout1 = vmaxq_f16(vout1, vzero);
        vout2 = vmaxq_f16(vout2, vzero);
        vout3 = vmaxq_f16(vout3, vzero);
        vout4 = vmaxq_f16(vout4, vzero);
        vout5 = vmaxq_f16(vout5, vzero);
        vout6 = vmaxq_f16(vout6, vzero);
        vout7 = vmaxq_f16(vout7, vzero);
    }
    if (fuse_flag & conv_fuse_flag::RELU6) {
        float16x8_t vsix = vdupq_n_f16(6.0f);
        vout0 = vminq_f16(vout0, vsix);
        vout1 = vminq_f16(vout1, vsix);
        vout2 = vminq_f16(vout2, vsix);
        vout3 = vminq_f16(vout3, vsix);
        vout4 = vminq_f16(vout4, vsix);
        vout5 = vminq_f16(vout5, vsix);
        vout6 = vminq_f16(vout6, vsix);
        vout7 = vminq_f16(vout7, vsix);
    }
    vst1q_f16(output_ptr              , vout0);
    vst1q_f16(output_ptr + OCBLK()    , vout1);
    vst1q_f16(output_ptr + OCBLK() * 2, vout2);
    vst1q_f16(output_ptr + OCBLK() * 3, vout3);
    vst1q_f16(output_ptr + OCBLK() * 4, vout4);
    vst1q_f16(output_ptr + OCBLK() * 5, vout5);
    vst1q_f16(output_ptr + OCBLK() * 6, vout6);
    vst1q_f16(output_ptr + OCBLK() * 7, vout7);
}

template<const uint32_t padding, const uint32_t stride>
void conv_n8cx_depthwise_f3sx_h1w4(
    const __fp16 *converted_filter,
    const __fp16 *bias,            
    const __fp16 *input,           
          __fp16 *output,          
          __fp16 *sum,             
    const int64_t fltC,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_batch,
    const uint32_t fuse_flag);

template<>
void conv_n8cx_depthwise_f3sx_h1w4<0, 1>(
    const __fp16 *converted_filter,
    const __fp16 *bias,            
    const __fp16 *input,           
          __fp16 *output,          
          __fp16 *sum,
    const int64_t fltC,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t fltC_pck = CEIL8(fltC);
    const int64_t inHW = inH * inW;
    const int64_t outHW = outH * outW;
    const int64_t input_batch_stride = fltC_pck * inHW;
    const int64_t output_batch_stride = fltC_pck * outHW;

    for (int64_t b = 0; b < num_batch; b++) {
        for (int64_t c = 0; c < fltC; c += CBLK()) {
            const __fp16 *converted_filter_c_base = converted_filter + c * 9;
            const __fp16 *bias_c_base = bias + c;
            const __fp16 *input_c_base = input + b * input_batch_stride + c * inHW;
            __fp16 *output_c_base = output + b * output_batch_stride + c * outHW;
            __fp16 *sum_c_base = sum + b * output_batch_stride + c * outHW;

            float16x8_t vflt[9];
            vflt[0] = vld1q_f16(converted_filter_c_base + 0 * CBLK());
            vflt[1] = vld1q_f16(converted_filter_c_base + 1 * CBLK());
            vflt[2] = vld1q_f16(converted_filter_c_base + 2 * CBLK());
            vflt[3] = vld1q_f16(converted_filter_c_base + 3 * CBLK());
            vflt[4] = vld1q_f16(converted_filter_c_base + 4 * CBLK());
            vflt[5] = vld1q_f16(converted_filter_c_base + 5 * CBLK());
            vflt[6] = vld1q_f16(converted_filter_c_base + 6 * CBLK());
            vflt[7] = vld1q_f16(converted_filter_c_base + 7 * CBLK());
            vflt[8] = vld1q_f16(converted_filter_c_base + 8 * CBLK());
            float16x8_t vbias = vld1q_f16(bias_c_base);
            float16x8_t vin[18];
            float16x8_t vout[4];

            int64_t outW_align4 = (outW & (~3));

            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t oh = 0; oh < outH; oh++) {
                const __fp16 *input_h_base = input_c_base + oh * inW * ICBLK();
                __fp16 *output_h_base = output_c_base + oh * outW * OCBLK();
                __fp16 *sum_h_base = sum_c_base + oh * outW * OCBLK();

                for (int64_t ow = 0; ow < outW_align4; ow+=4) {
                    const __fp16 *input_ptr = input_h_base + ow * ICBLK();
                    __fp16 *output_ptr = output_h_base + ow * OCBLK();
                    __fp16 *sum_ptr = sum_h_base + ow * OCBLK();

                    vout[0] = vbias;
                    vout[1] = vbias;
                    vout[2] = vbias;
                    vout[3] = vbias;

                    vin[0] = vld1q_f16(input_ptr              );
                    vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                    vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                    vin[3] = vld1q_f16(input_ptr + ICBLK() * 3);
                    vin[4] = vld1q_f16(input_ptr + ICBLK() * 4);
                    vin[5] = vld1q_f16(input_ptr + ICBLK() * 5);

                    vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                    vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                    vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                    vin[9]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 3);
                    vin[10] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 4);
                    vin[11] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 5);

                    vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                    vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                    vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                    vin[15] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 3);
                    vin[16] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 4);
                    vin[17] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 5);

                    vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                    vout[1] = vfmaq_f16(vout[1], vin[1] , vflt[0]);
                    vout[2] = vfmaq_f16(vout[2], vin[2] , vflt[0]);
                    vout[3] = vfmaq_f16(vout[3], vin[3] , vflt[0]);

                    vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                    vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[1]);
                    vout[2] = vfmaq_f16(vout[2], vin[3] , vflt[1]);
                    vout[3] = vfmaq_f16(vout[3], vin[4] , vflt[1]);

                    vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                    vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[2]);
                    vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[2]);
                    vout[3] = vfmaq_f16(vout[3], vin[5] , vflt[2]);

                    vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                    vout[1] = vfmaq_f16(vout[1], vin[7] , vflt[3]);
                    vout[2] = vfmaq_f16(vout[2], vin[8] , vflt[3]);
                    vout[3] = vfmaq_f16(vout[3], vin[9] , vflt[3]);

                    vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                    vout[1] = vfmaq_f16(vout[1], vin[8] , vflt[4]);
                    vout[2] = vfmaq_f16(vout[2], vin[9] , vflt[4]);
                    vout[3] = vfmaq_f16(vout[3], vin[10], vflt[4]);

                    vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                    vout[1] = vfmaq_f16(vout[1], vin[9] , vflt[5]);
                    vout[2] = vfmaq_f16(vout[2], vin[10], vflt[5]);
                    vout[3] = vfmaq_f16(vout[3], vin[11], vflt[5]);

                    vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                    vout[1] = vfmaq_f16(vout[1], vin[13], vflt[6]);
                    vout[2] = vfmaq_f16(vout[2], vin[14], vflt[6]);
                    vout[3] = vfmaq_f16(vout[3], vin[15], vflt[6]);

                    vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                    vout[1] = vfmaq_f16(vout[1], vin[14], vflt[7]);
                    vout[2] = vfmaq_f16(vout[2], vin[15], vflt[7]);
                    vout[3] = vfmaq_f16(vout[3], vin[16], vflt[7]);

                    vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                    vout[1] = vfmaq_f16(vout[1], vin[15], vflt[8]);
                    vout[2] = vfmaq_f16(vout[2], vin[16], vflt[8]);
                    vout[3] = vfmaq_f16(vout[3], vin[17], vflt[8]);

                    if (fuse_flag & conv_fuse_flag::SUM) {
                        vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr              ));
                        vout[1] = vaddq_f16(vout[1], vld1q_f16(sum_ptr + OCBLK() * 1));
                        vout[2] = vaddq_f16(vout[2], vld1q_f16(sum_ptr + OCBLK() * 2));
                        vout[3] = vaddq_f16(vout[3], vld1q_f16(sum_ptr + OCBLK() * 3));
                    }
                    if (fuse_flag & conv_fuse_flag::RELU) {
                        float16x8_t vzero = vdupq_n_f16(0.0f);
                        vout[0] = vmaxq_f16(vout[0], vzero);
                        vout[1] = vmaxq_f16(vout[1], vzero);
                        vout[2] = vmaxq_f16(vout[2], vzero);
                        vout[3] = vmaxq_f16(vout[3], vzero);
                    }
                    if (fuse_flag & conv_fuse_flag::RELU6) {
                        float16x8_t vsix = vdupq_n_f16(6.0f);
                        vout[0] = vminq_f16(vout[0], vsix);
                        vout[1] = vminq_f16(vout[1], vsix);
                        vout[2] = vminq_f16(vout[2], vsix);
                        vout[3] = vminq_f16(vout[3], vsix);
                    }

                    vst1q_f16(output_ptr              , vout[0]);
                    vst1q_f16(output_ptr + OCBLK() * 1, vout[1]);
                    vst1q_f16(output_ptr + OCBLK() * 2, vout[2]);
                    vst1q_f16(output_ptr + OCBLK() * 3, vout[3]);
                }
                for (int64_t ow = outW_align4; ow < outW; ow++) {
                    const __fp16 *input_ptr = input_h_base + ow * ICBLK();
                    __fp16 *output_ptr = output_h_base + ow * OCBLK();
                    __fp16 *sum_ptr = sum_h_base + ow * OCBLK();

                    vout[0] = vbias;

                    vin[0] = vld1q_f16(input_ptr              );
                    vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                    vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);

                    vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                    vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                    vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);

                    vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                    vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                    vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);

                    vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                    vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                    vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                    vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                    vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                    vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                    vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                    vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                    vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);

                    if (fuse_flag & conv_fuse_flag::SUM) {
                        vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr));
                    }
                    if (fuse_flag & conv_fuse_flag::RELU) {
                        vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                    }
                    if (fuse_flag & conv_fuse_flag::RELU6) {
                        vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                    }

                    vst1q_f16(output_ptr     , vout[0]);
                }
            }
        }
    }
}
}

template<>
void conv_n8cx_depthwise_f3sx_h1w4<1, 1>(
    const __fp16 *converted_filter,
    const __fp16 *bias,            
    const __fp16 *input,           
          __fp16 *output,          
          __fp16 *sum,          
    const int64_t fltC,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t fltC_pck = CEIL8(fltC);
    const int64_t inHW = inH * inW;
    const int64_t outHW = outH * outW;
    const int64_t input_batch_stride = fltC_pck * inHW;
    const int64_t output_batch_stride = fltC_pck * outHW;

    for (int64_t b = 0; b < num_batch; b++) {
        for (int64_t c = 0; c < fltC; c += CBLK()) {
            const __fp16 *converted_filter_c_base = converted_filter + c * 9;
            const __fp16 *bias_c_base = bias + c;
            const __fp16 *input_c_base = input + b * input_batch_stride + c * inHW;
            __fp16 *output_c_base = output + b * output_batch_stride + c * outHW;
            __fp16 *sum_c_base = sum + b * output_batch_stride + c * outHW;

            float16x8_t vflt[9];
            vflt[0] = vld1q_f16(converted_filter_c_base + 0 * CBLK());
            vflt[1] = vld1q_f16(converted_filter_c_base + 1 * CBLK());
            vflt[2] = vld1q_f16(converted_filter_c_base + 2 * CBLK());
            vflt[3] = vld1q_f16(converted_filter_c_base + 3 * CBLK());
            vflt[4] = vld1q_f16(converted_filter_c_base + 4 * CBLK());
            vflt[5] = vld1q_f16(converted_filter_c_base + 5 * CBLK());
            vflt[6] = vld1q_f16(converted_filter_c_base + 6 * CBLK());
            vflt[7] = vld1q_f16(converted_filter_c_base + 7 * CBLK());
            vflt[8] = vld1q_f16(converted_filter_c_base + 8 * CBLK());
            float16x8_t vbias = vld1q_f16(bias_c_base);
            float16x8_t vin[18];
            float16x8_t vout[4];

            const int64_t oh_inner_start = 1;  // inclusive index
            const int64_t ow_inner_start = 1;  // inclusive index
            int64_t oh_inner_end = inH - 1;  // exclusive index
            int64_t ow_inner_end = inW - 1;  // exclusive index

            oh_inner_end = std::max(oh_inner_end, oh_inner_start);
            ow_inner_end = std::max(ow_inner_end, ow_inner_start);

            int64_t ow_inner_end_align4 = ((ow_inner_end - ow_inner_start) & (~3)) + ow_inner_start;

            // std::cout << oh_inner_end << std::endl;
            // std::cout << ow_inner_end_align4 << std::endl;
            // std::cout << ow_inner_end << std::endl;

            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t oh = 0; oh < outH; oh++) {
                const __fp16 *input_h_base = input_c_base + (oh - 1) * inW * ICBLK();
                __fp16 *output_h_base = output_c_base + oh * outW * OCBLK();
                __fp16 *sum_h_base = sum_c_base + oh * outW * OCBLK();

                if (oh == 0 || oh == outH - 1) {
                    bool ih0_valid = (oh >= 1);
                    bool ih2_valid = (oh < inH - 1);
                    
                    {
                        const __fp16 *input_ptr = input_h_base;
                        bool iw2_valid = (1 < inW);

                        vout[0] = vbias;

                        if (ih0_valid) {
                            vin[1] =               vld1q_f16(input_ptr          )                    ;
                            vin[2] = (iw2_valid) ? vld1q_f16(input_ptr + ICBLK()) : vdupq_n_f16(0.0f);
                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        }

                        vin[7] =               vld1q_f16(input_ptr + inW * ICBLK()          )                    ;
                        vin[8] = (iw2_valid) ? vld1q_f16(input_ptr + inW * ICBLK() + ICBLK()) : vdupq_n_f16(0.0f);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);

                        if (ih2_valid) {
                            vin[13] =               vld1q_f16(input_ptr + inW * ICBLK() * 2          )                    ;
                            vin[14] = (iw2_valid) ? vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK()) : vdupq_n_f16(0.0f);
                            vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                            vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                        }
                        
                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }
                        vst1q_f16(output_h_base, vout[0]);
                    }
                    for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow+=4) {
                        const __fp16 *input_ptr = input_h_base + (ow - 1) * ICBLK();
                        __fp16 *output_ptr = output_h_base + ow * OCBLK();
                        __fp16 *sum_ptr = sum_h_base + ow * OCBLK();

                        vout[0] = vbias;
                        vout[1] = vbias;
                        vout[2] = vbias;
                        vout[3] = vbias;
                        if (ih0_valid) {
                            vin[0] = vld1q_f16(input_ptr              );
                            prefetch_l1(input_ptr, 3 * inW * sizeof(__fp16));
                            vin[1] = vld1q_f16(input_ptr + ICBLK()    );
                            vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                            vin[3] = vld1q_f16(input_ptr + ICBLK() * 3);
                            vin[4] = vld1q_f16(input_ptr + ICBLK() * 4);
                            vin[5] = vld1q_f16(input_ptr + ICBLK() * 5);

                            vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                            vout[1] = vfmaq_f16(vout[1], vin[1] , vflt[0]);
                            vout[2] = vfmaq_f16(vout[2], vin[2] , vflt[0]);
                            vout[3] = vfmaq_f16(vout[3], vin[3] , vflt[0]);

                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                            vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[1]);
                            vout[2] = vfmaq_f16(vout[2], vin[3] , vflt[1]);
                            vout[3] = vfmaq_f16(vout[3], vin[4] , vflt[1]);

                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                            vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[2]);
                            vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[2]);
                            vout[3] = vfmaq_f16(vout[3], vin[5] , vflt[2]);
                        }

                        vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        prefetch_l1(input_ptr + inW * ICBLK(), 3 * inW * sizeof(__fp16));
                        vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                        vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                        vin[9]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 3);
                        vin[10] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 4);
                        vin[11] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 5);

                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[1] = vfmaq_f16(vout[1], vin[7] , vflt[3]);
                        vout[2] = vfmaq_f16(vout[2], vin[8] , vflt[3]);
                        vout[3] = vfmaq_f16(vout[3], vin[9] , vflt[3]);

                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[1] = vfmaq_f16(vout[1], vin[8] , vflt[4]);
                        vout[2] = vfmaq_f16(vout[2], vin[9] , vflt[4]);
                        vout[3] = vfmaq_f16(vout[3], vin[10], vflt[4]);

                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                        vout[1] = vfmaq_f16(vout[1], vin[9] , vflt[5]);
                        vout[2] = vfmaq_f16(vout[2], vin[10], vflt[5]);
                        vout[3] = vfmaq_f16(vout[3], vin[11], vflt[5]);

                        if (ih2_valid) {
                            vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                            prefetch_l1(input_ptr + inW * ICBLK() * 2, 3 * inW * sizeof(__fp16));
                            vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK()    );
                            vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                            vin[15] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 3);
                            vin[16] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 4);
                            vin[17] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 5);

                            vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                            vout[1] = vfmaq_f16(vout[1], vin[13], vflt[6]);
                            vout[2] = vfmaq_f16(vout[2], vin[14], vflt[6]);
                            vout[3] = vfmaq_f16(vout[3], vin[15], vflt[6]);

                            vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                            vout[1] = vfmaq_f16(vout[1], vin[14], vflt[7]);
                            vout[2] = vfmaq_f16(vout[2], vin[15], vflt[7]);
                            vout[3] = vfmaq_f16(vout[3], vin[16], vflt[7]);

                            vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                            vout[1] = vfmaq_f16(vout[1], vin[15], vflt[8]);
                            vout[2] = vfmaq_f16(vout[2], vin[16], vflt[8]);
                            vout[3] = vfmaq_f16(vout[3], vin[17], vflt[8]);
                        }

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr              ));
                            vout[1] = vaddq_f16(vout[1], vld1q_f16(sum_ptr + OCBLK() * 1));
                            vout[2] = vaddq_f16(vout[2], vld1q_f16(sum_ptr + OCBLK() * 2));
                            vout[3] = vaddq_f16(vout[3], vld1q_f16(sum_ptr + OCBLK() * 3));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            float16x8_t vzero = vdupq_n_f16(0.0f);
                            vout[0] = vmaxq_f16(vout[0], vzero);
                            vout[1] = vmaxq_f16(vout[1], vzero);
                            vout[2] = vmaxq_f16(vout[2], vzero);
                            vout[3] = vmaxq_f16(vout[3], vzero);
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            float16x8_t vsix = vdupq_n_f16(6.0f);
                            vout[0] = vminq_f16(vout[0], vsix);
                            vout[1] = vminq_f16(vout[1], vsix);
                            vout[2] = vminq_f16(vout[2], vsix);
                            vout[3] = vminq_f16(vout[3], vsix);
                        }
                        
                        vst1q_f16(output_ptr              , vout[0]);
                        vst1q_f16(output_ptr + OCBLK() * 1, vout[1]);
                        vst1q_f16(output_ptr + OCBLK() * 2, vout[2]);
                        vst1q_f16(output_ptr + OCBLK() * 3, vout[3]);
                    }
                    for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                        const __fp16 *input_ptr = input_h_base + (ow - 1) * ICBLK();

                        vout[0] = vbias;
                        if (ih0_valid) {
                            vin[0] = vld1q_f16(input_ptr              );
                            vin[1] = vld1q_f16(input_ptr + ICBLK()    );
                            vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                            vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        }

                        vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK()    );
                        vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);

                        if (ih2_valid) {
                            vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                            vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK()    );
                            vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                            vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                            vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                        }

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base + ow * OCBLK()));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }
                        vst1q_f16(output_h_base + ow * OCBLK(), vout[0]);
                    }
                    if (ow_inner_end < outW) {
                        const __fp16 *input_ptr = input_h_base + (inW - 2) * ICBLK();

                        vout[0] = vbias;
                        if (ih0_valid) {
                            vin[0] = vld1q_f16(input_ptr          );
                            vin[1] = vld1q_f16(input_ptr + ICBLK());
                            vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        }

                        vin[6] = vld1q_f16(input_ptr + inW * ICBLK()          );
                        vin[7] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK());
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);

                        if (ih2_valid) {
                            vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2          );
                            vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK());
                            vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                        }

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base + (outW - 1) * OCBLK()));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }
                        vst1q_f16(output_h_base + (outW - 1) * OCBLK(), vout[0]);
                    }
                }
                else {
                    {
                        const __fp16 *input_ptr = input_h_base;
                        bool iw2_valid = (1 < inW);

                        vout[0] = vbias;

                        vin[1] = vld1q_f16(input_ptr);
                        vin[7] = vld1q_f16(input_ptr + inW * ICBLK());
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2);
                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);

                        if (iw2_valid) {
                            vin[2] = vld1q_f16(input_ptr + ICBLK());
                            vin[8] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK());
                            vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK());

                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                            vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                            vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                        }
                        
                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }
                        vst1q_f16(output_h_base, vout[0]);
                    }
                    for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow+=4) {
                        int64_t iw = -1 + ow;
                        const __fp16 *input_ptr = input_h_base + iw * ICBLK();
                        __fp16 *output_ptr = output_h_base + ow * OCBLK();
                        __fp16 *sum_ptr = sum_h_base + ow * OCBLK();

                        vout[0] = vbias;
                        vout[1] = vbias;
                        vout[2] = vbias;
                        vout[3] = vbias;

                        vin[0] = vld1q_f16(input_ptr              );
                        prefetch_l1(input_ptr, 3 * inW * sizeof(__fp16));
                        vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                        vin[3] = vld1q_f16(input_ptr + ICBLK() * 3);
                        vin[4] = vld1q_f16(input_ptr + ICBLK() * 4);
                        vin[5] = vld1q_f16(input_ptr + ICBLK() * 5);

                        vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        prefetch_l1(input_ptr + inW * ICBLK(), 3 * inW * sizeof(__fp16));
                        vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                        vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                        vin[9]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 3);
                        vin[10] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 4);
                        vin[11] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 5);

                        vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                        prefetch_l1(input_ptr + inW * ICBLK() * 2, 3 * inW * sizeof(__fp16));
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                        vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                        vin[15] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 3);
                        vin[16] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 4);
                        vin[17] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 5);

                        vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                        vout[1] = vfmaq_f16(vout[1], vin[1] , vflt[0]);
                        vout[2] = vfmaq_f16(vout[2], vin[2] , vflt[0]);
                        vout[3] = vfmaq_f16(vout[3], vin[3] , vflt[0]);

                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[1]);
                        vout[2] = vfmaq_f16(vout[2], vin[3] , vflt[1]);
                        vout[3] = vfmaq_f16(vout[3], vin[4] , vflt[1]);

                        vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[2]);
                        vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[2]);
                        vout[3] = vfmaq_f16(vout[3], vin[5] , vflt[2]);

                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[1] = vfmaq_f16(vout[1], vin[7] , vflt[3]);
                        vout[2] = vfmaq_f16(vout[2], vin[8] , vflt[3]);
                        vout[3] = vfmaq_f16(vout[3], vin[9] , vflt[3]);

                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[1] = vfmaq_f16(vout[1], vin[8] , vflt[4]);
                        vout[2] = vfmaq_f16(vout[2], vin[9] , vflt[4]);
                        vout[3] = vfmaq_f16(vout[3], vin[10], vflt[4]);

                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                        vout[1] = vfmaq_f16(vout[1], vin[9] , vflt[5]);
                        vout[2] = vfmaq_f16(vout[2], vin[10], vflt[5]);
                        vout[3] = vfmaq_f16(vout[3], vin[11], vflt[5]);

                        vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                        vout[1] = vfmaq_f16(vout[1], vin[13], vflt[6]);
                        vout[2] = vfmaq_f16(vout[2], vin[14], vflt[6]);
                        vout[3] = vfmaq_f16(vout[3], vin[15], vflt[6]);

                        vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                        vout[1] = vfmaq_f16(vout[1], vin[14], vflt[7]);
                        vout[2] = vfmaq_f16(vout[2], vin[15], vflt[7]);
                        vout[3] = vfmaq_f16(vout[3], vin[16], vflt[7]);

                        vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                        vout[1] = vfmaq_f16(vout[1], vin[15], vflt[8]);
                        vout[2] = vfmaq_f16(vout[2], vin[16], vflt[8]);
                        vout[3] = vfmaq_f16(vout[3], vin[17], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr              ));
                            vout[1] = vaddq_f16(vout[1], vld1q_f16(sum_ptr + OCBLK() * 1));
                            vout[2] = vaddq_f16(vout[2], vld1q_f16(sum_ptr + OCBLK() * 2));
                            vout[3] = vaddq_f16(vout[3], vld1q_f16(sum_ptr + OCBLK() * 3));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            float16x8_t vzero = vdupq_n_f16(0.0f);
                            vout[0] = vmaxq_f16(vout[0], vzero);
                            vout[1] = vmaxq_f16(vout[1], vzero);
                            vout[2] = vmaxq_f16(vout[2], vzero);
                            vout[3] = vmaxq_f16(vout[3], vzero);
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            float16x8_t vsix = vdupq_n_f16(6.0f);
                            vout[0] = vminq_f16(vout[0], vsix);
                            vout[1] = vminq_f16(vout[1], vsix);
                            vout[2] = vminq_f16(vout[2], vsix);
                            vout[3] = vminq_f16(vout[3], vsix);
                        }
                        
                        vst1q_f16(output_ptr              , vout[0]);
                        vst1q_f16(output_ptr + OCBLK() * 1, vout[1]);
                        vst1q_f16(output_ptr + OCBLK() * 2, vout[2]);
                        vst1q_f16(output_ptr + OCBLK() * 3, vout[3]);
                    }
                    for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                        int64_t iw = -1 + ow;
                        const __fp16 *input_ptr = input_h_base + iw * ICBLK();
                        __fp16 *output_ptr = output_h_base + ow * OCBLK();

                        vout[0] = vbias;

                        vin[0] = vld1q_f16(input_ptr              );
                        vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);

                        vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                        vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);

                        vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                        vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);

                        vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                        vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                        vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                        vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            __fp16 *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }

                        vst1q_f16(output_ptr     , vout[0]);
                    }
                    if (ow_inner_end < outW) {
                        // ow = outW - 1 == inW - 1 (as outW == inW f3p1s1d1)
                        const __fp16 *input_ptr = input_h_base + (inW - 2) * ICBLK();

                        vout[0] = vbias;

                        vin[0] = vld1q_f16(input_ptr          );
                        vin[1] = vld1q_f16(input_ptr + ICBLK());

                        vin[6] = vld1q_f16(input_ptr + inW * ICBLK()          );
                        vin[7] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK());

                        vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2          );
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK());

                        vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                        vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base + (outW - 1) * OCBLK()));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }
                        
                        vst1q_f16(output_h_base + (outW - 1) * OCBLK(), vout[0]);
                    }
                }
            }
        }
    }
}
}

template<>
void conv_n8cx_depthwise_f3sx_h1w4<0, 2>(
    const __fp16 *converted_filter,
    const __fp16 *bias,            
    const __fp16 *input,           
          __fp16 *output,
          __fp16 *sum,
    const int64_t fltC,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t fltC_pck = CEIL8(fltC);
    const int64_t inHW = inH * inW;
    const int64_t outHW = outH * outW;
    const int64_t input_batch_stride = fltC_pck * inHW;
    const int64_t output_batch_stride = fltC_pck * outHW;

    for (int64_t b = 0; b < num_batch; b++) {
        for (int64_t c = 0; c < fltC; c += CBLK()) {
            const __fp16 *converted_filter_c_base = converted_filter + c * 9;
            const __fp16 *bias_c_base = bias + c;
            const __fp16 *input_c_base = input + b * input_batch_stride + c * inHW;
            __fp16 *output_c_base = output + b * output_batch_stride + c * outHW;
            __fp16 *sum_c_base = sum + b * output_batch_stride + c * outHW;

            float16x8_t vflt[9];
            vflt[0] = vld1q_f16(converted_filter_c_base + 0 * CBLK());
            vflt[1] = vld1q_f16(converted_filter_c_base + 1 * CBLK());
            vflt[2] = vld1q_f16(converted_filter_c_base + 2 * CBLK());
            vflt[3] = vld1q_f16(converted_filter_c_base + 3 * CBLK());
            vflt[4] = vld1q_f16(converted_filter_c_base + 4 * CBLK());
            vflt[5] = vld1q_f16(converted_filter_c_base + 5 * CBLK());
            vflt[6] = vld1q_f16(converted_filter_c_base + 6 * CBLK());
            vflt[7] = vld1q_f16(converted_filter_c_base + 7 * CBLK());
            vflt[8] = vld1q_f16(converted_filter_c_base + 8 * CBLK());
            float16x8_t vbias = vld1q_f16(bias_c_base);
            float16x8_t vin[18];  // double buffer
            float16x8_t vout[4];

            int64_t outW_align4 = (outW & (~3));

            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t oh = 0; oh < outH; oh++) {
                const __fp16 *input_h_base = input_c_base + oh * 2 * inW * ICBLK();
                __fp16 *output_h_base = output_c_base + oh * outW * OCBLK();
                __fp16 *sum_h_base = sum_c_base + oh * outW * OCBLK();

                for (int64_t ow = 0; ow < outW_align4; ow+=4) {
                    const __fp16 *input_ptr = input_h_base + ow * 2 * ICBLK();
                    __fp16 *output_ptr = output_h_base + ow * OCBLK();
                    __fp16 *sum_ptr = sum_h_base + ow * OCBLK();

                    vout[0] = vbias;
                    vout[1] = vbias;
                    vout[2] = vbias;
                    vout[3] = vbias;

                    vin[0] = vld1q_f16(input_ptr              );
                    vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                    vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                    vin[3] = vld1q_f16(input_ptr + ICBLK() * 3);
                    vin[4] = vld1q_f16(input_ptr + ICBLK() * 4);
                    vin[5] = vld1q_f16(input_ptr + ICBLK() * 5);
                    vin[6] = vld1q_f16(input_ptr + ICBLK() * 6);
                    vin[7] = vld1q_f16(input_ptr + ICBLK() * 7);
                    vin[8] = vld1q_f16(input_ptr + ICBLK() * 8);

                    vin[9]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                    vin[10] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                    vin[11] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                    vin[12] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 3);
                    vin[13] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 4);
                    vin[14] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 5);
                    vin[15] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 6);
                    vin[16] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 7);
                    vin[17] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 8);

                    vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                    vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[0]);
                    vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[0]);
                    vout[3] = vfmaq_f16(vout[3], vin[6] , vflt[0]);

                    vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                    vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[1]);
                    vout[2] = vfmaq_f16(vout[2], vin[5] , vflt[1]);
                    vout[3] = vfmaq_f16(vout[3], vin[7] , vflt[1]);

                    vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                    vout[1] = vfmaq_f16(vout[1], vin[4] , vflt[2]);
                    vout[2] = vfmaq_f16(vout[2], vin[6] , vflt[2]);
                    vout[3] = vfmaq_f16(vout[3], vin[8] , vflt[2]);

                    vin[0] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                    vin[1] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                    vin[2] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                    vin[3] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 3);
                    vin[4] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 4);
                    vin[5] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 5);
                    vin[6] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 6);
                    vin[7] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 7);
                    vin[8] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 8);

                    vout[0] = vfmaq_f16(vout[0], vin[9] , vflt[3]);
                    vout[1] = vfmaq_f16(vout[1], vin[11], vflt[3]);
                    vout[2] = vfmaq_f16(vout[2], vin[13], vflt[3]);
                    vout[3] = vfmaq_f16(vout[3], vin[15], vflt[3]);

                    vout[0] = vfmaq_f16(vout[0], vin[10], vflt[4]);
                    vout[1] = vfmaq_f16(vout[1], vin[12], vflt[4]);
                    vout[2] = vfmaq_f16(vout[2], vin[14], vflt[4]);
                    vout[3] = vfmaq_f16(vout[3], vin[16], vflt[4]);

                    vout[0] = vfmaq_f16(vout[0], vin[11], vflt[5]);
                    vout[1] = vfmaq_f16(vout[1], vin[13], vflt[5]);
                    vout[2] = vfmaq_f16(vout[2], vin[15], vflt[5]);
                    vout[3] = vfmaq_f16(vout[3], vin[17], vflt[5]);

                    vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[6]);
                    vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[6]);
                    vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[6]);
                    vout[3] = vfmaq_f16(vout[3], vin[6] , vflt[6]);

                    vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[7]);
                    vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[7]);
                    vout[2] = vfmaq_f16(vout[2], vin[5] , vflt[7]);
                    vout[3] = vfmaq_f16(vout[3], vin[7] , vflt[7]);

                    vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[8]);
                    vout[1] = vfmaq_f16(vout[1], vin[4] , vflt[8]);
                    vout[2] = vfmaq_f16(vout[2], vin[6] , vflt[8]);
                    vout[3] = vfmaq_f16(vout[3], vin[8] , vflt[8]);

                    if (fuse_flag & conv_fuse_flag::SUM) {
                        vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr              ));
                        vout[1] = vaddq_f16(vout[1], vld1q_f16(sum_ptr + OCBLK() * 1));
                        vout[2] = vaddq_f16(vout[2], vld1q_f16(sum_ptr + OCBLK() * 2));
                        vout[3] = vaddq_f16(vout[3], vld1q_f16(sum_ptr + OCBLK() * 3));
                    }
                    if (fuse_flag & conv_fuse_flag::RELU) {
                        float16x8_t vzero = vdupq_n_f16(0.0f);
                        vout[0] = vmaxq_f16(vout[0], vzero);
                        vout[1] = vmaxq_f16(vout[1], vzero);
                        vout[2] = vmaxq_f16(vout[2], vzero);
                        vout[3] = vmaxq_f16(vout[3], vzero);
                    }
                    if (fuse_flag & conv_fuse_flag::RELU6) {
                        float16x8_t vsix = vdupq_n_f16(6.0f);
                        vout[0] = vminq_f16(vout[0], vsix);
                        vout[1] = vminq_f16(vout[1], vsix);
                        vout[2] = vminq_f16(vout[2], vsix);
                        vout[3] = vminq_f16(vout[3], vsix);
                    }

                    vst1q_f16(output_ptr              , vout[0]);
                    vst1q_f16(output_ptr + OCBLK() * 1, vout[1]);
                    vst1q_f16(output_ptr + OCBLK() * 2, vout[2]);
                    vst1q_f16(output_ptr + OCBLK() * 3, vout[3]);
                }
                for (int64_t ow = outW_align4; ow < outW; ow++) {
                    const __fp16 *input_ptr = input_h_base + ow * 2 * ICBLK();
                    __fp16 *output_ptr = output_h_base + ow * OCBLK();

                    vout[0] = vbias;

                    vin[0] = vld1q_f16(input_ptr              );
                    vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                    vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);

                    vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                    vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                    vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);

                    vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                    vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                    vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);

                    vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                    vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                    vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                    vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                    vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                    vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                    vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                    vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                    vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);

                    if (fuse_flag & conv_fuse_flag::SUM) {
                        __fp16 *sum_ptr = sum_h_base + ow * OCBLK();
                        vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr));
                    }
                    if (fuse_flag & conv_fuse_flag::RELU) {
                        vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                    }
                    if (fuse_flag & conv_fuse_flag::RELU6) {
                        vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                    }

                    vst1q_f16(output_ptr, vout[0]);
                }
            }
        }
    }
}
}

template<>
void conv_n8cx_depthwise_f3sx_h1w4<1, 2>(
    const __fp16 *converted_filter,
    const __fp16 *bias,            
    const __fp16 *input,           
          __fp16 *output,          
          __fp16 *sum,          
    const int64_t fltC,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t fltC_pck = CEIL8(fltC);
    const int64_t inHW = inH * inW;
    const int64_t outHW = outH * outW;
    const int64_t input_batch_stride = fltC_pck * inHW;
    const int64_t output_batch_stride = fltC_pck * outHW;

    for (int64_t b = 0; b < num_batch; b++) {
        for (int64_t c = 0; c < fltC; c += CBLK()) {
            const __fp16 *converted_filter_c_base = converted_filter + c * 9;
            const __fp16 *bias_c_base = bias + c;
            const __fp16 *input_c_base = input + b * input_batch_stride + c * inHW;
            __fp16 *output_c_base = output + b * output_batch_stride + c * outHW;
            __fp16 *sum_c_base = sum + b * output_batch_stride + c * outHW;


            float16x8_t vflt[9];
            vflt[0] = vld1q_f16(converted_filter_c_base + 0 * CBLK());
            vflt[1] = vld1q_f16(converted_filter_c_base + 1 * CBLK());
            vflt[2] = vld1q_f16(converted_filter_c_base + 2 * CBLK());
            vflt[3] = vld1q_f16(converted_filter_c_base + 3 * CBLK());
            vflt[4] = vld1q_f16(converted_filter_c_base + 4 * CBLK());
            vflt[5] = vld1q_f16(converted_filter_c_base + 5 * CBLK());
            vflt[6] = vld1q_f16(converted_filter_c_base + 6 * CBLK());
            vflt[7] = vld1q_f16(converted_filter_c_base + 7 * CBLK());
            vflt[8] = vld1q_f16(converted_filter_c_base + 8 * CBLK());
            float16x8_t vbias = vld1q_f16(bias_c_base);
            float16x8_t vin[18];  // double buffer
            float16x8_t vout[4];

            int64_t oh_inner_start, oh_inner_end;
            int64_t ow_inner_start, ow_inner_end;
            oh_inner_start = 1;  // inclusive index
            ow_inner_start = 1;  // inclusive index
            oh_inner_end = (inH - 2) / 2 + 1;  // exclusive index
            ow_inner_end = (inW - 2) / 2 + 1;  // exclusive index
            oh_inner_end = std::max(oh_inner_end, oh_inner_start);
            ow_inner_end = std::max(ow_inner_end, ow_inner_start);

            int64_t ow_inner_end_align4 = ((ow_inner_end - ow_inner_start) & (~3)) + ow_inner_start;

            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t oh = 0; oh < outH; oh++) {
                const int64_t ih = -1 + oh * 2;
                const __fp16 *input_h_base = input_c_base + ih * inW * ICBLK();
                __fp16 *output_h_base = output_c_base + oh * outW * OCBLK();
                __fp16 *sum_h_base = sum_c_base + oh * outW * OCBLK();

                if (oh == 0 || oh >= oh_inner_end) {
                    bool ih0_valid = (ih >= 0);
                    bool ih2_valid = (ih + 2 < inH);
                    
                    {
                        const __fp16 *input_ptr = input_h_base;
                        bool iw2_valid = (1 < inW);

                        vout[0] = vbias;

                        if (ih0_valid) {
                            vin[1] =               vld1q_f16(input_ptr          )                    ;
                            vin[2] = (iw2_valid) ? vld1q_f16(input_ptr + ICBLK()) : vdupq_n_f16(0.0f);
                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        }

                        vin[7] =               vld1q_f16(input_ptr + inW * ICBLK()          )                    ;
                        vin[8] = (iw2_valid) ? vld1q_f16(input_ptr + inW * ICBLK() + ICBLK()) : vdupq_n_f16(0.0f);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);

                        if (ih2_valid) {
                            vin[13] =               vld1q_f16(input_ptr + inW * ICBLK() * 2          )                    ;
                            vin[14] = (iw2_valid) ? vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK()) : vdupq_n_f16(0.0f);
                            vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                            vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                        }

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }
                        
                        vst1q_f16(output_h_base, vout[0]);
                    }
                    for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow+=4) {
                        const __fp16 *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();
                        __fp16 *output_ptr = output_h_base + ow * OCBLK();

                        vout[0] = vbias;
                        vout[1] = vbias;
                        vout[2] = vbias;
                        vout[3] = vbias;
                        if (ih0_valid) {
                            vin[0] = vld1q_f16(input_ptr              );
                            prefetch_l1(input_ptr, 3 * inW * sizeof(__fp16));
                            vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                            vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                            vin[3] = vld1q_f16(input_ptr + ICBLK() * 3);
                            vin[4] = vld1q_f16(input_ptr + ICBLK() * 4);
                            vin[5] = vld1q_f16(input_ptr + ICBLK() * 5);
                            vin[6] = vld1q_f16(input_ptr + ICBLK() * 6);
                            vin[7] = vld1q_f16(input_ptr + ICBLK() * 7);
                            vin[8] = vld1q_f16(input_ptr + ICBLK() * 8);

                            vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                            vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[0]);
                            vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[0]);
                            vout[3] = vfmaq_f16(vout[3], vin[6] , vflt[0]);

                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                            vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[1]);
                            vout[2] = vfmaq_f16(vout[2], vin[5] , vflt[1]);
                            vout[3] = vfmaq_f16(vout[3], vin[7] , vflt[1]);

                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                            vout[1] = vfmaq_f16(vout[1], vin[4] , vflt[2]);
                            vout[2] = vfmaq_f16(vout[2], vin[6] , vflt[2]);
                            vout[3] = vfmaq_f16(vout[3], vin[8] , vflt[2]);
                        }

                        vin[9]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        prefetch_l1(input_ptr + inW * ICBLK(), 3 * inW * sizeof(__fp16));
                        vin[10] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                        vin[11] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                        vin[12] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 3);
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 4);
                        vin[14] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 5);
                        vin[15] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 6);
                        vin[16] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 7);
                        vin[17] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 8);

                        vout[0] = vfmaq_f16(vout[0], vin[9] , vflt[3]);
                        vout[1] = vfmaq_f16(vout[1], vin[11], vflt[3]);
                        vout[2] = vfmaq_f16(vout[2], vin[13], vflt[3]);
                        vout[3] = vfmaq_f16(vout[3], vin[15], vflt[3]);

                        vout[0] = vfmaq_f16(vout[0], vin[10], vflt[4]);
                        vout[1] = vfmaq_f16(vout[1], vin[12], vflt[4]);
                        vout[2] = vfmaq_f16(vout[2], vin[14], vflt[4]);
                        vout[3] = vfmaq_f16(vout[3], vin[16], vflt[4]);

                        vout[0] = vfmaq_f16(vout[0], vin[11], vflt[5]);
                        vout[1] = vfmaq_f16(vout[1], vin[13], vflt[5]);
                        vout[2] = vfmaq_f16(vout[2], vin[15], vflt[5]);
                        vout[3] = vfmaq_f16(vout[3], vin[17], vflt[5]);

                        if (ih2_valid) {
                            vin[0] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                            prefetch_l1(input_ptr + inW * ICBLK() * 2, 3 * inW * sizeof(__fp16));
                            vin[1] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                            vin[2] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                            vin[3] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 3);
                            vin[4] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 4);
                            vin[5] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 5);
                            vin[6] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 6);
                            vin[7] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 7);
                            vin[8] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 8);

                            vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[6]);
                            vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[6]);
                            vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[6]);
                            vout[3] = vfmaq_f16(vout[3], vin[6] , vflt[6]);

                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[7]);
                            vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[7]);
                            vout[2] = vfmaq_f16(vout[2], vin[5] , vflt[7]);
                            vout[3] = vfmaq_f16(vout[3], vin[7] , vflt[7]);

                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[8]);
                            vout[1] = vfmaq_f16(vout[1], vin[4] , vflt[8]);
                            vout[2] = vfmaq_f16(vout[2], vin[6] , vflt[8]);
                            vout[3] = vfmaq_f16(vout[3], vin[8] , vflt[8]);
                        }

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            __fp16 *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr              ));
                            vout[1] = vaddq_f16(vout[1], vld1q_f16(sum_ptr + OCBLK() * 1));
                            vout[2] = vaddq_f16(vout[2], vld1q_f16(sum_ptr + OCBLK() * 2));
                            vout[3] = vaddq_f16(vout[3], vld1q_f16(sum_ptr + OCBLK() * 3));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            float16x8_t vzero = vdupq_n_f16(0.0f);
                            vout[0] = vmaxq_f16(vout[0], vzero);
                            vout[1] = vmaxq_f16(vout[1], vzero);
                            vout[2] = vmaxq_f16(vout[2], vzero);
                            vout[3] = vmaxq_f16(vout[3], vzero);
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            float16x8_t vsix = vdupq_n_f16(6.0f);
                            vout[0] = vminq_f16(vout[0], vsix);
                            vout[1] = vminq_f16(vout[1], vsix);
                            vout[2] = vminq_f16(vout[2], vsix);
                            vout[3] = vminq_f16(vout[3], vsix);
                        }
                        
                        vst1q_f16(output_ptr              , vout[0]);
                        vst1q_f16(output_ptr + OCBLK() * 1, vout[1]);
                        vst1q_f16(output_ptr + OCBLK() * 2, vout[2]);
                        vst1q_f16(output_ptr + OCBLK() * 3, vout[3]);
                    }
                    for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                        const __fp16 *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();

                        vout[0] = vbias;
                        if (ih0_valid) {
                            vin[0] = vld1q_f16(input_ptr              );
                            vin[1] = vld1q_f16(input_ptr + ICBLK()    );
                            vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                            vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                            vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        }

                        vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK()    );
                        vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);

                        if (ih2_valid) {
                            vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                            vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK()    );
                            vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                            vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                            vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);
                        }

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base + ow * OCBLK()));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }

                        vst1q_f16(output_h_base + ow * OCBLK(), vout[0]);
                    }
                    if (ow_inner_end < outW) { // NOTE: when in_size, k_size and stride are unmatched, the tail is no more than 1.
                        const __fp16 *input_ptr = input_h_base + (ow_inner_end * 2 - 1) * ICBLK();

                        vout[0] = vbias;
                        if (ih0_valid) {
                            vin[0] = vld1q_f16(input_ptr          );
                            vin[1] = vld1q_f16(input_ptr + ICBLK());
                            vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                            vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        }

                        vin[6] = vld1q_f16(input_ptr + inW * ICBLK()          );
                        vin[7] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK());
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);

                        if (ih2_valid) {
                            vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2          );
                            vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK());
                            vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                        }

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base + ow_inner_end * OCBLK()));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }

                        vst1q_f16(output_h_base + ow_inner_end * OCBLK(), vout[0]);
                    }
                }
                else {
                    {
                        const __fp16 *input_ptr = input_h_base;
                        bool iw2_valid = (1 < inW);

                        vout[0] = vbias;

                        vin[1] =               vld1q_f16(input_ptr          )                    ;
                        vin[2] = (iw2_valid) ? vld1q_f16(input_ptr + ICBLK()) : vdupq_n_f16(0.0f);
                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);

                        vin[7] =               vld1q_f16(input_ptr + inW * ICBLK()          )                    ;
                        vin[8] = (iw2_valid) ? vld1q_f16(input_ptr + inW * ICBLK() + ICBLK()) : vdupq_n_f16(0.0f);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);

                        vin[13] =               vld1q_f16(input_ptr + inW * ICBLK() * 2          )                    ;
                        vin[14] = (iw2_valid) ? vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK()) : vdupq_n_f16(0.0f);
                        vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                        vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }
                        
                        vst1q_f16(output_h_base, vout[0]);
                    }
                    for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow+=4) {
                        const __fp16 *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();
                        __fp16 *output_ptr = output_h_base + ow * OCBLK();

                        vout[0] = vbias;
                        vout[1] = vbias;
                        vout[2] = vbias;
                        vout[3] = vbias;

                        vin[0] = vld1q_f16(input_ptr              );
                        prefetch_l1(input_ptr, 3 * inW * sizeof(__fp16));
                        vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);
                        vin[3] = vld1q_f16(input_ptr + ICBLK() * 3);
                        vin[4] = vld1q_f16(input_ptr + ICBLK() * 4);
                        vin[5] = vld1q_f16(input_ptr + ICBLK() * 5);
                        vin[6] = vld1q_f16(input_ptr + ICBLK() * 6);
                        vin[7] = vld1q_f16(input_ptr + ICBLK() * 7);
                        vin[8] = vld1q_f16(input_ptr + ICBLK() * 8);

                        vin[9]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        prefetch_l1(input_ptr + inW * ICBLK(), 3 * inW * sizeof(__fp16));
                        vin[10] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                        vin[11] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);
                        vin[12] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 3);
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 4);
                        vin[14] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 5);
                        vin[15] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 6);
                        vin[16] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 7);
                        vin[17] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 8);

                        vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                        vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[0]);
                        vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[0]);
                        vout[3] = vfmaq_f16(vout[3], vin[6] , vflt[0]);

                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[1]);
                        vout[2] = vfmaq_f16(vout[2], vin[5] , vflt[1]);
                        vout[3] = vfmaq_f16(vout[3], vin[7] , vflt[1]);

                        vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        vout[1] = vfmaq_f16(vout[1], vin[4] , vflt[2]);
                        vout[2] = vfmaq_f16(vout[2], vin[6] , vflt[2]);
                        vout[3] = vfmaq_f16(vout[3], vin[8] , vflt[2]);

                        vin[0] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                        prefetch_l1(input_ptr + inW * ICBLK() * 2, 3 * inW * sizeof(__fp16));
                        vin[1] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                        vin[2] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);
                        vin[3] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 3);
                        vin[4] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 4);
                        vin[5] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 5);
                        vin[6] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 6);
                        vin[7] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 7);
                        vin[8] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 8);

                        vout[0] = vfmaq_f16(vout[0], vin[9] , vflt[3]);
                        vout[1] = vfmaq_f16(vout[1], vin[11], vflt[3]);
                        vout[2] = vfmaq_f16(vout[2], vin[13], vflt[3]);
                        vout[3] = vfmaq_f16(vout[3], vin[15], vflt[3]);

                        vout[0] = vfmaq_f16(vout[0], vin[10], vflt[4]);
                        vout[1] = vfmaq_f16(vout[1], vin[12], vflt[4]);
                        vout[2] = vfmaq_f16(vout[2], vin[14], vflt[4]);
                        vout[3] = vfmaq_f16(vout[3], vin[16], vflt[4]);

                        vout[0] = vfmaq_f16(vout[0], vin[11], vflt[5]);
                        vout[1] = vfmaq_f16(vout[1], vin[13], vflt[5]);
                        vout[2] = vfmaq_f16(vout[2], vin[15], vflt[5]);
                        vout[3] = vfmaq_f16(vout[3], vin[17], vflt[5]);

                        vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[6]);
                        vout[1] = vfmaq_f16(vout[1], vin[2] , vflt[6]);
                        vout[2] = vfmaq_f16(vout[2], vin[4] , vflt[6]);
                        vout[3] = vfmaq_f16(vout[3], vin[6] , vflt[6]);

                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[7]);
                        vout[1] = vfmaq_f16(vout[1], vin[3] , vflt[7]);
                        vout[2] = vfmaq_f16(vout[2], vin[5] , vflt[7]);
                        vout[3] = vfmaq_f16(vout[3], vin[7] , vflt[7]);

                        vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[8]);
                        vout[1] = vfmaq_f16(vout[1], vin[4] , vflt[8]);
                        vout[2] = vfmaq_f16(vout[2], vin[6] , vflt[8]);
                        vout[3] = vfmaq_f16(vout[3], vin[8] , vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            __fp16 *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr              ));
                            vout[1] = vaddq_f16(vout[1], vld1q_f16(sum_ptr + OCBLK() * 1));
                            vout[2] = vaddq_f16(vout[2], vld1q_f16(sum_ptr + OCBLK() * 2));
                            vout[3] = vaddq_f16(vout[3], vld1q_f16(sum_ptr + OCBLK() * 3));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            float16x8_t vzero = vdupq_n_f16(0.0f);
                            vout[0] = vmaxq_f16(vout[0], vzero);
                            vout[1] = vmaxq_f16(vout[1], vzero);
                            vout[2] = vmaxq_f16(vout[2], vzero);
                            vout[3] = vmaxq_f16(vout[3], vzero);
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            float16x8_t vsix = vdupq_n_f16(6.0f);
                            vout[0] = vminq_f16(vout[0], vsix);
                            vout[1] = vminq_f16(vout[1], vsix);
                            vout[2] = vminq_f16(vout[2], vsix);
                            vout[3] = vminq_f16(vout[3], vsix);
                        }

                        vst1q_f16(output_ptr              , vout[0]);
                        vst1q_f16(output_ptr + OCBLK() * 1, vout[1]);
                        vst1q_f16(output_ptr + OCBLK() * 2, vout[2]);
                        vst1q_f16(output_ptr + OCBLK() * 3, vout[3]);
                    }
                    for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                        const __fp16 *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();
                        __fp16 *output_ptr = output_h_base + ow * OCBLK();

                        vout[0] = vbias;

                        vin[0] = vld1q_f16(input_ptr              );
                        vin[1] = vld1q_f16(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f16(input_ptr + ICBLK() * 2);

                        vin[6]  = vld1q_f16(input_ptr + inW * ICBLK()              );
                        vin[7]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 1);
                        vin[8]  = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK() * 2);

                        vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2              );
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 1);
                        vin[14] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK() * 2);

                        vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);
                        vout[0] = vfmaq_f16(vout[0], vin[2] , vflt[2]);
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);
                        vout[0] = vfmaq_f16(vout[0], vin[8] , vflt[5]);
                        vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                        vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);
                        vout[0] = vfmaq_f16(vout[0], vin[14], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            __fp16 *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_ptr));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }

                        vst1q_f16(output_ptr     , vout[0]);
                    }
                    if (ow_inner_end < outW) { // NOTE: when in_size, k_size and stride are unmatched, the tail is no more than 1.
                        const __fp16 *input_ptr = input_h_base + (ow_inner_end * 2 - 1) * ICBLK();

                        vout[0] = vbias;

                        vin[0] = vld1q_f16(input_ptr          );
                        vin[1] = vld1q_f16(input_ptr + ICBLK());
                        vout[0] = vfmaq_f16(vout[0], vin[0] , vflt[0]);
                        vout[0] = vfmaq_f16(vout[0], vin[1] , vflt[1]);

                        vin[6] = vld1q_f16(input_ptr + inW * ICBLK()          );
                        vin[7] = vld1q_f16(input_ptr + inW * ICBLK() + ICBLK());
                        vout[0] = vfmaq_f16(vout[0], vin[6] , vflt[3]);
                        vout[0] = vfmaq_f16(vout[0], vin[7] , vflt[4]);

                        vin[12] = vld1q_f16(input_ptr + inW * ICBLK() * 2          );
                        vin[13] = vld1q_f16(input_ptr + inW * ICBLK() * 2 + ICBLK());
                        vout[0] = vfmaq_f16(vout[0], vin[12], vflt[6]);
                        vout[0] = vfmaq_f16(vout[0], vin[13], vflt[7]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            vout[0] = vaddq_f16(vout[0], vld1q_f16(sum_h_base + ow_inner_end * OCBLK()));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f16(vout[0], vdupq_n_f16(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f16(vout[0], vdupq_n_f16(6.0f));
                        }

                        vst1q_f16(output_h_base + ow_inner_end * OCBLK(), vout[0]);
                    }
                }
            }
        }
    }
}
}


static void conv_n8cx_depthwise_f3sx_convolution(
    const __fp16 *converted_filter,
    const __fp16 *bias,
    const __fp16 *input,
    __fp16 *output,
    __fp16 *sum,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t fltC,
    const int64_t padding,
    const int64_t stride,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
    int64_t case_id = padding * 10 + stride;

    switch (case_id) {
        case 01: // p0s1
            conv_n8cx_depthwise_f3sx_h1w4<0, 1>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                inH,
                inW,
                outH,
                outW,
                num_batch,
                fuse_flag);
            return;

        case 11: // p1s1
            conv_n8cx_depthwise_f3sx_h1w4<1, 1>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                inH,
                inW,
                outH,
                outW,
                num_batch,
                fuse_flag);
            return;

        case 02: // p0s2
            conv_n8cx_depthwise_f3sx_h1w4<0, 2>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                inH,
                inW,
                outH,
                outW,
                num_batch,
                fuse_flag);
            return;

        case 12: // p1s2
            conv_n8cx_depthwise_f3sx_h1w4<1, 2>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                inH,
                inW,
                outH,
                outW,
                num_batch,
                fuse_flag);
            return;

        default:
            return;
    }
}

static void conv_n8cx_depthwise_general_convolution(
    const __fp16 *converted_filter,
    const __fp16 *bias,
    const __fp16 *input,
    __fp16 *output,
    __fp16 *sum,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t fltC,
    const int64_t fltH,
    const int64_t fltW,
    const int64_t padH,
    const int64_t padW,
    const int64_t strdH,
    const int64_t strdW,
    const int64_t dltnH,
    const int64_t dltnW,
    const int64_t num_batch,
    const int64_t fuse_flag)
{
    int64_t ow_inner_start = std::max((int64_t)0, DIV_CEIL((padW - 0 * dltnW), strdW));  // inclusive
    int64_t ow_inner_end   = std::min((int64_t)outW, DIV_CEIL((inW + padW - (fltW-1) * dltnW), strdW));  // exclusive
    ow_inner_start = std::min(ow_inner_start, outW);
    ow_inner_end   = std::max(ow_inner_end, ow_inner_start);

    constexpr int otw = 8;
    int64_t ow_inner_end_align = ((ow_inner_end - ow_inner_start) / otw * otw) + ow_inner_start;

    const int64_t fltC_pck = CEIL8(fltC);
    const int64_t inHW = inH * inW;
    const int64_t outHW = outH * outW;
    const int64_t input_batch_stride = fltC_pck * inHW;
    const int64_t output_batch_stride = fltC_pck * outHW;

PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (int64_t b = 0; b < num_batch; b++) {
        for (int64_t c = 0; c < fltC_pck; c += CBLK()) {
            for (int64_t oh = 0; oh < outH; oh++) {
                const __fp16 *cvt_filter_c_base = converted_filter + c * fltH * fltW;
                float16x8_t vbias = vld1q_f16(bias + c);
                const __fp16 *input_c_base = input + b * input_batch_stride + c * inHW;
                __fp16 *output_c_base = output + b * output_batch_stride + c * outHW;
                __fp16 *sum_c_base = sum + b * output_batch_stride + c * outHW;

                const int64_t ih_base = -padH + oh * strdH;

                const int64_t fltH_start = std::max(-ih_base+dltnH-1, (int64_t)0) / dltnH;  // inclusive
                const int64_t fltH_end = std::min(fltH, (inH-ih_base+dltnH-1) / dltnH);  // exclusive
                if (fltH_end - fltH_start <= 0) continue;
                
                const __fp16 *input_h_base = input_c_base + ih_base * inW * ICBLK();
                __fp16 *output_h_base = output_c_base + oh * outW * OCBLK();
                __fp16 *sum_h_base = sum_c_base + oh * outW * OCBLK();
            
                
                for (int64_t ow = 0; ow < ow_inner_start; ow++) {
                    int64_t iw_base = -padW + ow * strdW;
                    int64_t fltW_start = std::max(-iw_base+dltnW-1, (int64_t)0) / dltnW;  // inclusive
                    int64_t fltW_end = std::min(fltW, (inW-iw_base+dltnW-1) / dltnW);  // exclusive

                    conv_n8cx_depthwise_general_h1w1_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(),
                        vbias, inW, fltW,  ih_base, iw_base, fltH_start, fltH_end, fltW_start, fltW_end, dltnH, dltnW, fuse_flag);
                }
                for (int64_t ow = ow_inner_start; ow < ow_inner_end_align; ow+=otw) {
                    int64_t iw_base = -padW + ow * strdW;

                    conv_n8cx_depthwise_general_h1w8_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(),
                        vbias, fltW, strdW, ih_base, iw_base, fltH_start, fltH_end, dltnH * inW, dltnW, fuse_flag);
                }
                // TODO(kyu): use h1wx function
                for (int64_t ow = ow_inner_end_align; ow < ow_inner_end; ow++) {
                    int64_t iw_base = -padW + ow * strdW;

                    conv_n8cx_depthwise_general_h1w1_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(),
                        vbias, inW, fltW, ih_base, iw_base, fltH_start, fltH_end, 0, fltW, dltnH, dltnW, fuse_flag);
                }
                for (int64_t ow = ow_inner_end; ow < outW; ow++) {
                    int64_t iw_base = -padW + ow * strdW;
                    // TODO(kyu): check if fltW_start is always 0.
                    int64_t fltW_start = std::max(-iw_base+dltnW-1, (int64_t)0) / dltnW;  // inclusive
                    int64_t fltW_end = std::min(fltW, (inW-iw_base+dltnW-1) / dltnW);  // exclusive

                    conv_n8cx_depthwise_general_h1w1_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(),
                        vbias, inW, fltW, ih_base, iw_base, fltH_start, fltH_end, fltW_start, fltW_end, dltnH, dltnW, fuse_flag);
                }
            }
        }
    }
}

uint64_t conv2d_n8cx_depthwise_fp16_runtime_executor::cal_temp_buffer_size() {
    return 0;
}

void conv2d_n8cx_depthwise_fp16_runtime_executor::adjust_schedule_param() {
    return;
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp16_runtime_executor::prepare() {
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp16_runtime_executor::execute() {
    const conv2d_param &cp = *conv_param_;

    const __fp16 *converted_filter = (const __fp16 *)cvt_filter_;
    const __fp16 *bias = (const __fp16 *)cvt_bias_;
    const __fp16 *input = (const __fp16 *)src_;
    __fp16 *output = (__fp16 *)dst_;
    __fp16 *sum = (__fp16 *)sum_;
    const uint32_t fuse_flag = cp.fuse_flag;

    const int64_t inH = src_shape_->GetDim(2);
    const int64_t inW = src_shape_->GetDim(3);
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
    const int64_t num_batch = src_shape_->GetDim(0);

    if (fltH  == 3 && fltW   == 3     &&
        padH  <  2 && padH   == padW  &&
        strdH <  3 && strdH  == strdW &&
        dltnH == 1 && dltnW  == 1        ) {
        conv_n8cx_depthwise_f3sx_convolution(
            converted_filter,
            bias,
            input,
            output,
            sum,
            inH,
            inW,
            outH,
            outW,
            outC,
            padH,
            strdH,
            num_batch,
            fuse_flag);
    }
    else {
        conv_n8cx_depthwise_general_convolution(
            converted_filter,
            bias,
            input,
            output,
            sum,
            inH,
            inW,
            outH,
            outW,
            outC,
            fltH,
            fltW,
            padH,
            padW,
            strdH,
            strdW,
            dltnH,
            dltnW,
            num_batch,
            fuse_flag);
    }
    return ppl::common::RC_SUCCESS;
}


size_t conv_n8cx_depthwise_get_converted_filter_size(
    const int64_t c_out,
    const int64_t h_flt,
    const int64_t w_flt)
{
    return CEIL128(CEIL8(c_out) * h_flt * w_flt * sizeof(__fp16));
}

void conv_n8cx_depthwise_convert_filter(
    const __fp16 *filter,
    __fp16 *converted_filter,
    const int64_t c_out,
    const int64_t h_flt,
    const int64_t w_flt)
{
    const int64_t oc_pck = CEIL8(c_out);
    const int64_t oc_floor8 = FLOOR8(c_out);
    const int64_t hw_flt = h_flt * w_flt;

    for (int64_t c = 0; c < oc_floor8; c+= CBLK()) {
        for (int64_t idx = 0; idx < hw_flt; idx++) {
            for (int64_t c_in = 0; c_in < CBLK(); c_in++) {
                converted_filter[hw_flt * c + idx * CBLK() + c_in] = filter[hw_flt * c + c_in * hw_flt + idx];
            }
        }
    }

    const int64_t oc_tail = oc_pck - oc_floor8;
    if (oc_tail) {
        for (int64_t idx = 0; idx < hw_flt; idx++) {
            for (int64_t c = 0; c < oc_tail; c++) {
                converted_filter[hw_flt * oc_floor8 + idx * CBLK() + c] = filter[hw_flt * oc_floor8 + c * hw_flt + idx];
            }
            for (int64_t c = oc_tail; c < CBLK(); c++) {
                converted_filter[hw_flt * oc_floor8 + idx * CBLK() + c] = 0.0f;
            }
        }
    }
}

bool conv2d_n8cx_depthwise_fp16_offline_manager::is_supported() {
    return (param_.group == param_.channels) && (param_.group == param_.num_output);
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp16_offline_manager::fast_init_schedule_param() {
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp16_offline_manager::pick_best_schedule_param(const ppl::nn::TensorShape &src_shape, double &run_time, bool tune_blocksize) {
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp16_offline_manager::gen_cvt_weights(const void *filter, const void *bias) {
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_output = param_.num_output;
    const int64_t channels = param_.channels;
    const int64_t kernel_h = param_.kernel_h;
    const int64_t kernel_w = param_.kernel_w;
    const int64_t num_group = param_.group;
    
    cvt_bias_size_ = CEIL8(num_output) * sizeof(__fp16);
    cvt_bias_      = allocator_->Alloc(cvt_bias_size_);
    int64_t padding_offset_bytes = num_output * sizeof(__fp16);
    int64_t padding_bytes        = (CEIL8(num_output) - num_output) * sizeof(__fp16);
    memcpy(cvt_bias_, bias, num_output * sizeof(__fp16));
    memset(cvt_bias_ + padding_offset_bytes, 0, padding_bytes);
    
    cvt_filter_size_ = conv_n8cx_depthwise_get_converted_filter_size(num_output, kernel_h, kernel_w);
    cvt_filter_      = allocator_->Alloc(cvt_filter_size_);
    conv_n8cx_depthwise_convert_filter(
        (const __fp16 *)filter, (__fp16 *)cvt_filter_, num_output, kernel_h, kernel_w);
    
    return ppl::common::RC_SUCCESS;
}

conv2d_runtime_executor *conv2d_n8cx_depthwise_fp16_offline_manager::gen_executor() {
    return new conv2d_n8cx_depthwise_fp16_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}

}}}