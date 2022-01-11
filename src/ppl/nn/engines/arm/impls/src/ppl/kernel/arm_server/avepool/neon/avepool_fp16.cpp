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

#include "ppl/kernel/arm_server/avepool/neon/avepool.h" 

#include <arm_neon.h>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <iostream>
#if defined PPL_USE_ARM_SERVER_OMP
#include <omp.h>
#endif
#include <string.h>

#include "ppl/common/arm/sysinfo.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

#define CVL() 8

namespace ppl { namespace kernel { namespace arm_server {

static void ppl_arm_server_kernel_fp16_pooling_ave_exclude_general(
    const __fp16 *input,
    __fp16 *output,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_channel,
    const int64_t num_batch,
    const int64_t kerH,
    const int64_t kerW,
    const int64_t strdH,
    const int64_t strdW,
    const int64_t padH,
    const int64_t padW,
    const int64_t dltnH,
    const int64_t dltnW)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t num_channel_ceil8 = CEIL8(num_channel);
    const float16x8_t vzero = vdupq_n_f16(0.0f);
    for (int64_t n = 0; n < num_batch; n++) {
        PRAGMA_OMP_FOR_NOWAIT()
        for (int64_t c = 0; c < num_channel_ceil8; c += CVL()) {
        const __fp16 *input_b_base = input + n * num_channel_ceil8 * inH * inW;
        __fp16 *output_b_base = output + n * num_channel_ceil8 * outH * outW;
            const __fp16 *input_c_base = input_b_base + c * inH * inW;
            __fp16 *output_c_base = output_b_base + c * outH * outW;
            for (int64_t oh = 0; oh < outH; oh++) {
                const int64_t ih_base = -padH + oh * strdH;
                const __fp16 *input_h_base = input_c_base + ih_base * inW * CVL();
                const int64_t fltH_start = std::max(-ih_base+dltnH-1, (int64_t)0) / dltnH;  // inclusive
                const int64_t fltH_end = std::min(kerH, (inH-ih_base+dltnH-1) / dltnH);  // exclusive

                for (int64_t ow = 0; ow < outW; ow++) {
                    
                    int64_t iw_base = -padW + ow * strdW;
                    const __fp16 *input_w_base = input_h_base + iw_base * CVL();
                    int64_t fltW_start = std::max(-iw_base+dltnW-1, (int64_t)0) / dltnW;  // inclusive
                    int64_t fltW_end = std::min(kerW, (inW-iw_base+dltnW-1) / dltnW);  // exclusive
                    
                    float16x8_t vout = vzero;
                    __fp16 ave_coeff = 1.0f / ((float)(fltH_end - fltH_start) * (fltW_end - fltW_start));
                    float16x8_t vave_coeff = vdupq_n_f16(ave_coeff);

                    for (int64_t kh = fltH_start; kh < fltH_end; kh++) {
                        for (int64_t kw = fltW_start; kw < fltW_end; kw++) {
                            float16x8_t vin = vld1q_f16(input_w_base + (kh * dltnH * inW + kw * dltnW) * CVL());
                            vout = vaddq_f16(vout, vin);
                        }
                    }
                    vout = vmulq_f16(vout, vave_coeff);

                    vst1q_f16(output_c_base + oh * outW * CVL() + ow * CVL(), vout);
                }
            }
        }
    }
}
}

static void ppl_arm_server_kernel_fp16_pooling_ave_include_general(
    const __fp16 *input,
    __fp16 *output,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_channel,
    const int64_t num_batch,
    const int64_t kerH,
    const int64_t kerW,
    const int64_t strdH,
    const int64_t strdW,
    const int64_t padH,
    const int64_t padW,
    const int64_t dltnH,
    const int64_t dltnW)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t num_channel_ceil8 = CEIL8(num_channel);
    const float16x8_t vzero = vdupq_n_f16(0.0f);
    const __fp16 kernel_size_recp = 1.0f / ((float)kerH * (float)kerW);
    const float16x8_t vave_coeff = vdupq_n_f16(kernel_size_recp);
    for (int64_t n = 0; n < num_batch; n++) {
        PRAGMA_OMP_FOR_NOWAIT()
        for (int64_t c = 0; c < num_channel_ceil8; c += CVL()) {
        const __fp16 *input_b_base = input + n * num_channel_ceil8 * inH *inW;
        __fp16 *output_b_base = output + n * num_channel_ceil8 * outH * outW;
            const __fp16 *input_c_base = input_b_base + c * inH * inW;
            __fp16 *output_c_base = output_b_base + c * outH * outW;
            for (int64_t oh = 0; oh < outH; oh++) {
                const int64_t ih_base = -padH + oh * strdH;
                const __fp16 *input_h_base = input_c_base + ih_base * inW * CVL();
                const int64_t fltH_start = std::max(-ih_base+dltnH-1, (int64_t)0) / dltnH;  // inclusive
                const int64_t fltH_end = std::min(kerH, (inH-ih_base+dltnH-1) / dltnH);  // exclusive

                for (int64_t ow = 0; ow < outW; ow++) {
                    
                    int64_t iw_base = -padW + ow * strdW;
                    const __fp16 *input_w_base = input_h_base + iw_base * CVL();
                    int64_t fltW_start = std::max(-iw_base+dltnW-1, (int64_t)0) / dltnW;  // inclusive
                    int64_t fltW_end = std::min(kerW, (inW-iw_base+dltnW-1) / dltnW);  // exclusive
                    
                    float16x8_t vout = vzero;

                    for (int64_t kh = fltH_start; kh < fltH_end; kh++) {
                        for (int64_t kw = fltW_start; kw < fltW_end; kw++) {
                            float16x8_t vin = vld1q_f16(input_w_base + (kh * dltnH * inW + kw * dltnW) * CVL());
                            vout = vaddq_f16(vout, vin);
                        }
                    }
                    vout = vmulq_f16(vout, vave_coeff);

                    vst1q_f16(output_c_base + oh * outW * CVL() + ow * CVL(), vout);
                }
            }
        }
    }
}
}



static void ppl_arm_server_kernel_fp16_pooling_ave_global(
    const __fp16 *input,
    __fp16 *output,
    const int64_t inH,
    const int64_t inW,
    const int64_t num_channel,
    const int64_t num_batch)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t num_channel_ceil8 = CEIL8(num_channel);
    const float16x8_t vzero = vdupq_n_f16(0.0f);
    const __fp16 in_size_recp = 1.0f / ((float)inH * (float)inW);
    float16x8_t vave_coeff = vdupq_n_f16(in_size_recp);

    for (int64_t n = 0; n < num_batch; n++) {
        const __fp16 *input_b_base = input + n * num_channel_ceil8 * inH *inW;
        __fp16 *output_b_base = output + n * num_channel_ceil8;
        PRAGMA_OMP_FOR_NOWAIT()
        for (int64_t c = 0; c < num_channel_ceil8; c += CVL()) {
            const __fp16 *input_c_base = input_b_base + c * inH * inW;
                    
            float16x8_t vout = vzero;
            for (int64_t idx = 0; idx < inH*inW; idx++) {
                float16x8_t vin = vld1q_f16(input_c_base + idx * CVL());
                vout = vaddq_f16(vout, vin);
            }
            vout = vmulq_f16(vout, vave_coeff);

            vst1q_f16(output_b_base + c, vout);
        }
    }
}
}

ppl::common::RetCode avepool_n8cx_fp16(
    const __fp16 *src,
    __fp16 *dst,
    const int32_t src_n,
    const int32_t src_c,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t dilation_h,
    const int32_t dilation_w,
    const int32_t global_pooling,
    const bool count_include_pad) {

    if (global_pooling) {
        ppl_arm_server_kernel_fp16_pooling_ave_global(src, dst, src_h, src_w, src_c, src_n);
    }
    else if (count_include_pad) {
        ppl_arm_server_kernel_fp16_pooling_ave_include_general(src, dst, src_h, src_w, dst_h, dst_w, src_c, src_n, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    }
    else {
        ppl_arm_server_kernel_fp16_pooling_ave_exclude_general(src, dst, src_h, src_w, dst_h, dst_w, src_c, src_n, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    }

    return ppl::common::RC_SUCCESS;
}


}}}; // namespace ppl::kernel::arm_server

