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

#include "ppl/kernel/arm_server/maxpool/neon/maxpool.h"

#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/math.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

#define CVL() 4

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

static void maxpool2d_n4cx_general_fp32(
    const float *input,
    float *output,
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
        const int64_t num_channel_ceil4 = CEIL4(num_channel);
        const float32x4_t vmin          = vdupq_n_f32(-numeric_max<float>());
        for (int64_t n = 0; n < num_batch; n++) {
            const float *input_b_base = input + n * num_channel_ceil4 * inH * inW;
            float *output_b_base      = output + n * num_channel_ceil4 * outH * outW;
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel_ceil4; c += CVL()) {
                const float *input_c_base = input_b_base + c * inH * inW;
                float *output_c_base      = output_b_base + c * outH * outW;
                for (int64_t oh = 0; oh < outH; oh++) {
                    const int64_t ih_base     = -padH + oh * strdH;
                    const float *input_h_base = input_c_base + ih_base * inW * CVL();
                    const int64_t fltH_start  = std::max(-ih_base + dltnH - 1, (int64_t)0) / dltnH; // inclusive
                    const int64_t fltH_end    = std::min(kerH, (inH - ih_base + dltnH - 1) / dltnH); // exclusive

                    for (int64_t ow = 0; ow < outW; ow++) {
                        int64_t iw_base           = -padW + ow * strdW;
                        const float *input_w_base = input_h_base + iw_base * CVL();
                        int64_t fltW_start        = std::max(-iw_base + dltnW - 1, (int64_t)0) / dltnW; // inclusive
                        int64_t fltW_end          = std::min(kerW, (inW - iw_base + dltnW - 1) / dltnW); // exclusive

                        float32x4_t vout = vmin;

                        for (int64_t kh = fltH_start; kh < fltH_end; kh++) {
                            for (int64_t kw = fltW_start; kw < fltW_end; kw++) {
                                float32x4_t vin = vld1q_f32(input_w_base + (kh * dltnH * inW + kw * dltnW) * CVL());
                                vout            = vmaxq_f32(vin, vout);
                            }
                        }

                        vst1q_f32(output_c_base + oh * outW * CVL() + ow * CVL(), vout);
                    }
                }
            }
        }
    }
}

static void maxpool2d_n4cx_global_fp32(
    const float *input,
    float *output,
    const int64_t inH,
    const int64_t inW,
    const int64_t num_channel,
    const int64_t num_batch)
{
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t num_channel_ceil4 = CEIL4(num_channel);
        const float32x4_t vmin          = vdupq_n_f32(-numeric_max<float>());
        for (int64_t n = 0; n < num_batch; n++) {
            const float *input_b_base = input + n * num_channel_ceil4 * inH * inW;
            float *output_b_base      = output + n * num_channel_ceil4;
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel_ceil4; c += CVL()) {
                const float *input_c_base = input_b_base + c * inH * inW;

                float32x4_t vout = vmin;
                for (int64_t idx = 0; idx < inH * inW; idx++) {
                    float32x4_t vin = vld1q_f32(input_c_base + idx * CVL());
                    vout            = vmaxq_f32(vin, vout);
                }

                vst1q_f32(output_b_base + c, vout);
            }
        }
    }
}

static void maxpool2d_n4cx_f2s2p0_fp32(
    const float *input,
    float *output,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_channel,
    const int64_t num_batch)
{
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t num_channel_ceil4 = CEIL4(num_channel);
        for (int64_t n = 0; n < num_batch; n++) {
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel; c += CVL()) {
                const float *input_c_base = input + (n * num_channel_ceil4 + c) * inH * inW;
                float *output_c_base      = output + (n * num_channel_ceil4 + c) * outH * outW;
                for (int64_t oh = 0; oh < outH; oh++) {
                    const float *input_h_base = input_c_base + oh * 2 * inW * CVL();
                    float *output_h_base      = output_c_base + oh * outW * CVL();

                    int64_t ow = 0;
                    for (; ow <= outW - 4; ow += 4) {
                        const float *input_base0 = input_h_base + ow * 2 * CVL();
                        const float *input_base1 = input_h_base + inW * CVL() + ow * 2 * CVL();
                        float *output_base       = output_h_base + ow * CVL();
                        float32x4_t vin0[8];
                        float32x4_t vin1[8];
                        float32x4_t vout[4];

                        vin0[0] = vld1q_f32(input_base0);
                        vin0[1] = vld1q_f32(input_base0 + CVL());
                        vin0[2] = vld1q_f32(input_base0 + CVL() * 2);
                        vin0[3] = vld1q_f32(input_base0 + CVL() * 3);
                        vin0[4] = vld1q_f32(input_base0 + CVL() * 4);
                        vin0[5] = vld1q_f32(input_base0 + CVL() * 5);
                        vin0[6] = vld1q_f32(input_base0 + CVL() * 6);
                        vin0[7] = vld1q_f32(input_base0 + CVL() * 7);

                        vin1[0] = vld1q_f32(input_base1);
                        vin1[1] = vld1q_f32(input_base1 + CVL());
                        vin1[2] = vld1q_f32(input_base1 + CVL() * 2);
                        vin1[3] = vld1q_f32(input_base1 + CVL() * 3);
                        vin1[4] = vld1q_f32(input_base1 + CVL() * 4);
                        vin1[5] = vld1q_f32(input_base1 + CVL() * 5);
                        vin1[6] = vld1q_f32(input_base1 + CVL() * 6);
                        vin1[7] = vld1q_f32(input_base1 + CVL() * 7);

                        vout[0] = vmaxq_f32(vin0[0], vin0[1]);
                        vout[1] = vmaxq_f32(vin0[2], vin0[3]);
                        vout[2] = vmaxq_f32(vin0[4], vin0[5]);
                        vout[3] = vmaxq_f32(vin0[6], vin0[7]);

                        vout[0] = vmaxq_f32(vout[0], vin1[0]);
                        vout[1] = vmaxq_f32(vout[1], vin1[2]);
                        vout[2] = vmaxq_f32(vout[2], vin1[4]);
                        vout[3] = vmaxq_f32(vout[3], vin1[6]);

                        vout[0] = vmaxq_f32(vout[0], vin1[1]);
                        vout[1] = vmaxq_f32(vout[1], vin1[3]);
                        vout[2] = vmaxq_f32(vout[2], vin1[5]);
                        vout[3] = vmaxq_f32(vout[3], vin1[7]);

                        vst1q_f32(output_base, vout[0]);
                        vst1q_f32(output_base + CVL(), vout[1]);
                        vst1q_f32(output_base + CVL() * 2, vout[2]);
                        vst1q_f32(output_base + CVL() * 3, vout[3]);
                    }
                    for (; ow < outW; ow++) {
                        const float *input_base = input_h_base + ow * 2 * CVL();
                        float32x4_t vin[4];
                        float32x4_t vout;

                        vin[0] = vld1q_f32(input_base);
                        vin[1] = vld1q_f32(input_base + CVL());
                        vin[2] = vld1q_f32(input_base + inW * CVL());
                        vin[3] = vld1q_f32(input_base + inW * CVL() + CVL());

                        vout = vmaxq_f32(vin[0], vin[1]);
                        vout = vmaxq_f32(vout, vin[2]);
                        vout = vmaxq_f32(vout, vin[3]);

                        vst1q_f32(output_h_base + ow * CVL(), vout);
                    }
                }
            }
        }
    }
}

static void maxpool2d_n4cx_f3s2_fp32(
    const float *input,
    float *output,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_channel,
    const int64_t num_batch,
    const int64_t padH,
    const int64_t padW)
{
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t num_channel_pck = CEIL4(num_channel);
        int64_t ow_no_padding_start   = (padW + 1) / 2;
        int64_t ow_no_padding_end     = (inW - 3 + padW) / 2 + 1;
        ow_no_padding_end             = std::max(ow_no_padding_end, ow_no_padding_start);
        ow_no_padding_end             = ow_no_padding_start + ((ow_no_padding_end - ow_no_padding_start) & (~7));

        const float32x4_t vmin = vdupq_n_f32(-numeric_max<float>());
        for (int64_t n = 0; n < num_batch; n++) {
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel; c += CVL()) {
                const float *input_c_base = input + (n * num_channel_pck + c) * inH * inW;
                float *output_c_base      = output + (n * num_channel_pck + c) * outH * outW;
                for (int64_t oh = 0; oh < outH; oh++) {
                    float *output_h_base = output_c_base + oh * outW * CVL();
                    int64_t ih_start     = oh * 2 - padH;
                    int64_t ih_end       = std::min(ih_start + 3, inH);
                    ih_start             = std::max(ih_start, (int64_t)0);

                    if (ih_end - ih_start != 3) {
                        for (int64_t ow = 0; ow < outW; ow++) {
                            int64_t iw_start = ow * 2 - padW;
                            int64_t iw_end   = std::min(iw_start + 3, inW);
                            iw_start         = std::max(iw_start, (int64_t)0);

                            float32x4_t vout = vmin;
                            for (int ih = ih_start; ih < ih_end; ih++) {
                                const float *input_h_base = input_c_base + ih * inW * CVL();
                                for (int iw = iw_start; iw < iw_end; iw++) {
                                    float32x4_t vin = vld1q_f32(input_h_base + iw * CVL());
                                    vout            = vmaxq_f32(vout, vin);
                                }
                            }
                            vst1q_f32(output_h_base + ow * CVL(), vout);
                        }
                    } else {
                        const float *input_h_base = input_c_base + ih_start * inW * CVL();

                        int64_t ow = 0;
                        for (; ow < ow_no_padding_start; ow++) {
                            int64_t iw_start = ow * 2 - padW;
                            int64_t iw_end   = std::min(iw_start + 3, inW);
                            iw_start         = std::max(iw_start, (int64_t)0);

                            float32x4_t vout = vmin;
                            for (int ih = ih_start; ih < ih_end; ih++) {
                                const float *input_h_base = input_c_base + ih * inW * CVL();
                                for (int iw = iw_start; iw < iw_end; iw++) {
                                    float32x4_t vin = vld1q_f32(input_h_base + iw * CVL());
                                    vout            = vmaxq_f32(vout, vin);
                                }
                            }
                            vst1q_f32(output_h_base + ow * CVL(), vout);
                        }
                        for (; ow < ow_no_padding_end; ow += 4) {
                            int64_t iw_start = ow * 2 - padW;

                            const float *input_base0 = input_h_base + iw_start * CVL();

                            const float *input_base1 = input_base0 + inW * CVL();
                            const float *input_base2 = input_base0 + inW * CVL() * 2;

                            float *output_base = output_h_base + ow * CVL();

                            float32x4_t vin[27];
                            vin[0] = vld1q_f32(input_base0);
                            vin[1] = vld1q_f32(input_base0 + CVL());
                            vin[2] = vld1q_f32(input_base0 + CVL() * 2);
                            vin[3] = vld1q_f32(input_base0 + CVL() * 3);
                            vin[4] = vld1q_f32(input_base0 + CVL() * 4);
                            vin[5] = vld1q_f32(input_base0 + CVL() * 5);
                            vin[6] = vld1q_f32(input_base0 + CVL() * 6);
                            vin[7] = vld1q_f32(input_base0 + CVL() * 7);
                            vin[8] = vld1q_f32(input_base0 + CVL() * 8);

                            vin[9]  = vld1q_f32(input_base1);
                            vin[10] = vld1q_f32(input_base1 + CVL());
                            vin[11] = vld1q_f32(input_base1 + CVL() * 2);
                            vin[12] = vld1q_f32(input_base1 + CVL() * 3);
                            vin[13] = vld1q_f32(input_base1 + CVL() * 4);
                            vin[14] = vld1q_f32(input_base1 + CVL() * 5);
                            vin[15] = vld1q_f32(input_base1 + CVL() * 6);
                            vin[16] = vld1q_f32(input_base1 + CVL() * 7);
                            vin[17] = vld1q_f32(input_base1 + CVL() * 8);

                            vin[18] = vld1q_f32(input_base2);
                            vin[19] = vld1q_f32(input_base2 + CVL());
                            vin[20] = vld1q_f32(input_base2 + CVL() * 2);
                            vin[21] = vld1q_f32(input_base2 + CVL() * 3);
                            vin[22] = vld1q_f32(input_base2 + CVL() * 4);
                            vin[23] = vld1q_f32(input_base2 + CVL() * 5);
                            vin[24] = vld1q_f32(input_base2 + CVL() * 6);
                            vin[25] = vld1q_f32(input_base2 + CVL() * 7);
                            vin[26] = vld1q_f32(input_base2 + CVL() * 8);

                            float32x4_t vout[4];
                            vout[0] = vmaxq_f32(vin[0], vin[1]);
                            vout[1] = vmaxq_f32(vin[2], vin[3]);
                            vout[2] = vmaxq_f32(vin[4], vin[5]);
                            vout[3] = vmaxq_f32(vin[6], vin[7]);

                            vout[0] = vmaxq_f32(vout[0], vin[2]);
                            vout[1] = vmaxq_f32(vout[1], vin[4]);
                            vout[2] = vmaxq_f32(vout[2], vin[6]);
                            vout[3] = vmaxq_f32(vout[3], vin[8]);

                            vout[0] = vmaxq_f32(vout[0], vin[9]);
                            vout[1] = vmaxq_f32(vout[1], vin[11]);
                            vout[2] = vmaxq_f32(vout[2], vin[13]);
                            vout[3] = vmaxq_f32(vout[3], vin[15]);

                            vout[0] = vmaxq_f32(vout[0], vin[10]);
                            vout[1] = vmaxq_f32(vout[1], vin[12]);
                            vout[2] = vmaxq_f32(vout[2], vin[14]);
                            vout[3] = vmaxq_f32(vout[3], vin[16]);

                            vout[0] = vmaxq_f32(vout[0], vin[11]);
                            vout[1] = vmaxq_f32(vout[1], vin[13]);
                            vout[2] = vmaxq_f32(vout[2], vin[15]);
                            vout[3] = vmaxq_f32(vout[3], vin[17]);

                            vout[0] = vmaxq_f32(vout[0], vin[18]);
                            vout[1] = vmaxq_f32(vout[1], vin[20]);
                            vout[2] = vmaxq_f32(vout[2], vin[22]);
                            vout[3] = vmaxq_f32(vout[3], vin[24]);

                            vout[0] = vmaxq_f32(vout[0], vin[19]);
                            vout[1] = vmaxq_f32(vout[1], vin[21]);
                            vout[2] = vmaxq_f32(vout[2], vin[23]);
                            vout[3] = vmaxq_f32(vout[3], vin[25]);

                            vout[0] = vmaxq_f32(vout[0], vin[20]);
                            vout[1] = vmaxq_f32(vout[1], vin[22]);
                            vout[2] = vmaxq_f32(vout[2], vin[24]);
                            vout[3] = vmaxq_f32(vout[3], vin[26]);

                            vst1q_f32(output_base, vout[0]);
                            vst1q_f32(output_base + CVL(), vout[1]);
                            vst1q_f32(output_base + CVL() * 2, vout[2]);
                            vst1q_f32(output_base + CVL() * 3, vout[3]);
                        }
                        for (; ow < outW; ow++) {
                            int64_t iw_start = ow * 2 - padW;
                            int64_t iw_end   = std::min(iw_start + 3, inW);
                            iw_start         = std::max(iw_start, (int64_t)0);

                            float32x4_t vout = vmin;
                            for (int ih = ih_start; ih < ih_end; ih++) {
                                const float *input_h_base = input_c_base + ih * inW * CVL();
                                for (int iw = iw_start; iw < iw_end; iw++) {
                                    float32x4_t vin = vld1q_f32(input_h_base + iw * CVL());
                                    vout            = vmaxq_f32(vout, vin);
                                }
                            }
                            vst1q_f32(output_h_base + ow * CVL(), vout);
                        }
                    }
                }
            }
        }
    }
}

static void maxpool2d_n4cx_f3s1_fp32(
    const float *input,
    float *output,
    const int64_t inH,
    const int64_t inW,
    const int64_t outH,
    const int64_t outW,
    const int64_t num_channel,
    const int64_t num_batch,
    const int64_t padH,
    const int64_t padW)
{
    (void)maxpool2d_n4cx_f3s1_fp32;
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t num_channel_ceil4 = CEIL4(num_channel);
        int64_t ow_no_padding_start     = padW;
        int64_t ow_no_padding_end       = padW + ((inW - 3) & (~7));
        // ow_no_padding_end = ow_no_padding_start + ((ow_no_padding_end - ow_no_padding_start) & (~7));

        const float32x4_t vmin = vdupq_n_f32(-numeric_max<float>());
        for (int64_t n = 0; n < num_batch; n++) {
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel; c += CVL()) {
                const float *input_c_base = input + (n * num_channel_ceil4 + c) * inH * inW;
                float *output_c_base      = output + (n * num_channel_ceil4 + c) * outH * outW;
                for (int64_t oh = 0; oh < outH; oh++) {
                    float *output_h_base = output_c_base + oh * outW * CVL();
                    int64_t ih_start     = oh - padH;
                    int64_t ih_end       = std::min(ih_start + 3, inH);
                    ih_start             = std::max(ih_start, (int64_t)0);

                    if (ih_end - ih_start != 3) {
                        for (int64_t ow = 0; ow < outW; ow++) {
                            int64_t iw_start = ow - padW;
                            int64_t iw_end   = std::min(iw_start + 3, inW);
                            iw_start         = std::max(ih_start, (int64_t)0);

                            float32x4_t vout = vmin;
                            for (int ih = ih_start; ih < ih_end; ih++) {
                                const float *input_h_base = input_c_base + ih * inW * CVL();
                                for (int iw = iw_start; iw < iw_end; iw++) {
                                    float32x4_t vin = vld1q_f32(input_h_base + iw * CVL());
                                    vout            = vmaxq_f32(vout, vin);
                                }
                            }
                            vst1q_f32(output_h_base + ow * CVL(), vout);
                        }
                    } else {
                        const float *input_h_base = input_c_base + ih_start * inW * CVL();

                        int64_t ow = 0;
                        for (; ow < ow_no_padding_start; ow++) {
                            int64_t iw_start = ow - padW;
                            int64_t iw_end   = std::min(iw_start + 3, inW);
                            iw_start         = std::max(ih_start, (int64_t)0);

                            float32x4_t vout = vmin;
                            for (int ih = ih_start; ih < ih_end; ih++) {
                                const float *input_h_base = input_c_base + ih * inW * CVL();
                                for (int iw = iw_start; iw < iw_end; iw++) {
                                    float32x4_t vin = vld1q_f32(input_h_base + iw * CVL());
                                    vout            = vmaxq_f32(vout, vin);
                                }
                            }
                            vst1q_f32(output_h_base + ow * CVL(), vout);
                        }
                        for (; ow < ow_no_padding_end; ow += 4) {
                            int64_t iw_start = ow - padH;

                            const float *input_base0 = input_h_base + iw_start * CVL();

                            const float *input_base1 = input_base0 + inW * CVL();
                            const float *input_base2 = input_base0 + inW * CVL() * 2;

                            float *output_base = output_h_base + ow * CVL();

                            float32x4_t vin[18];
                            vin[0] = vld1q_f32(input_base0);
                            vin[1] = vld1q_f32(input_base0 + CVL());
                            vin[2] = vld1q_f32(input_base0 + CVL() * 2);
                            vin[3] = vld1q_f32(input_base0 + CVL() * 3);
                            vin[4] = vld1q_f32(input_base0 + CVL() * 4);
                            vin[5] = vld1q_f32(input_base0 + CVL() * 5);

                            vin[6]  = vld1q_f32(input_base1);
                            vin[7]  = vld1q_f32(input_base1 + CVL());
                            vin[8]  = vld1q_f32(input_base1 + CVL() * 2);
                            vin[9]  = vld1q_f32(input_base1 + CVL() * 3);
                            vin[10] = vld1q_f32(input_base1 + CVL() * 4);
                            vin[11] = vld1q_f32(input_base1 + CVL() * 5);

                            vin[12] = vld1q_f32(input_base2);
                            vin[13] = vld1q_f32(input_base2 + CVL());
                            vin[14] = vld1q_f32(input_base2 + CVL() * 2);
                            vin[15] = vld1q_f32(input_base2 + CVL() * 3);
                            vin[16] = vld1q_f32(input_base2 + CVL() * 4);
                            vin[17] = vld1q_f32(input_base2 + CVL() * 5);

                            float32x4_t vout[4];
                            vout[0] = vmaxq_f32(vin[0], vin[1]);
                            vout[1] = vmaxq_f32(vin[1], vin[2]);
                            vout[2] = vmaxq_f32(vin[2], vin[3]);
                            vout[3] = vmaxq_f32(vin[3], vin[4]);

                            vout[0] = vmaxq_f32(vout[0], vin[2]);
                            vout[1] = vmaxq_f32(vout[1], vin[3]);
                            vout[2] = vmaxq_f32(vout[2], vin[4]);
                            vout[3] = vmaxq_f32(vout[3], vin[5]);

                            vout[0] = vmaxq_f32(vout[0], vin[6]);
                            vout[1] = vmaxq_f32(vout[1], vin[7]);
                            vout[2] = vmaxq_f32(vout[2], vin[8]);
                            vout[3] = vmaxq_f32(vout[3], vin[9]);

                            vout[0] = vmaxq_f32(vout[0], vin[7]);
                            vout[1] = vmaxq_f32(vout[1], vin[8]);
                            vout[2] = vmaxq_f32(vout[2], vin[9]);
                            vout[3] = vmaxq_f32(vout[3], vin[10]);

                            vout[0] = vmaxq_f32(vout[0], vin[8]);
                            vout[1] = vmaxq_f32(vout[1], vin[9]);
                            vout[2] = vmaxq_f32(vout[2], vin[10]);
                            vout[3] = vmaxq_f32(vout[3], vin[11]);

                            vout[0] = vmaxq_f32(vout[0], vin[12]);
                            vout[1] = vmaxq_f32(vout[1], vin[13]);
                            vout[2] = vmaxq_f32(vout[2], vin[14]);
                            vout[3] = vmaxq_f32(vout[3], vin[15]);

                            vout[0] = vmaxq_f32(vout[0], vin[14]);
                            vout[1] = vmaxq_f32(vout[1], vin[15]);
                            vout[2] = vmaxq_f32(vout[2], vin[16]);
                            vout[3] = vmaxq_f32(vout[3], vin[17]);

                            vout[0] = vmaxq_f32(vout[0], vin[15]);
                            vout[1] = vmaxq_f32(vout[1], vin[16]);
                            vout[2] = vmaxq_f32(vout[2], vin[17]);
                            vout[3] = vmaxq_f32(vout[3], vin[18]);

                            vst1q_f32(output_base, vout[0]);
                            vst1q_f32(output_base + CVL(), vout[1]);
                            vst1q_f32(output_base + CVL() * 2, vout[2]);
                            vst1q_f32(output_base + CVL() * 3, vout[3]);
                        }
                        for (; ow < outW; ow++) {
                            int64_t iw_start = ow - padW;
                            int64_t iw_end   = std::min(iw_start + 3, inW);
                            iw_start         = std::max(ih_start, (int64_t)0);

                            float32x4_t vout = vmin;
                            for (int ih = ih_start; ih < ih_end; ih++) {
                                const float *input_h_base = input_c_base + ih * inW * CVL();
                                for (int iw = iw_start; iw < iw_end; iw++) {
                                    float32x4_t vin = vld1q_f32(input_h_base + iw * CVL());
                                    vout            = vmaxq_f32(vout, vin);
                                }
                            }
                            vst1q_f32(output_h_base + ow * CVL(), vout);
                        }
                    }
                }
            }
        }
    }
}

ppl::common::RetCode maxpool2d_n4cx_fp32(
    const float *src,
    float *dst,
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
    const int32_t global_pooling)
{
    if (global_pooling) {
        maxpool2d_n4cx_global_fp32(src, dst, src_h, src_w, src_c, src_n);
    } else if (kernel_h == 2 && kernel_w == 2 && stride_h == 2 && stride_w == 2 && pad_h == 0 && pad_w == 0 && dilation_h == 1 && dilation_w == 1) {
        maxpool2d_n4cx_f2s2p0_fp32(src, dst, src_h, src_w, dst_h, dst_w, src_c, src_n);
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
        maxpool2d_n4cx_f3s2_fp32(src, dst, src_h, src_w, dst_h, dst_w, src_c, src_n, pad_h, pad_w);
    } else {
        maxpool2d_n4cx_general_fp32(src, dst, src_h, src_w, dst_h, dst_w, src_c, src_n, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    }
    return ppl::common::RC_SUCCESS;
}

}}}}; // namespace ppl::kernel::arm_server::neon
