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

#include "n8cx_hgemm.h"

#include <algorithm>
#include <arm_neon.h>
#include <iostream>

#define CEIL8(val) (((val) + 7) & (~7))

#define V_TRANSPOSE_FP16_8x8(v)                                                                               \
    do {                                                                                                      \
        float16x8x2_t vpf16[4];                                                                               \
        float32x4x2_t vpf32[4];                                                                               \
        vpf16[0] = vtrnq_f16(v[0], v[1]);                                                                     \
        vpf16[1] = vtrnq_f16(v[2], v[3]);                                                                     \
        vpf16[2] = vtrnq_f16(v[4], v[5]);                                                                     \
        vpf16[3] = vtrnq_f16(v[6], v[7]);                                                                     \
        vpf32[0] = vtrnq_f32((float32x4_t)vpf16[0].val[0], (float32x4_t)vpf16[1].val[0]);                     \
        vpf32[1] = vtrnq_f32((float32x4_t)vpf16[0].val[1], (float32x4_t)vpf16[1].val[1]);                     \
        vpf32[2] = vtrnq_f32((float32x4_t)vpf16[2].val[0], (float32x4_t)vpf16[3].val[0]);                     \
        vpf32[3] = vtrnq_f32((float32x4_t)vpf16[2].val[1], (float32x4_t)vpf16[3].val[1]);                     \
        v[0]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[2].val[0]));   \
        v[1]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[0]), vget_low_f32(vpf32[3].val[0]));   \
        v[2]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[2].val[1]));   \
        v[3]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[1]), vget_low_f32(vpf32[3].val[1]));   \
        v[4]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[2].val[0])); \
        v[5]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[0]), vget_high_f32(vpf32[3].val[0])); \
        v[6]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[2].val[1])); \
        v[7]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[1]), vget_high_f32(vpf32[3].val[1])); \
    } while (0)

static void ppl_arm_server_kernel_fp16_n8cx_hgemm_blocking_an_inner_8x8(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k)
{
    const float16x8_t vzeros = vdupq_n_f16(0.0f);

    for (int64_t i = 0; i < m; i += 8) {
        for (int64_t p = 0; p < k; p += 8) {
            int64_t m_l = std::min(m - i, (int64_t)8);
            int64_t k_l = std::min(k - p, (int64_t)8);

            float16x8_t v[8]; // 8 vec reg

            const __fp16 *a_ptr = a + i * lda + p;

            if (k_l == 8) {
                if (m_l == 8) {
                    v[0] = vld1q_f16(a_ptr + 0 * lda);
                    v[1] = vld1q_f16(a_ptr + 1 * lda);
                    v[2] = vld1q_f16(a_ptr + 2 * lda);
                    v[3] = vld1q_f16(a_ptr + 3 * lda);
                    v[4] = vld1q_f16(a_ptr + 4 * lda);
                    v[5] = vld1q_f16(a_ptr + 5 * lda);
                    v[6] = vld1q_f16(a_ptr + 6 * lda);
                    v[7] = vld1q_f16(a_ptr + 7 * lda);
                } else {
                    for (int64_t id = 0; id < m_l; id++) {
                        v[id] = vld1q_f16(a_ptr + id * lda);
                    }
                    for (int64_t id = m_l; id < 8; id++) {
                        v[id] = vzeros;
                    }
                }

                V_TRANSPOSE_FP16_8x8(v);

                vst1q_f16(converted_a + 0, v[0]);
                vst1q_f16(converted_a + 8, v[1]);
                vst1q_f16(converted_a + 16, v[2]);
                vst1q_f16(converted_a + 24, v[3]);
                vst1q_f16(converted_a + 32, v[4]);
                vst1q_f16(converted_a + 40, v[5]);
                vst1q_f16(converted_a + 48, v[6]);
                vst1q_f16(converted_a + 56, v[7]);
            } else {
                for (int64_t ii = 0; ii < m_l; ii++) {
                    for (int64_t pp = 0; pp < k_l; pp++) {
                        converted_a[pp * 8 + ii] = a_ptr[ii * lda + pp];
                    }
                }
                for (int64_t ii = m_l; ii < 8; ii++) {
                    for (int64_t pp = 0; pp < k_l; pp++) {
                        converted_a[pp * 8 + ii] = 0.0f;
                    }
                }
                for (int64_t pp = k_l; pp < 8; pp++) {
                    vst1q_f16(converted_a + pp * 8, vzeros);
                }
            }

            converted_a += 64;
        } // close loop over inner k blocks
    } // close loop over inner m blocks
}

template <>
void ppl_arm_server_kernel_fp16_n8cx_hgemm_blocking_an_outer<N8cxHgemmBlockingOrd::M_N_K>(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1)
{
    for (int64_t i = 0; i < m; i += m_block1) {
        for (int64_t p = 0; p < k; p += k_block1) {
            int64_t m_l1 = std::min(m - i, m_block1);
            int64_t k_l1 = std::min(k - p, k_block1);

            ppl_arm_server_kernel_fp16_n8cx_hgemm_blocking_an_inner_8x8(
                a + i * lda + p,
                converted_a + i * CEIL8(k) + p * CEIL8(m_l1),
                lda,
                m_l1,
                k_l1);

        } // close loop over outer K blocks
    } // close loop over outer M blocks
}

#if 0
template<>
void ppl_arm_server_kernel_fp16_n8cx_hgemm_blocking_an_outer<N8cxHgemmBlockingOrd::N_K_M>(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1)
{
    for (int64_t p = 0; p < k; p += k_block1) {
        for (int64_t i = 0; i < m; i += m_block1) {
            int64_t m_l1 = std::min(m-i, m_block1);
            int64_t k_l1 = std::min(k-p, k_block1);

            ppl3_arm_server_kernel_fp16_nchw8c_hgemm_blocking_an_inner_8x8(
                a + i * lda + p,
                converted_a + p * CEIL8(m) + i * CEIL8(k_l1),
                lda,
                m_l1,
                k_l1);
            
        }  // close loop over outer M blocks
    }  // close loop over outer K blocks
}
#endif

#define FUSE_T()   0
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   1
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   2
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   4
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   5
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   6
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n8cx_hgemm_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T

template <>
void ppl_arm_server_kernel_fp16_n8cx_hgemm<N8cxHgemmBlockingOrd::M_N_K>(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *constant_data,
    const __fp16 *fused_data,
    __fp16 *c,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ld_fused_data,
    const int64_t ldc,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t m_block1,
    const int64_t n_block1,
    const int64_t k_block1,
    const uint32_t fuse_type)
{
    (void)fused_data;

    const int64_t k_m_block0 = CBLK();
    const int64_t k_n_block0 = HGEMM_N_BLOCK0();

    for (int64_t i2 = 0; i2 < m; i2 += m_block1) {
        for (int64_t j2 = 0; j2 < n; j2 += n_block1) {
            for (int64_t p2 = 0; p2 < k; p2 += k_block1) {
                const bool is_first_k = (p2 == 0);
                const bool is_last_k  = (p2 + k_block1 >= k);
                const int64_t m_l1    = std::min(m - i2, m_block1);
                const int64_t n_l1    = std::min(n - j2, n_block1);
                const int64_t k_l1    = std::min(k - p2, k_block1);

                // TODO: distribute FMA chains evenly
                const int64_t n_block0 = k_n_block0;

                const __fp16 *a_ptr     = a + i2 * lda + p2 * CEIL8(m_l1);
                const __fp16 *b_ptr     = b + p2 * ldb + j2 * CBLK();
                const __fp16 *const_ptr = constant_data + i2;
                const __fp16 *fused_ptr = fused_data + i2 * ldc + j2 * CBLK();
                __fp16 *c_ptr           = c + i2 * ldc + j2 * CBLK();

                uint32_t init_id = (is_first_k) ? ((constant_data) ? 1 : 0) : 2;
                // std::cout << "INIT: " << init_id << std::endl;
                uint32_t fuse_id = (is_last_k) ? fuse_type : 0;
                // std::cout << "FUSE: " << fuse_id << std::endl;

                for (int64_t j = 0; j < n_l1; j += n_block0) {
                    for (int64_t i = 0; i < m_l1; i += k_m_block0) {
                        const int64_t m_l0 = std::min((m_l1 - i), k_m_block0);
                        const int64_t n_l0 = std::min((n_l1 - j), n_block0);

                        // std::cout << "N0: " << n_l0 << std::endl;
                        n8cx_hgemm_kernel_func_table[n_l0 - 1][init_id][fuse_id](
                            a_ptr + i * CEIL8(k_l1),
                            b_ptr + j * CBLK(),
                            const_ptr + i,
                            fused_ptr + i * ldc + j * CBLK(),
                            c_ptr + i * ldc + j * CBLK(),
                            m_l0,
                            n_l0,
                            k_l1,
                            lda,
                            ldb,
                            ld_fused_data,
                            ldc);
                    }
                }
            }
        }
    }
}

template <>
void ppl_arm_server_kernel_fp16_n8cx_hgemm<N8cxHgemmBlockingOrd::N_M_K>(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *constant_data,
    const __fp16 *fused_data,
    __fp16 *c,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ld_fused_data,
    const int64_t ldc,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t m_block1,
    const int64_t n_block1,
    const int64_t k_block1,
    const uint32_t fuse_type)
{
    (void)fused_data;

    const int64_t k_m_block0 = CBLK();
    const int64_t k_n_block0 = HGEMM_N_BLOCK0();

    for (int64_t j2 = 0; j2 < n; j2 += n_block1) {
        for (int64_t i2 = 0; i2 < m; i2 += m_block1) {
            for (int64_t p2 = 0; p2 < k; p2 += k_block1) {
                const bool is_first_k = (p2 == 0);
                const bool is_last_k  = (p2 + k_block1 >= k);
                const int64_t m_l1    = std::min(m - i2, m_block1);
                const int64_t n_l1    = std::min(n - j2, n_block1);
                const int64_t k_l1    = std::min(k - p2, k_block1);

                // TODO: distribute FMA chains evenly
                const int64_t n_block0 = k_n_block0;

                const __fp16 *a_ptr     = a + i2 * lda + p2 * CEIL8(m_l1);
                const __fp16 *b_ptr     = b + p2 * ldb + j2 * CBLK();
                const __fp16 *const_ptr = constant_data + p2;
                const __fp16 *fused_ptr = fused_data + i2 * ldc + j2 * CBLK();
                __fp16 *c_ptr           = c + i2 * ldc + j2 * CBLK();

                uint32_t init_id = (is_first_k) ? ((constant_data) ? 1 : 0) : 2;
                uint32_t fuse_id = (is_last_k) ? fuse_type : 0;

                for (int64_t j = 0; j < n_l1; j += n_block0) {
                    for (int64_t i = 0; i < m_l1; i += k_m_block0) {
                        const int64_t m_l0 = std::min((m_l1 - i), k_m_block0);
                        const int64_t n_l0 = std::min((n_l1 - j), n_block0);

                        n8cx_hgemm_kernel_func_table[n_l0 - 1][init_id][fuse_id](
                            a_ptr + i * CEIL8(k_l1),
                            b_ptr + j * CBLK(),
                            const_ptr,
                            fused_ptr + i * ldc + j * CBLK(),
                            c_ptr + i * ldc + j * CBLK(),
                            m_l0,
                            n_l0,
                            k_l1,
                            lda,
                            ldb,
                            ld_fused_data,
                            ldc);
                    }
                }
            }
        }
    }
}