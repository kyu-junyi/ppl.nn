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

#include "ppl/kernel/arm_server/conv2d/neon/fp32/n4cx_sgemm/n4cx_sgemm.h"

#include <algorithm>
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

#define CVL()            4
#define SGEMM_N_BLOCK0() 12

#define V_TRANSPOSE_FP32_4x4(v)                                                                  \
    do {                                                                                         \
        float32x4x2_t vpf32[2];                                                                  \
        vpf32[0] = vtrnq_f32(v[0], v[1]);                                                        \
        vpf32[1] = vtrnq_f32(v[2], v[3]);                                                        \
        v[0]     = vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[1].val[0]));   \
        v[1]     = vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[1].val[1]));   \
        v[2]     = vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[1].val[0])); \
        v[3]     = vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[1].val[1])); \
    } while (0)

static void sgemm_n4cx_inner_blocking_4x4_fp32(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k)
{
    (void)sgemm_n4cx_inner_blocking_4x4_fp32;
    const float32x4_t vzeros = vdupq_n_f32(0.0f);

    for (int64_t i = 0; i < m; i += 4) {
        for (int64_t p = 0; p < k; p += 4) {
            int64_t m_l = std::min(m - i, (int64_t)4);
            int64_t k_l = std::min(k - p, (int64_t)4);

            float32x4_t v[4]; // 4 vec reg
            float32x4x2_t vpf32[2]; // 4 vec reg

            const float *a_ptr = a + i * lda + p;

            if (k_l == 4) {
                if (m_l == 4) {
                    v[0] = vld1q_f32(a_ptr + 0 * lda);
                    v[1] = vld1q_f32(a_ptr + 1 * lda);
                    v[2] = vld1q_f32(a_ptr + 2 * lda);
                    v[3] = vld1q_f32(a_ptr + 3 * lda);
                } else {
                    for (int64_t id = 0; id < m_l; id++) {
                        v[id] = vld1q_f32(a_ptr + id * lda);
                    }
                    for (int64_t id = m_l; id < 4; id++) {
                        v[id] = vzeros;
                    }
                }

                vpf32[0] = vtrnq_f32(v[0], v[1]);
                vpf32[1] = vtrnq_f32(v[2], v[3]);
                v[0]     = vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[1].val[0]));
                v[1]     = vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[1].val[1]));
                v[2]     = vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[1].val[0]));
                v[3]     = vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[1].val[1]));

                vst1q_f32(converted_a + 0, v[0]);
                vst1q_f32(converted_a + 4, v[1]);
                vst1q_f32(converted_a + 8, v[2]);
                vst1q_f32(converted_a + 12, v[3]);
            } else {
                for (int64_t ii = 0; ii < m_l; ii++) {
                    for (int64_t pp = 0; pp < k_l; pp++) {
                        converted_a[pp * 4 + ii] = a_ptr[ii * lda + pp];
                    }
                }
                for (int64_t ii = m_l; ii < 4; ii++) {
                    for (int64_t pp = 0; pp < k_l; pp++) {
                        converted_a[pp * 4 + ii] = 0.0f;
                    }
                }
                for (int64_t pp = k_l; pp < 4; pp++) {
                    vst1q_f32(converted_a + pp * 4, vzeros);
                }
            }

            converted_a += 16;
        } // close loop over inner k blocks
    } // close loop over inner m blocks
}

template <>
void sgemm_n4cx_blocking_fp32<N4cxSgemmBlockingOrd::M_N_K>(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1)
{
    for (int64_t i = 0; i < m; i += m_block1) {
        for (int64_t p = 0; p < k; p += k_block1) {
            const int64_t m_l1 = std::min(m - i, m_block1);
            const int64_t k_l1 = std::min(k - p, k_block1);

            sgemm_n4cx_inner_blocking_4x4_fp32(
                a + i * lda + p,
                converted_a + i * CEIL4(k) + p * CEIL4(m_l1),
                lda,
                m_l1,
                k_l1);

        } // close loop over outer K blocks
    } // close loop over outer M blocks
}

template <>
void sgemm_n4cx_blocking_fp32<N4cxSgemmBlockingOrd::N_K_M>(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1)
{
    for (int64_t p = 0; p < k; p += k_block1) {
        for (int64_t i = 0; i < m; i += m_block1) {
            const int64_t m_l1 = std::min(m - i, m_block1);
            const int64_t k_l1 = std::min(k - p, k_block1);

            sgemm_n4cx_inner_blocking_4x4_fp32(
                a + i * lda + p,
                converted_a + p * CEIL4(m) + i * CEIL4(k_l1),
                lda,
                m_l1,
                k_l1);

        } // close loop over outer M blocks
    } // close loop over outer K blocks
}

template <>
void sgemm_n4cx_fp32<N4cxSgemmBlockingOrd::M_N_K>(
    const float *a,
    const float *b,
    const float *constant_data,
    const float *fused_data,
    float *c,
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

    const int64_t k_m_block0 = CVL();
    const int64_t k_n_block0 = SGEMM_N_BLOCK0();

    for (int64_t i2 = 0; i2 < m; i2 += m_block1) {
        for (int64_t j2 = 0; j2 < n; j2 += n_block1) {
            for (int64_t p2 = 0; p2 < k; p2 += k_block1) {
                const bool is_first_k = (p2 == 0);
                const bool is_last_k  = (p2 + k_block1 >= k);
                const int64_t m_l1    = std::min(m - i2, m_block1);
                const int64_t n_l1    = std::min(n - j2, n_block1);
                const int64_t k_l1    = std::min(k - p2, k_block1);

                int64_t n_block0 = k_n_block0;

                const float *a_ptr     = a + i2 * lda + p2 * CEIL4(m_l1);
                const float *b_ptr     = b + p2 * ldb + j2 * CVL();
                const float *const_ptr = constant_data + i2;
                const float *fused_ptr = fused_data + i2 * ldc + j2 * CVL();
                float *c_ptr           = c + i2 * ldc + j2 * CVL();

                uint32_t init_id = (is_first_k) ? ((constant_data) ? 1 : 0) : 2;
                uint32_t fuse_id = (is_last_k) ? fuse_type : 0;

                for (int64_t j = 0; j < n_l1; j += n_block0) {
                    for (int64_t i = 0; i < m_l1; i += k_m_block0) {
                        const int64_t m_l0 = std::min(m_l1 - i, k_m_block0);
                        const int64_t n_l0 = std::min(n_l1 - j, n_block0);

                        sgemm_n4cx_kernel_m4nx_fp32_func_table[n_l0 - 1][init_id][fuse_id](
                            a_ptr + i * CEIL4(k_l1),
                            b_ptr + j * CVL(),
                            const_ptr + i,
                            fused_ptr + i * ldc + j * CVL(),
                            c_ptr + i * ldc + j * CVL(),
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
void sgemm_n4cx_fp32<N4cxSgemmBlockingOrd::N_M_K>(
    const float *a,
    const float *b,
    const float *constant_data,
    const float *fused_data,
    float *c,
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

    const int64_t k_m_block0 = CVL();
    const int64_t k_n_block0 = SGEMM_N_BLOCK0();

    for (int64_t j2 = 0; j2 < n; j2 += n_block1) {
        for (int64_t i2 = 0; i2 < m; i2 += m_block1) {
            for (int64_t p2 = 0; p2 < k; p2 += k_block1) {
                const bool is_first_k = (p2 == 0);
                const bool is_last_k  = (p2 + k_block1 >= k);
                const int64_t m_l1    = std::min(m - i2, m_block1);
                const int64_t n_l1    = std::min(n - j2, n_block1);
                const int64_t k_l1    = std::min(k - p2, k_block1);

                int64_t n_block0 = k_n_block0;

                const float *a_ptr     = a + i2 * lda + p2 * CEIL4(m_l1);
                const float *b_ptr     = b + p2 * ldb + j2 * CVL();
                const float *const_ptr = constant_data + p2;
                const float *fused_ptr = fused_data + i2 * ldc + j2 * CVL();
                float *c_ptr           = c + i2 * ldc + j2 * CVL();

                uint32_t init_id = (is_first_k) ? ((constant_data) ? 1 : 0) : 2;
                uint32_t fuse_id = (is_last_k) ? fuse_type : 0;

                for (int64_t j = 0; j < n_l1; j += n_block0) {
                    for (int64_t i = 0; i < m_l1; i += k_m_block0) {
                        const int64_t m_l0 = std::min(m_l1 - i, k_m_block0);
                        const int64_t n_l0 = std::min(n_l1 - j, n_block0);

                        sgemm_n4cx_kernel_m4nx_fp32_func_table[n_l0 - 1][init_id][fuse_id](
                            a_ptr + i * CEIL4(k_l1),
                            b_ptr + j * CVL(),
                            const_ptr,
                            fused_ptr + i * ldc + j * CVL(),
                            c_ptr + i * ldc + j * CVL(),
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

}}} // namespace ppl::kernel::arm_server
