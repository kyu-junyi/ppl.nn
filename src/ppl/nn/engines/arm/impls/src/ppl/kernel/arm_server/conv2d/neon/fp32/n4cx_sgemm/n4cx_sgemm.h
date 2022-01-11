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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N4CX_SGEMM_N4CX_SGEMM_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N4CX_SGEMM_N4CX_SGEMM_H_

#include <cstdint>

#define CVL() 4
#define SGEMM_N_BLOCK0() 12

enum class N4cxSgemmBlockingOrd {
    M_N_K,
    N_M_K,
    N_K_M
};

enum class N4cxSgemmFusionOp {
    NONE,      // C <- A x B
    // TODO(kyu): check if we need relu6
    RELU,      // 
    // TODO(kyu): check if this could be combined with "ACCUM"
    // TODO(kyu): check if we need scale sum: C' <- A x B + v_beta * C
    // TODO(kyu): check if we need gemm: C' <- v_alpha * A x B + v_beta * C
    SUM,       // C' <- A x B + C
    // TODO(kyu): specified impls for broadcast at runtime
    SUM_BDCT,  // C <- A x B + __broadcast(vector_c)
};


template<N4cxSgemmBlockingOrd order>
void ppl_arm_server_kernel_fp32_n4cx_sgemm_blocking_an_outer(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

template<N4cxSgemmBlockingOrd order, N4cxSgemmFusionOp fusion>
void ppl_arm_server_kernel_fp32_n4cx_sgemm(
    const float *a,
    const float *b,
    const float *vc,
    float *c,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t M_blocksize,
    const int64_t N_blocksize,
    const int64_t K_blocksize);

#include "n4cx_sgemm_header.inc"
#include "n4cx_sgemm_m8n10_header.inc"

typedef void (*ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_func_t)(
    const float* A,
    const float* B,
    const float* constant,
    const float* DX,
    float* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t lddx,
    const int64_t ldc);

const ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_func_t n4cx_sgemm_m4nx_kernel_func_table[12][3][6] = {
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<1, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<2, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<3, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<4, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<5, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<6, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<7, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<8, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<9, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<10, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<11, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m4nx_func<12, 2, 6>
        }
    }
};

const ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_func_t n4cx_sgemm_m8n10_kernel_func_table[10][3][6] = {
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<1, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<2, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<3, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<4, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<5, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<6, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<7, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<8, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<9, 2, 6>
        }
    },
    {
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 0, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 0, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 0, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 0, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 0, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 0, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 1, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 1, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 1, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 1, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 1, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 1, 6>
        },
        {
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 2, 0>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 2, 1>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 2, 2>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 2, 4>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 2, 5>,
            ppl_arm_server_kernel_fp32_conv_n4cx_sgemm_kernel_m8nx_func<10, 2, 6>
        }
    }
};

template<>
void ppl_arm_server_kernel_fp32_n4cx_sgemm_blocking_an_outer<N4cxSgemmBlockingOrd::M_N_K>(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

template<N4cxSgemmBlockingOrd order>
void ppl_arm_server_kernel_fp32_n4cx_sgemm(
    const float *a,
    const float *b,
    const float *contant,
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
    const uint32_t fuse_type);

template<>
void ppl_arm_server_kernel_fp32_n4cx_sgemm<N4cxSgemmBlockingOrd::M_N_K>(
    const float *a,
    const float *b,
    const float *contant,
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
    const uint32_t fuse_type);

template<>
void ppl_arm_server_kernel_fp32_n4cx_sgemm<N4cxSgemmBlockingOrd::N_M_K>(
    const float *a,
    const float *b,
    const float *contant,
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
    const uint32_t fuse_type);

#endif