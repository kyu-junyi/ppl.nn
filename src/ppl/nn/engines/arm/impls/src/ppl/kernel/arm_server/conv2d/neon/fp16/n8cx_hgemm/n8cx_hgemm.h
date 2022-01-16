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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N8CX_HGEMM_N8CX_HGEMM_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N8CX_HGEMM_N8CX_HGEMM_H_

#include <cstdint>

#define CBLK()           8
#define HGEMM_N_BLOCK0() 12

enum class N8cxHgemmBlockingOrd {
    M_N_K,
    N_M_K
};

template <N8cxHgemmBlockingOrd order>
void ppl_arm_server_kernel_fp16_n8cx_hgemm_blocking_an_outer(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

#include "n8cx_hgemm_header.inc"

typedef void (*ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_func_t)(
    const __fp16 *A,
    const __fp16 *B,
    const __fp16 *constant,
    const __fp16 *DX,
    __fp16 *C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t lddx,
    const int64_t ldc);

const ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_func_t n8cx_hgemm_kernel_func_table[12][3][6] = {
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<1, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<2, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<3, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<4, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<5, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<6, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<7, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<8, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<9, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<10, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<11, 2, 6>}},
    {{ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 0, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 0, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 0, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 0, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 0, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 0, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 1, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 1, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 1, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 1, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 1, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 1, 6>},
     {ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 2, 0>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 2, 1>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 2, 2>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 2, 4>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 2, 5>,
      ppl_arm_server_kernel_fp16_conv_n8cx_hgemm_kernel_m8nx_func<12, 2, 6>}}};

template <>
void ppl_arm_server_kernel_fp16_n8cx_hgemm_blocking_an_outer<N8cxHgemmBlockingOrd::M_N_K>(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

template <N8cxHgemmBlockingOrd order>
void ppl_arm_server_kernel_fp16_n8cx_hgemm(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *contant,
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
    const uint32_t fuse_type);

template <>
void ppl_arm_server_kernel_fp16_n8cx_hgemm<N8cxHgemmBlockingOrd::M_N_K>(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *contant,
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
    const uint32_t fuse_type);

template <>
void ppl_arm_server_kernel_fp16_n8cx_hgemm<N8cxHgemmBlockingOrd::N_M_K>(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *contant,
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
    const uint32_t fuse_type);

#endif