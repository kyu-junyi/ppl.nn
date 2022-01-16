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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP32_DIRECT_NDARRAY_CONV_DIRECT_NDARRAY_KERNEL_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP32_DIRECT_NDARRAY_CONV_DIRECT_NDARRAY_KERNEL_H_

#include <arm_neon.h>
#include <cstdlib>
#include <iostream>

#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

#define CBLK() 4

template <const int64_t oc_section, const int64_t h_tile>
void ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel(
    const float *input_base,
    const float *filter_base,
    const float *bias_base,
    float *output_base,
    float *sum_base,
    const int64_t inH,
    const int64_t inW,
    const int64_t inC,
    const int64_t fltH_valid,
    const int64_t fltW,
    const int64_t strdW,
    const int64_t dltnH,
    const int64_t dltnW,
    const int64_t filter_ic_stride,
    const int64_t outBCHW_stride,
    const uint32_t fuse_type);

#define OUT_TILE_W() 14
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 13
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 12
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 11
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 10
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 9
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 8
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 7
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 6
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 5
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 4
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 3
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 2
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W
#define OUT_TILE_W() 1
#include "conv_direct_ndarray_kernel.inc"
#undef OUT_TILE_W

typedef void (*ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel_func_t)(
    const float *input_base,
    const float *filter_base,
    const float *bias_base,
    float *output_base,
    float *sum_base,
    const int64_t inH,
    const int64_t inW,
    const int64_t inC,
    const int64_t fltH_valid,
    const int64_t fltW,
    const int64_t strdW,
    const int64_t dltnH,
    const int64_t dltnW,
    const int64_t filter_ic_stride,
    const int64_t outBCHW_stride,
    const uint32_t fuse_type);

#define OW_CASE() 14
static ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel_func_t ppl_arm_server_kernel_fp32_conv_direct_ndarray_oc8_kernel_func_table[OW_CASE() + 1] =
    {
        nullptr,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 1>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 2>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 3>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 4>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 5>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 6>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 7>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 8>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 9>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 10>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 11>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 12>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 13>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, 14>,
};

static ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel_func_t ppl_arm_server_kernel_fp32_conv_direct_ndarray_oc4_kernel_func_table[OW_CASE() + 1] =
    {
        nullptr,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 1>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 2>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 3>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 4>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 5>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 6>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 7>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 8>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 9>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 10>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 11>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 12>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 13>,
        ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, 14>,
};
#undef OW_CASE()

}}} // namespace ppl::kernel::arm_server

#endif