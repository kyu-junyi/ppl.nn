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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP32_UTILS_CONV2D_UTILS_FP32_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP32_UTILS_CONV2D_UTILS_FP32_H_

#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

void conv2d_n4cx_load_group_fp32(
    const float *input_b_base,
    float *input_gbuf_g_base,
    const int64_t hw_in,
    const int64_t ic_group,
    const int64_t gid_global,
    const int64_t gid_local);

void conv2d_n4cx_store_group_fp32(
    const float *output_gbuf_g_base,
    float *output,
    float *sum,
    const int64_t hw_out,
    const int64_t oc_group,
    const int64_t gid_global,
    const int64_t gid_local,
    const int64_t fuse_flag);

}}}

#endif