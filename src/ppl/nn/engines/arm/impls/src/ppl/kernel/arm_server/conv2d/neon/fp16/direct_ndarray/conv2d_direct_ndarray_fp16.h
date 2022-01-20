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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_DIRECT_NDARRAY_CONV2D_DIRECT_NDARRAY_FP16_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_DIRECT_NDARRAY_CONV2D_DIRECT_NDARRAY_FP16_H_

#ifdef PPL_USE_ARM_SERVER_FP16

#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

// forward declare;
class conv2d_direct_ndarray_fp16_offline_manager;

typedef struct {
    // constants in kerrnel design
    int64_t ocblk2 = 16;
    int64_t dst_tile_h = 1;
    int64_t dst_tile_w = 10;
} conv2d_direct_ndarray_fp16_kernel_param;

typedef struct {
    // weight conversion related; should be assigend offline
    int64_t ic_tile;
} conv2d_direct_ndarray_fp16_schedule_param;

class conv2d_direct_ndarray_fp16_runtime_executor final : public conv2d_runtime_executor {
public:
    conv2d_direct_ndarray_fp16_runtime_executor() {}
    conv2d_direct_ndarray_fp16_runtime_executor(const conv2d_param *conv_param, const void *cvt_filter, const void *bias, conv2d_direct_ndarray_fp16_schedule_param sched_param)
        : conv2d_runtime_executor(conv_param, cvt_filter, bias)
        , sched_param_(sched_param) {}
    // calculate overall temp buffer size
    uint64_t cal_temp_buffer_size() override;
    // prepare runtime scheduling params if needed
    ppl::common::RetCode prepare() override;
    // execute op
    ppl::common::RetCode execute() override;

private:
    const conv2d_direct_ndarray_fp16_kernel_param ker_param_;
    conv2d_direct_ndarray_fp16_schedule_param sched_param_;
    // adjust scheduling params if needed during preparation.
    void adjust_schedule_param();

    friend conv2d_direct_ndarray_fp16_offline_manager;
};

class conv2d_direct_ndarray_fp16_offline_manager final : public conv2d_offline_manager {
public:
    conv2d_direct_ndarray_fp16_offline_manager() {}
    conv2d_direct_ndarray_fp16_offline_manager(const conv2d_param &param, ppl::common::Allocator *allocator)
        : conv2d_offline_manager(param, allocator) {}
    bool is_supported() override;
    // initialize scheduling params, e.g., block size, correspoding temp buffer size, etc.
    // for fast algo selection
    ppl::common::RetCode fast_init_schedule_param() override;
    // for offline selecting best algo
    ppl::common::RetCode pick_best_schedule_param(const ppl::nn::TensorShape &src_shape, double &run_time, bool tune_blocksize) override;
    // convert filter according to scheduling params.
    ppl::common::RetCode gen_cvt_weights(const void *filter, const void *bias) override;
    // generate executor for runtime
    conv2d_runtime_executor *gen_executor() override;

private:
    conv2d_direct_ndarray_fp16_schedule_param sched_param_;
};

}}}; // namespace ppl::kernel::arm_server

#endif
#endif
