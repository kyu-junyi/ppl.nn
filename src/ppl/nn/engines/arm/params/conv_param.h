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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_PARAMS_CONV_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_PARAMS_CONV_PARAM_H_

#include "ppl/nn/common/logger.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"

namespace ppl { namespace nn { namespace arm {

struct Convolution2DParam{
    ppl::kernel::arm_server::conv2d_param param;
//    ppl::kernel::arm_server::conv2d_algo_info algo_info;
    ppl::kernel::arm_server::conv2d_offline_manager* mgr = nullptr;
    ppl::kernel::arm_server::conv2d_offline_manager* fallback_mgr = nullptr;
};
#if 0
struct ConvBaseParam {
		int32_t kernel_h;
		int32_t kernel_w;
		int32_t stride_h;
		int32_t stride_w;
		int32_t pad_h;
		int32_t pad_w;
		int32_t hole_h;
		int32_t hole_w;
		int32_t group;
		int32_t num_output;
		bool bias_term;

    ConvBaseParam() {
        kernel_h = 3;
        kernel_w = 3;
        stride_h = 1;
        stride_w = 1;
        pad_h = 0;
        pad_w = 0;
        hole_h = 1;
        hole_w = 1;
        group = 1;
        num_output = 16;
        bias_term = false;
    }

    void print() {
      LOG(INFO) << "[" <<
          kernel_h << " " <<
          kernel_w << " " <<
          stride_h << " " <<
          stride_w << " " <<
          pad_h << " " <<
          pad_w << " " <<
          hole_h << " " <<
          hole_w << " " <<
          group << " " <<
          num_output << " " <<
          bias_term << " " <<
          "]";
    }
};

struct ConvEnhanceParam {
		int32_t post_func; // 0: only bias; 1: relu; 2: relu6
		int32_t pad_type; // 0: normal; 1: reflectpad
		int32_t use_original_filter;
    ConvEnhanceParam() {
      post_func = 0;
      pad_type = 0;
      use_original_filter = 0;
    }
    void print() {
      LOG(INFO) << "[" <<
          post_func << " " <<
          pad_type << " " <<
          use_original_filter << " " <<
          "]";
    }
};

struct ConvWinoParam {
		int32_t h_batch; 
		int32_t w_batch; 
		int32_t channel_batch; 
		int32_t num_outs_batch;
    ConvWinoParam() {
      h_batch = 0;
      w_batch = 0;
      channel_batch = 0;
      num_outs_batch = 0;
    }
    void print() {
      LOG(INFO) << "[" <<
          h_batch << " " <<
          w_batch << " " <<
          channel_batch << " " <<
          num_outs_batch << " " <<
          "]";
    }
};

struct ConvDirectParam {
		int32_t channel_blk_size; 
		int32_t h_blk_size; 
		int32_t w_blk_size; 
    ConvDirectParam() {
      channel_blk_size = 0;
      h_blk_size = 0;
      w_blk_size = 0;
    }
    void print() {
      LOG(INFO) << "[" <<
          channel_blk_size << " " <<
          h_blk_size << " " <<
          w_blk_size << " " <<
          "]";
    }
};

struct ConvGemmParam {
		int32_t M_L; 
		int32_t N_L;
		int32_t K_L;
    ConvGemmParam() {
      M_L = 0;
      N_L = 0;
      K_L = 0;
    }
    void print() {
      LOG(INFO) << "[" <<
          M_L << " " <<
          N_L << " " <<
          K_L << " " <<
          "]";
    }
};

struct ConvWeightParam {
    uint32_t filter_size;
    void *filter;
    uint32_t bias_size;
    void *bias;
    ConvWeightParam() {
      filter_size = 0;
      bias_size = 0;
      filter = nullptr;
      bias = nullptr;
    }
    void print() {
      LOG(INFO) << "[" <<
          filter_size << " " <<
          filter << " " <<
          bias_size << " " <<
          bias << " " <<
          "]";
    }

};

struct ConvParam {
    const char* kernel_name;
    ConvBaseParam conv_base_param;
    ConvEnhanceParam conv_enhance_param;
    ConvWinoParam conv_wino_param;
    ConvGemmParam conv_gemm_param;
    ConvDirectParam conv_direct_param;
    ConvWeightParam conv_weight_param;

    void print() {
      LOG(INFO) << kernel_name;
      conv_base_param.print();
      conv_enhance_param.print();
      conv_wino_param.print();
      conv_gemm_param.print();
      conv_direct_param.print();
      conv_weight_param.print();
    }
};
#endif
}}}

#endif
