#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_PARAMS_FC_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_PARAMS_FC_PARAM_H_

#include "ppl/nn/common/logger.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/kernel/arm_server/fc/neon/fc.h"
namespace ppl { namespace nn { namespace arm {

struct FCParam{
    ppl::kernel::arm_server::neon::fc_param param;
    ppl::kernel::arm_server::neon::fc_algo_info algo_info;
    ppl::kernel::arm_server::neon::fc_manager* mgr = nullptr;
};
#if 0
struct FCEnhanceParam {
    int32_t post_func; // 0: only bias; 1: relu; 2: relu6
    int32_t pad_type; // 0: normal; 1: reflectpad
    int32_t use_original_filter;
    FCEnhanceParam() {
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

struct FCBaseParam {
    int32_t axis; 
    int32_t num_output; 
    int bias_term;
    FCBaseParam() {
		axis = 0; 
		num_output = 16; 
        bias_term = 0;
    }
    void print() {
      LOG(INFO) << "[" <<
          axis << " " <<
          num_output << " " <<
          "]";
    }
};

struct FCWeightParam {
    uint32_t filter_size;
    void *filter;
    uint32_t bias_size;
    void *bias;
    FCWeightParam() {
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

struct FCParam {
    const char* kernel_name;
    FCBaseParam fc_base_param;
    FCEnhanceParam fc_enhance_param;
    FCWeightParam fc_weight_param;

    void print() {
      LOG(INFO) << kernel_name;
      fc_base_param.print();
      fc_enhance_param.print();
      fc_weight_param.print();
    }
};
#endif

}
}
}

#endif
