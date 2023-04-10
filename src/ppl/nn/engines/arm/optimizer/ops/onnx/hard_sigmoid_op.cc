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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/hard_sigmoid_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/hard_sigmoid_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

HardSigmoidOp::HardSigmoidOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = GenericInferDims;
    infer_layout_func_ = GenericInferLayout;
}

RetCode HardSigmoidOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode HardSigmoidOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    GenericSelectInputLayout(options.io_info, common_param_);
    if (common_param_.input_types[0] == DATATYPE_BFLOAT16) {
        common_param_.input_types[0] = options.engine_options->forward_precision;
        common_param_.input_formats[0] = GetMajorFormat_(common_param_.input_types[0], common_param_.input_formats[0]);
    }
    if (!CheckMajorFloat_(common_param_.input_types[0])) {
        LOG(ERROR) << "Unsupported input type for HardSigmoid Op: " << GetDataTypeStr(common_param_.input_types[0]);
        return RC_UNSUPPORTED;
    }

    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

KernelImpl* HardSigmoidOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<HardSigmoidKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
