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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/exp_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/exp_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ExpOp::ExpOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = GenericInferDims;
    infer_layout_func_ = GenericInferLayout;
}

RetCode ExpOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode ExpOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    auto info = options.io_info;
    GenericSelectInputLayout(info, common_param_);

    if (common_param_.input_types[0] == DATATYPE_BFLOAT16) {
        common_param_.input_types[0] = options.engine_options->forward_precision;
        common_param_.input_formats[0] = GetMajorFormat_(common_param_.input_types[0], common_param_.input_formats[0]);
    }
    if (!CheckMajorFloat_(common_param_.input_types[0])) {
        LOG(ERROR) << "Unsupported input type for Exp Op: " << GetDataTypeStr(common_param_.input_types[0]);
        return RC_UNSUPPORTED;
    }

    GenericSelectOutputLayout(info, common_param_);
    return RC_SUCCESS;
}

KernelImpl* ExpOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ExpKernel>();
}

}}} // namespace ppl::nn::arm
