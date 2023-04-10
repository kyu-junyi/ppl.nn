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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/tanh_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/tanh_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

TanHOp::TanHOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = GenericInferDims;
    infer_layout_func_ = GenericInferLayout;
}

RetCode TanHOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode TanHOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    GenericSelectInputLayout(options.io_info, common_param_);
    if (!CheckMajorFloat_(common_param_.input_types[0])) {
        LOG(ERROR) << "Unsupported input type for TanH Op: " << GetDataTypeStr(common_param_.input_types[0]);
        return RC_UNSUPPORTED;
    }
    
    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

KernelImpl* TanHOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<TanHKernel>();
}

}}} // namespace ppl::nn::arm
