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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/scatter_nd_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/scatter_nd_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_scatter_nd.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ScatterNDOp::ScatterNDOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeScatterND(info, nullptr);
    };

    infer_layout_func_ = GenericInferLayout;
}

RetCode ScatterNDOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode ScatterNDOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    auto& in0_shape = *options.io_info->GetInput<TensorImpl>(0)->GetShape();
    common_param_.input_types[0] = in0_shape.GetDataType();
    common_param_.input_formats[0] = DATAFORMAT_NDARRAY;
    if (!CheckMajorFloat_(common_param_.input_types[0])) {
        LOG(ERROR) << "Unsupported input[0] type for ScatterND Op: " << GetDataTypeStr(common_param_.input_types[0]);
        return RC_UNSUPPORTED;
    }
    // common_param_.input_types[1] = DATATYPE_INT64;
    if (common_param_.input_types[1] != DATATYPE_INT64) {
        LOG(ERROR) << "Unsupported input[1] type for ScatterND Op: " << GetDataTypeStr(common_param_.input_types[1]);
    }
    common_param_.input_formats[1] = DATAFORMAT_NDARRAY;
    common_param_.input_types[2] = common_param_.input_types[0];
    common_param_.input_formats[2] = DATAFORMAT_NDARRAY;

    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

KernelImpl* ScatterNDOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ScatterNdKernel>();
}

}}} // namespace ppl::nn::arm
