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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/constant_of_shape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_constant_of_shape.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ConstantOfShapeOp::ConstantOfShapeOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConstantOfShape(info, param_.get());
    };

    infer_layout_func_ = [this](InputOutputInfo* info) -> void {
        TensorShape* shape = info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(param_->data_type);
        shape->SetDataFormat(DATAFORMAT_NDARRAY);
    };
}

RetCode ConstantOfShapeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ConstantOfShapeOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    GenericSelectInputLayout(options.io_info, common_param_);
    if (common_param_.input_types[0] != DATATYPE_INT64) {
        LOG(ERROR) << "Unsupported input type for ConstantOfShape Op: "
                   << GetDataTypeStr(common_param_.input_types[0]);
        return RC_UNSUPPORTED;
    }
    common_param_.input_formats[0] = DATAFORMAT_NDARRAY;

    // Op, param
    common_param_.output_types[0] = param_->data_type;
    common_param_.output_formats[0] = DATAFORMAT_NDARRAY;
    return RC_SUCCESS;
}

KernelImpl* ConstantOfShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConstantOfShapeKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
