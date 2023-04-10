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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/where_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_where.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

WhereOp::WhereOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeWhere(info, nullptr);
    };

    infer_layout_func_ = [](InputOutputInfo* info) -> void {
        auto in1_shape = info->GetInput<TensorImpl>(1)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataType(in1_shape->GetDataType());
        output_shape->SetDataFormat(in1_shape->GetDataFormat());
    };
}

RetCode WhereOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode WhereOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    GenericSelectInputLayout(options.io_info, common_param_);
    if (common_param_.input_types[0] != DATATYPE_BOOL) {
        LOG(ERROR) << "Unsupported input[0] type for Where Op: " << GetDataTypeStr(common_param_.input_types[0]);
        return RC_UNSUPPORTED;
    }
    // simple resolution for broadcasting 
    common_param_.input_types[2] = common_param_.input_types[1];
    if (common_param_.input_formats[2] != common_param_.input_formats[1]) {
        common_param_.input_formats[2] = common_param_.input_formats[1] = DATAFORMAT_NDARRAY;
    }
    //!CAVEAT: leave broadcasting with two nbcx to runtime
    common_param_.input_formats[0] = common_param_.input_formats[1];

    // Op: main tensor - 1
    common_param_.output_types[0] = common_param_.input_types[1];
    common_param_.output_formats[0] = common_param_.input_formats[1];
    return RC_SUCCESS;
}

KernelImpl* WhereOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<WhereKernel>();
}

}}} // namespace ppl::nn::arm
