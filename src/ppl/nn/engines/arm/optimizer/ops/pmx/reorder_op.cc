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

#include "ppl/nn/engines/arm/optimizer/ops/pmx/reorder_op.h"
#include "ppl/nn/engines/arm/kernels/pmx/reorder_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {


// uint32_t ReorderOp::reorder_cases_[8][4] = {
//     {DATATYPE_FLOAT32,  DATAFORMAT_NDARRAY,  /**/    DATATYPE_FLOAT32,  DATAFORMAT_N4CX},
//     {DATATYPE_FLOAT32,  DATAFORMAT_N4CX,     /**/    DATATYPE_FLOAT32,  DATAFORMAT_NDARRAY},

//     {DATATYPE_FLOAT16,  DATAFORMAT_NDARRAY,  /**/    DATATYPE_FLOAT16,  DATAFORMAT_N8CX},
//     {DATATYPE_FLOAT16,  DATAFORMAT_N8CX,     /**/    DATATYPE_FLOAT16,  DATAFORMAT_NDARRAY},

//     {DATATYPE_FLOAT32,  DATAFORMAT_NDARRAY,  /**/    DATATYPE_FLOAT16,  DATAFORMAT_NDARRAY},
//     {DATATYPE_FLOAT16,  DATAFORMAT_NDARRAY,  /**/    DATATYPE_FLOAT32,  DATAFORMAT_NDARRAY},

//     {DATATYPE_FLOAT32,  DATAFORMAT_N4CX,     /**/    DATATYPE_FLOAT16,  DATAFORMAT_N8CX},
//     {DATATYPE_FLOAT16,  DATAFORMAT_N8CX,     /**/    DATATYPE_FLOAT32,  DATAFORMAT_N4CX},
// };

ReorderOp::ReorderOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto& input = *info->GetInput<TensorImpl>(0)->GetShape();
        auto& output = *info->GetOutput<TensorImpl>(0)->GetShape();
        if ((output.GetDataFormat() == DATAFORMAT_N4CX || output.GetDataFormat() == DATAFORMAT_N8CX) 
            && input.GetDimCount() == 2) {
            int64_t new_dims[4] = {input.GetDim(0), input.GetDim(1), 1, 1};
            output.Reshape(new_dims, 4);
        } else if (output.GetDataFormat() == DATAFORMAT_N8CX && input.GetDimCount() < 3) {
            auto padded_output_shape = PadShapeTo3Dims(input);
            output.Reshape(padded_output_shape.GetDims(), padded_output_shape.GetDimCount());
        } else {
            if (input.IsScalar()) {
                output.ReshapeAsScalar();
            } else {
                output.Reshape(input.GetDims(), input.GetDimCount());
            }
        }
        LOG(DEBUG) << "Call ReorderOp::[]::infer_dims_func_";
        return RC_SUCCESS;
    };

    infer_layout_func_ = [this](InputOutputInfo* info) -> void {
        DynamicInferLayout(info, &this->common_param_);
    };
}

RetCode ReorderOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode ReorderOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    auto in_type = common_param_.input_types[0];
    auto in_format = common_param_.input_formats[0];
    auto out_type = common_param_.output_types[0];
    auto out_format = common_param_.output_formats[0];

    // for (int i = 0; i < 8; i++) {
    //     if (in_type == reorder_cases_[i][0] && in_format == reorder_cases_[i][1] &&
    //         out_type == reorder_cases_[i][2] && out_format == reorder_cases_[i][3]) {
            
    //         return RC_SUCCESS;
    //     }
    // }
    if (in_type == out_type && in_format == out_format) {
        return RC_SUCCESS;
    }
    if (OptLayoutManager::Check(in_type, in_format, out_type, out_format)) {
        return RC_SUCCESS;
    }
    if (in_format == out_format) {  // cast
        return RC_SUCCESS;
    }

    LOG(ERROR) << "Cannot reorder data from <" << GetDataTypeStr(in_type) << ", " << GetDataFormatStr(in_format) << "> "
               << "to <" << GetDataTypeStr(out_type) << ", " << GetDataFormatStr(out_format) << ">.";
    return RC_UNSUPPORTED;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ReorderOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder output_builder;
    auto fb_output_data = ppl::nn::pmx::arm::CreateOutputDataDirect(output_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(output_builder, ppl::nn::pmx::arm::PrivateDataType_OutputData, fb_output_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(output_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_data = op_builder.CreateVector(output_builder.GetBufferPointer(), output_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_NONE, 0, fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ReorderOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);
    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    
    auto arm_output_data = arm_op_data->value_as_OutputData();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dformat(), &common_param_.output_formats);

    return RC_SUCCESS;
}

#endif

KernelImpl* ReorderOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReorderKernel>();
}

}}} // namespace ppl::nn::arm
