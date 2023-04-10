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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/concat_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_concat.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/concat.h"
#endif

#include <set>

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ConcatOp::ConcatOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConcat(info, param_.get());
    };

    infer_layout_func_ = GenericInferLayout;
}

RetCode ConcatOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ConcatOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    auto info = options.io_info;
    auto const num_inputs = info->GetInputCount();
    if (num_inputs == 1) {
        GenericSelectInputLayout(info, common_param_);
        GenericSelectOutputLayout(info, common_param_);

        return RC_SUCCESS;
    }

    GenericSelectInputLayout(info, common_param_);

    datatype_t common_type = common_param_.input_types[0];
    datatype_t common_major_fp = CheckMajorFloat_(common_param_.input_types[0]) ? common_param_.input_types[0] : DATATYPE_UNKNOWN;
    bool has_integer = CheckDTypes<DATATYPE_INT64>(common_param_.input_types[0]);
    int num_ndarray = (common_param_.input_formats[0] == DATAFORMAT_NDARRAY) ? 1 : 0;
    for (auto i = 1; i < num_inputs; i++) {
        common_type = (common_type == common_param_.input_types[i]) ? common_type : DATATYPE_UNKNOWN;
        if (CheckMajorFloat_(common_param_.input_types[i])) {
            if (DATATYPE_UNKNOWN == common_major_fp) {
                common_major_fp = common_param_.input_types[i];
            } else {
                common_major_fp = (common_major_fp == common_param_.input_types[i]) ? common_major_fp : options.engine_options->forward_precision;
            }
        }
        has_integer = has_integer || CheckDTypes<DATATYPE_INT64>(common_param_.input_types[i]);

        if (common_param_.input_formats[i] == DATAFORMAT_NDARRAY) {
            num_ndarray++;
        }
    }

    if (common_type != DATATYPE_UNKNOWN) {
        if (num_ndarray != 0 && num_ndarray != num_inputs) {
            const dataformat_t common_format = (num_ndarray >= num_inputs - num_inputs) ? DATAFORMAT_NDARRAY : GetNbcxFormat_(common_type);
            for (auto i = 0; i < num_inputs; i++) {
                common_param_.input_formats[i] = common_format;
            }
        }
    } else if (common_major_fp != DATATYPE_UNKNOWN && !has_integer) {
        const dataformat_t common_format = (num_ndarray >= num_inputs - num_inputs) ? DATAFORMAT_NDARRAY : GetNbcxFormat_(common_major_fp);
        for (auto i = 0; i < num_inputs; i++) {
            common_param_.input_types[i] = common_major_fp;
            common_param_.input_formats[i] = common_format;
        }
    } else {
        LOG(ERROR) << "Unsupported input types for Concat Op: found both floating-point types and integer types.";
        return RC_UNSUPPORTED;
    }

    GenericSelectOutputLayout(info, common_param_);
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ConcatOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder output_builder;
    auto fb_output_data = ppl::nn::pmx::arm::CreateOutputDataDirect(output_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(output_builder, ppl::nn::pmx::arm::PrivateDataType_OutputData, fb_output_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(output_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeConcatParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(output_builder.GetBufferPointer(), output_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_ConcatParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ConcatOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    param_ = std::make_shared<ppl::nn::onnx::ConcatParam>();
    ppl::nn::pmx::onnx::DeserializeConcatParam(*fb_op_param->value_as_ConcatParam(), param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    auto arm_output_data = arm_op_data->value_as_OutputData();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dformat(), &common_param_.output_formats);
    return RC_SUCCESS;
}

#endif

KernelImpl* ConcatOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConcatKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
