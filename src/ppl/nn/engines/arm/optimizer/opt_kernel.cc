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

#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"
#include "ppl/common/sys.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ArmOptKernel::ArmOptKernel(const ir::Node* node) : OptKernel(node) {
    common_param_.input_formats.resize(node->GetInputCount(), DATAFORMAT_NDARRAY);
    common_param_.input_types.resize(node->GetInputCount(), DATATYPE_FLOAT32);
    common_param_.output_formats.resize(node->GetOutputCount(), DATAFORMAT_NDARRAY);
    common_param_.output_types.resize(node->GetOutputCount(), DATATYPE_FLOAT32);
}

RetCode ArmOptKernel::ArithmeticSelectTwoInputsLayout(const std::string& op_name, InputOutputInfo* info, ArmCommonParam &common_param)
{
        if (info->GetInputCount() != 2) {
            LOG(ERROR) << op_name << " Op should have 2 inputs.";
            return RC_INVALID_VALUE;
        }

        GenericSelectInputLayout(info, common_param);

        if (!CheckDTypes<DATATYPE_FLOAT32, DATATYPE_FLOAT16, DATATYPE_BFLOAT16, DATATYPE_INT64>(
                    common_param.input_types[0]) ||
            !CheckDTypes<DATATYPE_FLOAT32, DATATYPE_FLOAT16, DATATYPE_BFLOAT16, DATATYPE_INT64>(
                    common_param.input_types[1])) {
            LOG(ERROR) << "Unsupported two input types for " << op_name << " Op: "
                    << GetDataTypeStr(common_param.input_types[0]) << " , "
                    << GetDataTypeStr(common_param.input_types[1]);
            return RC_UNSUPPORTED;
        }

        bool has_float = ((1 << common_param.input_types[0]) | (1 << common_param.input_types[1])) &
                        ((1 << DATATYPE_FLOAT32) | (1 << DATATYPE_FLOAT16) | (1 << DATATYPE_BFLOAT16));
        bool has_int = ((1 << common_param.input_types[0]) | (1 << common_param.input_types[1])) &
                    ((1 << DATATYPE_INT64));
        
        if (has_float && has_int) {
            LOG(ERROR) << "Unsupported two input types for " << op_name << " Op: "
                       << GetDataTypeStr(common_param.input_types[0]) << " , "
                       << GetDataTypeStr(common_param.input_types[1]);
            return RC_UNSUPPORTED;
        }

        if (common_param.input_types[0] == common_param.input_types[1]) {
            common_param.input_formats[1] = common_param.input_formats[0];
        }
        
        // conditionally fallback bfloat16
        int bf16idx = (common_param.input_types[0] == DATATYPE_BFLOAT16) ? 0 : 
                      (common_param.input_types[1] == DATATYPE_BFLOAT16) ? 1 : -1;
        if (bf16idx != -1) {
            common_param.input_types[bf16idx] = common_param.input_types[1-bf16idx];
            common_param.input_formats[bf16idx] = common_param.input_formats[1-bf16idx];
        }

        // Generic
        common_param.output_types[0] = common_param.input_types[0];
        common_param.output_formats[0] = common_param.input_formats[0];
        return RC_SUCCESS;
    }

RetCode ArmOptKernel::RelationSelectTwoInputsLayout(const std::string& op_name, InputOutputInfo* info, ArmCommonParam &common_param) {
    if (info->GetInputCount() != 2) {
        LOG(ERROR) << op_name << " Op should have 2 inputs.";
        return RC_INVALID_VALUE;
    }

    GenericSelectInputLayout(info, common_param);

    bool has_float = ((1 << common_param.input_types[0]) | (1 << common_param.input_types[1])) &
                    ((1 << DATATYPE_FLOAT32) | (1 << DATATYPE_FLOAT16) | (1 << DATATYPE_BFLOAT16));
    bool has_int = ((1 << common_param.input_types[0]) | (1 << common_param.input_types[1])) &
                ((1 << DATATYPE_INT64));
    
    if (has_float && has_int) {
        LOG(ERROR) << "Unsupported two input types for " << op_name << " Op: "
                    << GetDataTypeStr(common_param.input_types[0]) << " , "
                    << GetDataTypeStr(common_param.input_types[1]);
        return RC_UNSUPPORTED;
    }

    if (common_param.input_types[0] == common_param.input_types[1]) {
        common_param.output_types[0] = common_param.input_types[0];
        common_param.output_formats[0] = common_param.input_formats[1] = common_param.input_formats[0];
    }
    
    // conditionally fallback bfloat16
    int bf16idx = (common_param.input_types[0] == DATATYPE_BFLOAT16) ? 0 : 
                    (common_param.input_types[1] == DATATYPE_BFLOAT16) ? 1 : -1;
    if (bf16idx != -1) {
        common_param.input_types[bf16idx] = common_param.input_types[1-bf16idx];
        common_param.input_formats[bf16idx] = common_param.input_formats[1-bf16idx];
    }

    // Op
    common_param.output_types[0] = DATATYPE_BOOL;
    common_param.output_formats[0] = common_param.input_formats[0];
    return RC_SUCCESS;
}

RetCode ArmOptKernel::ReduceSelectLayout(const std::string& op_name,
                                         InputOutputInfo* info, datatype_t major_float_type, bool keep_dims, const std::vector<int32_t>& axes,
                                         ArmCommonParam &common_param) {
    GenericSelectInputLayout(info, common_param);

    if (common_param.input_types[0] == DATATYPE_BFLOAT16) { // fallback bfloat16
        common_param.input_types[0] = major_float_type;
        common_param.input_formats[0] = GetMajorFormat_(common_param.input_types[0], common_param.input_formats[0]);
    }
    if (!CheckDTypes<DATATYPE_FLOAT32, DATATYPE_FLOAT16, DATATYPE_INT64>(common_param.input_types[0])) {
        LOG(ERROR) << "Unsupported input type for " << op_name << " Op: "
                << GetDataTypeStr(common_param.input_types[0]);
        return RC_UNSUPPORTED;
    }

    dataformat_t selected_data_format = DATAFORMAT_NDARRAY;
    const int64_t dim_count = info->GetInput<TensorImpl>(0)->GetShape()->GetDimCount();
    if (dim_count > 0) { // dims has been infered
        if (common_param.input_formats[0] != DATAFORMAT_NDARRAY) { // for NBCX
            if (keep_dims == true) {
                selected_data_format = common_param.input_formats[0];
            } else {
                const int64_t remain_dim_count = dim_count - axes.size();
                if (remain_dim_count >= 3) {
                    bool no_reduce_on_batch_channel_dim = true;
                    for (auto axis : axes) {
                        if (axis == 0 || axis + dim_count == 0 || axis == 1 || axis + dim_count == 1) {
                            no_reduce_on_batch_channel_dim = false;
                            break;
                        }
                    }
                    if (no_reduce_on_batch_channel_dim) {
                        selected_data_format = common_param.input_formats[0];
                    }
                }
            }
        }
    }
    common_param.input_formats[0] = selected_data_format;

    // Generic
    common_param.output_types[0] = common_param.input_types[0];
    common_param.output_formats[0] = common_param.input_formats[0];
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
