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

#include "ppl/nn/engines/arm/kernels/pmx/reorder_kernel.h"

#include <arm_neon.h>
#include <stdio.h>
#include <string.h>
#include "ppl/common/types.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/arm/optimizer/opt_layout.h"
#include "ppl/nn/engines/arm/utils/macros.h"
#include "ppl/kernel/arm_server/common/memory.h"
#include "ppl/kernel/arm_server/cast/neon/cast.h"
#include "ppl/kernel/arm_server/reorder/neon/reorder.h"

using namespace ppl::common;
using namespace ppl::kernel::arm_server;
using namespace ppl::kernel::arm_server::neon;

namespace ppl { namespace nn { namespace arm {
RetCode ReorderKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [input]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto in_shape_desc  = input->GetShape();
    const auto out_shape_desc = output->GetShape();

    const auto input_type  = in_shape_desc->GetDataType();
    const auto output_type = out_shape_desc->GetDataType();
    const auto input_format  = in_shape_desc->GetDataFormat();
    const auto output_format = out_shape_desc->GetDataFormat();

    LOG(DEBUG) << "ARM Data Converter from data format " << input_format << " to " << output_format;
    LOG(DEBUG) << "ARM Data Converter from data type " << input_type << " to " << output_type;

    if (output_format == input_format && output_type == input_type) {
        return memory_copy(input->GetBufferPtr(), in_shape_desc->CalcBytesIncludingPadding(), output->GetBufferPtr());
    }

    const auto num_dims = in_shape_desc->GetDimCount();
    const int64_t shape[4] = {                 in_shape_desc->GetDim(0),     (num_dims > 1) ? in_shape_desc->GetDim(1) : 1,
                              (num_dims > 2) ? in_shape_desc->GetDim(2) : 1, (num_dims > 3) ? in_shape_desc->GetDim(3) : 1 };
    if (output_type == input_type) {
        if (shape[2] == 1 && shape[3] == 1 && input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL) {
            output->TransferBufferFrom(input);
            return RC_SUCCESS;
        }
    }

    auto rc = OptLayoutManager::ConvertLayout(input->GetBufferPtr(), shape, input_type, input_format, output_type, output_format, output->GetBufferPtr());
    if (rc == RC_SUCCESS) {
        return RC_SUCCESS;
    }

    if (output_format == input_format && output_type != input_type) {
        return cast(input->GetShape(), output->GetShape(), input->GetBufferPtr(), output->GetBufferPtr());
    }

    LOG(ERROR) << "Invalid data type conversion";
    return RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
