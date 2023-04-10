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

#include <vector>

#include "ppl/nn/engines/arm/optimizer/rules/convert_constant.h"
#include "ppl/nn/engines/arm/optimizer/rules/utils.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

bool ConvertConstantRule::Apply(const OptKernelOptions& options) {

    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto& tensors = *options.tensors;
    auto& io = *options.io_info;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        io.SetNode(node);

        auto optor = dynamic_cast<ArmOptKernel*>(info->kernels[node->GetId()].get());
        auto status = optor->ConvertConstants(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " Failed";
            status_ = status;
            return false;
        }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        // LOG(INFO) << ".";
#endif
    }

    status_ = RC_SUCCESS;
    // one-time pass
    return false;
}

}}} // namespace ppl::nn::arm
