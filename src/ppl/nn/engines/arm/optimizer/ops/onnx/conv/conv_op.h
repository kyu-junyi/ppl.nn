#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_ONNX_OPS_ONNX_CONV_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_ONNX_OPS_ONNX_CONV_OP_H_

#include "ppl/nn/params/onnx/convolution_param.h"
#include "ppl/nn/engines/arm/params/conv_param.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace arm {

class ConvOp final : public ArmOptKernel {
public:
    ConvOp(const ir::Node* node) : ArmOptKernel(node), conv2d_param_(nullptr) {}

    ~ConvOp();
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                        std::vector<ppl::common::datatype_t>* selected_input_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_types,
                                        const ppl::common::datatype_t preferred_fp_datatype) override;
    ppl::common::RetCode SelectAlgorithm(const InputOutputInfo& info, const OptKernelOptions& options) override;
    bool TryFuseReLU(void);
    bool TryFuseReLU6(void);
    bool TryFuseSum(void);

private:
    Convolution2DParam* conv2d_param_;
    std::shared_ptr<ppl::nn::common::ConvolutionParam> param_;
};

}}} // namespace ppl::nn::arm

#endif