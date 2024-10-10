// Autogenerated file by gen_op.py. Do not edit directly!

#include "hpu_ops/op_validator.h"
#include "hpu_ops/backend/reduction_template.h"
#include "_native_batch_norm_legit.h"
#include "convolution_backward_overrideable.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {





struct Gen_native_batch_norm_legit : BatchNormOpBackend {
  Gen_native_batch_norm_legit(int device_id, c10::ScalarType scalar_type) :
      BatchNormOpBackend(device_id, "None", scalar_type, {0, 0, 0}, {}, {}, false) {
        SetSynapseLayouts({synapse_helpers::layouts::SynapseLayoutFormat::WHCN, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE}, {synapse_helpers::layouts::SynapseLayoutFormat::WHCN, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE, synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
        SetOutputMetaFn(BatchNormFwdMeta);
        SetFillParams(FillBatchNormFwdParams);
  }
};

struct Genconvolution_backward_overrideable : ConvolutionBackwardOverrideable {
  Genconvolution_backward_overrideable(int device_id, c10::ScalarType scalar_type) :
      ConvolutionBackwardOverrideable(device_id, "None", scalar_type, {0, 0, 0}, {}, {}, false) {
        SetOutputMetaFn(ConvolutionOverrideableMetaBwd);
  }
};



static const auto& kr_gen_8 = KernelRegistry()
.REGISTER_HPU_BACKEND("aten::_native_batch_norm_legit", Gen_native_batch_norm_legit)
.REGISTER_HPU_BACKEND("aten::convolution_backward_overrideable", Genconvolution_backward_overrideable)
;




}  // namespace habana

