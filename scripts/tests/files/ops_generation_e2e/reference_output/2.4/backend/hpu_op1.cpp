// Autogenerated file by gen_op.py. Do not edit directly!

#include "hpu_ops/op_validator.h"
#include "hpu_ops/backend/reduction_template.h"
#include "_fused_dropout.h"
#include "native_dropout.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {





struct Gen_fused_dropout : FusedNativeDropout {
  Gen_fused_dropout(int device_id, c10::ScalarType scalar_type) :
      FusedNativeDropout(device_id, "None", scalar_type, {0, 0}, {}, {}, false) {
        SetOutputMetaFn(FusedNativeDropoutMeta);
        SetFillParams(FillFusedNativeDropoutParams);
  }
};

struct Gennative_dropout : FusedNativeDropout {
  Gennative_dropout(int device_id, c10::ScalarType scalar_type) :
      FusedNativeDropout(device_id, "None", scalar_type, {0, 0}, {}, {}, false) {
        SetOutputMetaFn(FusedNativeDropoutMeta);
        SetFillParams(FillFusedNativeDropoutParams);
  }
};



static const auto& kr_gen_1 = KernelRegistry()
.REGISTER_HPU_BACKEND("aten::_fused_dropout", Gen_fused_dropout)
.REGISTER_HPU_BACKEND("hpu::_fused_dropout", Gen_fused_dropout)
.REGISTER_HPU_BACKEND("aten::native_dropout", Gennative_dropout)
.REGISTER_HPU_BACKEND("hpu::native_dropout", Gennative_dropout)
;



TORCH_LIBRARY_FRAGMENT(hpu, m) {
  static_cast<void>(m);
  m.def("_fused_dropout(Tensor self, float p, Tensor? seed) -> (Tensor, Tensor)");
  m.def("native_dropout(Tensor input, float p, Tensor? seed) -> (Tensor, Tensor)");

}
}  // namespace habana

