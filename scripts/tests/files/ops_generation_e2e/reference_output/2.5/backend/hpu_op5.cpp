// Autogenerated file by gen_op.py. Do not edit directly!

#include "hpu_ops/op_validator.h"
#include "mul.h"
#include "sort.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {





struct Genmul_Scalar_out : OpBackend {
  Genmul_Scalar_out(int device_id, c10::ScalarType scalar_type) :
      OpBackend(device_id, "mult_fwd", scalar_type, {}, {}, {1}, true) {
        SetOutputMetaFn(PointwiseMeta<static_cast<int>(DTypeHelper::DtypePromoteVariant::kPromoteToCommon), true, 0, 1>);
        EnableTypePromotion();
  }
};

struct Gensort_values_stable : SortStable {
  Gensort_values_stable(int device_id, c10::ScalarType scalar_type) :
      SortStable(device_id, "None", scalar_type, {}, {}, {}, true) {
        SetNumOutTensors(2);
  }
};



static const auto& kr_gen_5 = KernelRegistry()
.REGISTER_HPU_BACKEND("aten::mul.Scalar_out", Genmul_Scalar_out)
.REGISTER_HPU_BACKEND("aten::sort.values_stable", Gensort_values_stable)
;




}  // namespace habana

