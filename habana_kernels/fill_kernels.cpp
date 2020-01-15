#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

Tensor& habana_fill_(Tensor& self, Scalar value) {
  std::cout << "habana_fill_ called\n";
  TORCH_WARN("habana_fill_ executes CPU kernel internally");

  auto hpu = self.device();
  auto self_ = self.to(DeviceType::CPU);

  auto result = at::native::fill_(self_, value);
  self = self_.to(hpu);
  return self;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(habana_fill_), &habana_fill_>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
