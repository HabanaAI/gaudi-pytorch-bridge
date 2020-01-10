#include <torch/script.h>
#include <tpc_kernels/include/perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

std::tuple<Tensor, Tensor> habana_nll_loss_forward(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  TORCH_WARN("habana_nll_loss_forward executes CPU kernel internally");
  auto hpu = self.device();
  auto self_ = self.to(DeviceType::CPU);
  auto target_ = target.to(DeviceType::CPU);
  Tensor weight_ = weight;
  if (weight_.defined())
    weight_ = weight.to(DeviceType::CPU);

  auto result = at::native::nll_loss_forward_cpu(
      self_, target_, weight_, reduction, ignore_index);
  return std::make_tuple(
      std::get<0>(result).to(hpu), std::get<1>(result).to(hpu));
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) ->(Tensor output, Tensor total_weight) ")
        .impl_unboxedOnlyKernel<
            decltype(habana_nll_loss_forward),
            &habana_nll_loss_forward>(TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
