#include <torch/script.h>
#include <tpc_kernels/include/perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

Tensor to_cpu(const Tensor& hpu_tensor) {
  if (hpu_tensor.defined())
    return hpu_tensor.to(DeviceType::CPU);
  else
    return hpu_tensor;
}

std::tuple<Tensor, Tensor> nll_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  TORCH_WARN("nll_loss_forward_hpu executes CPU kernel internally");
  auto hpu = self.device();
  auto result = at::native::nll_loss_forward_cpu(
      to_cpu(self), to_cpu(target), to_cpu(weight), reduction, ignore_index);
  return std::make_tuple(
      std::get<0>(result).to(hpu), std::get<1>(result).to(hpu));
}

Tensor nll_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  TORCH_WARN("nll_loss_backward_hpu executes CPU kernel internally");
  auto hpu = self.device();
  auto grad_input =
      to_cpu(at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  at::native::nll_loss_backward_out_cpu(
      grad_input,
      to_cpu(grad_output),
      to_cpu(self),
      to_cpu(target),
      to_cpu(weight),
      reduction,
      ignore_index,
      to_cpu(total_weight));
  return grad_input.to(hpu);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) ->(Tensor output, Tensor total_weight)")
                .impl_unboxedOnlyKernel<
                    decltype(nll_loss_forward_hpu),
                    &nll_loss_forward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(nll_loss_backward_hpu),
                    &nll_loss_backward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
