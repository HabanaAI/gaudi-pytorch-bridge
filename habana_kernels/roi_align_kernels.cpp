#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/roi_align_kernels.h"

using namespace habana;

void RoiAlignFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs expected for RoiAlignFwd operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for RoiAlign operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg1 expected to be tensor for RoiAlign operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg1 expected to be tensor for RoiAlign operator");

  auto input = inputs[0].toTensor();
  auto rois = inputs[1].toTensor();
  auto num_rois = inputs[2].toTensor();
  auto output_h = inputs[3].toInt();
  auto output_w = inputs[4].toInt();
  auto mode = inputs[5].toInt();
  auto sampling_ratio = inputs[6].toInt();
  auto spatial_scale = inputs[7].toScalar().toFloat();
  auto aligned = inputs[8].toBool();

  ns_RoiAlignKernel::ParamsAlignment roi_params{};
  roi_params.mode =
      mode ? RoiAlignMode_t::ROI_ALIGN_MAX : RoiAlignMode_t::ROI_ALIGN_AVG;
  roi_params.sampling_ratio = sampling_ratio;
  roi_params.spatial_scale = spatial_scale;
  roi_params.aligned = aligned;

  std::vector<int64_t> out_shape{
      num_rois.sizes()[0], output_h, output_w, input.sizes()[3]};
  auto output = habana_helpers::createPTTensor(
      input, out_shape, input.options(), is_output_persistent);

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &roi_params, sizeof(roi_params));
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "hpu::roi_align_fwd",
    KERNEL_FN(RoiAlignFwdOperator));
