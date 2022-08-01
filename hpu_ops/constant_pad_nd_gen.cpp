/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/constant_pad_nd.h"
#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_kernels/embedding_kernels.h"

namespace habana {

template <>
ConstantPadNdInputs<at::Tensor>::ConstantPadNdInputs(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  auto self = inputs[0].toTensor();
  auto pad = inputs[1].toIntVector();
  std::vector<at::Tensor> shape_tensors;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    // Keep IDST implementation also, but use H2D implementation by default
    bool isIDST =
        (GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_PAD_HOST_TENSOR) == false);
    if (isIDST) {
      std::vector<int64_t> pad_before(MAX_DIMENSIONS_NUM);
      std::vector<int64_t> pad_after(MAX_DIMENSIONS_NUM);

      for (unsigned int i = 0; i < pad.size() / 2; i++) {
        pad_before[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i];
        pad_after[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i + 1];
      }

      auto pad_before_tensor = habana_lazy::empty_hpu_lazy(
          c10::IntArrayRef(pad_before),
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          false,
          INPUT_DESCRIBING_SHAPE_TENSOR);
      shape_tensors.emplace_back(pad_before_tensor);
      auto pad_after_tensor = habana_lazy::empty_hpu_lazy(
          c10::IntArrayRef(pad_after),
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          false,
          INPUT_DESCRIBING_SHAPE_TENSOR);
      shape_tensors.emplace_back(pad_after_tensor);
    } else {
      std::vector<uint32_t> pad_ht_vec(MAX_DIMENSIONS_NUM * 2, 0);
      // assuming that "pad" has a pair of pad values corresponding to each dim
      // that needs to be padded.
      for (unsigned int i = 0; i < pad.size() / 2; i++) {
        // Host tensor layout 1D - 10 elements: pad_before[0]...pad_before[4],
        // pad_after[0] ... pad_after[4] (for dimensionality IFM less then 5
        // some elements not in use)
        pad_ht_vec[i] = pad[2 * i];
        pad_ht_vec[MAX_DIMENSIONS_NUM + i] = pad[2 * i + 1];
      }

      auto pad_tensor = habana_lazy::empty_hpu_lazy(
          pad_ht_vec.size(),
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          false,
          HOST_TO_DEVICE_TENSOR);
      auto hl_params_shape = habana_lazy::GetHbLazyTensor(pad_tensor);
      auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
      habana_lazy::HbInternalTensorImpl* impl =
          habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
      HABANA_ASSERT(impl);
      impl->set_host_data(
          pad_ht_vec.data(),
          pad_ht_vec.size(),
          sizeof(uint32_t),
          habana_lazy::HostDataType::UINT32_T);
      shape_tensors.emplace_back(pad_tensor);

      auto output_shape_tensor = habana_lazy::empty_hpu_lazy(
          c10::IntArrayRef(PadOperator::compute_output_shape(self, pad)),
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          false,
          SHAPE_TENSOR);
      shape_tensors.emplace_back(output_shape_tensor);
    }
  } else {
    // These tensor won't we used
    auto dummy_tensor1 = habana_lazy::empty_hpu_lazy(
        c10::IntArrayRef(PadOperator::compute_output_shape(self, pad)),
        self.options().dtype(c10::ScalarType::Int),
        self.suggest_memory_format(),
        false,
        SHAPE_TENSOR);
    shape_tensors.emplace_back(dummy_tensor1);
    auto dummy_tensor2 = habana_lazy::empty_hpu_lazy(
        c10::IntArrayRef(PadOperator::compute_output_shape(self, pad)),
        self.options().dtype(c10::ScalarType::Int),
        self.suggest_memory_format(),
        false,
        SHAPE_TENSOR);
    shape_tensors.emplace_back(dummy_tensor2);
  }
  set_inputs({self, pad, inputs[2], shape_tensors});
}

template <>
at::Tensor ConstantPadNdInputs<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}

sizes_vec ConstantPadNdOutputShape(const at::Stack& stack, bool) {
  auto self = stack.at(0).toTensor();
  auto pad = stack.at(1).toIntVector();
  auto outshape = PadOperator::compute_output_shape(self, pad);
  return {{outshape}};
}

void ConstantPadNd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto pad = stack.at(1).toIntVector();
  auto value = stack.at(2).toScalar().to<float>();

  auto output_shape = ConstantPadNdOutputShape(stack, true)[0];

  ns_PadKernelEx::Params params{};
  params.mode = PadMode_t::PAD_MODE_CONSTANT;
  params.value.f = value;

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    if (GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_PAD_HOST_TENSOR)) {
      at::Tensor host_tensor = stack.at(3).toTensorList()[0];
      auto impl = habana_lazy::GetHbInternalTensorImpl(host_tensor);
      HABANA_ASSERT(impl);
      auto h2d_shape = host_tensor.sizes().vec();
      auto input_shape = self.sizes().vec();
      TORCH_CHECK(
          impl->get_host_dt_type() == habana_lazy::HostDataType::UINT32_T,
          "Incorrect datatype of HOST");
      if (habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE) {
        auto ndim = self.dim();
        auto in_data = self.sizes().vec();
        std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
        for (unsigned int i = 0; i < ndim; i++) {
          // order of dims is reversed in H2D tensor
          data[ndim - i - 1] = h2d_shape[i] - input_shape[i];
        }
        impl->set_min<uint32_t>(data);
      } else if (
          habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MAX_SHAPE) {
        auto ndim = self.dim();
        auto in_data = self.sizes().vec();
        std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
        for (unsigned int i = 0; i < ndim; i++) {
          // order of dims is reversed in H2D tensor
          data[ndim - i - 1] = h2d_shape[i] - input_shape[i];
        }
        impl->set_max<uint32_t>(data);
      }
      // pads value shall be picked from H2D tensor, set this to 0's to be safe
      memset(params.pads, 0, sizeof(params.pads));
    }

    auto result = BuildOp(
        graph,
        "pad_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1), syn_in(2)},
        {{output_shape, ScalarType(), 0}},
        &params,
        sizeof(params));
    syn_out(0) = std::move(result.at(0));
  } else {
    auto ndim = self.dim();
    auto lpad = pad.size() / 2;

    memset(params.pads, 0, sizeof(params.pads));
    for (unsigned int i = 0; i < lpad; i++) {
      params.pads[i] = pad[2 * i];
      params.pads[i + ndim] = pad[2 * i + 1];
    }

    std::vector<synTensor> input = {syn_in(0)};

    if (graph.is_dynamic_graph()) {
      std::vector<int64_t> pad_before(MAX_DIMENSIONS_NUM);
      std::vector<int64_t> pad_after(MAX_DIMENSIONS_NUM);

      for (unsigned int i = 0; i < pad.size() / 2; i++) {
        pad_before[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i];
        pad_after[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i + 1];
      }
      input.emplace_back(this->CreateShapeTensorInput(
                                 graph,
                                 torch::kInt,
                                 c10::IntArrayRef(pad_before),
                                 INPUT_DESCRIBING_SHAPE_TENSOR)
                             .get());
      input.emplace_back(this->CreateShapeTensorInput(
                                 graph,
                                 torch::kInt,
                                 c10::IntArrayRef(pad_after),
                                 INPUT_DESCRIBING_SHAPE_TENSOR)
                             .get());
    }

    auto result = BuildOp(
        graph,
        "pad_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {std::move(input)},
        {{output_shape, ScalarType(), 0}},
        &params,
        sizeof(params));
    syn_out(0) = std::move(result.at(0));
  }
}
} // namespace habana