/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "generated/backend/logspace.h"

namespace habana {

OutputMetaDataVector LogspaceMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {stack.at(2).toInt()};

  meta.dtype = stack.at(4).toOptional<at::ScalarType>().value_or(
      torch::get_default_dtype_as_scalartype());
  meta.layout =
      stack.at(5).toOptional<at::Layout>().value_or(at::Layout::Strided);

  const auto device = stack.at(6).toOptional<at::Device>().value_or(at::kHPU);
  TORCH_INTERNAL_ASSERT(device.is_hpu());

  const bool pin_memory = stack.at(7).toOptional<bool>().value_or(false);
  TORCH_CHECK(!pin_memory, "Only dense CPU tensors can be pinned");

  return {meta};
}

OutputMetaDataVector LogspaceOutMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {stack.at(2).toInt()};
  meta.dtype = stack.at(4).toTensor().scalar_type();

  return {meta};
}

SharedMetaDataVector LogspaceSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  c10::ScalarType dtype;
  if (stack.at(4).isTensor())
    dtype = stack.at(4).toTensor().scalar_type();
  else
    dtype = stack.at(4).toOptional<at::ScalarType>().value_or(
        torch::get_default_dtype_as_scalartype());

  int64_t len = stack.at(2).toScalar().to<int64_t>();
  float base = stack.at(3).toScalar().to<float>();

  auto castNeeded = c10::isIntegralType(dtype, true);
  auto outType = castNeeded ? at::kFloat : dtype;
  if (len == 0) {
    SharedMetaData memsetSharedMeta{"memset"};
    memsetSharedMeta.outputs_data.emplace_back(1, outType);

    return {memsetSharedMeta};
  } else if (base == 1.f) {
    // [SW-205149] return empty vector because shape tensor validation will
    // block shape agnostic flow
    return {};
  }

  float start = stack.at(0).toScalar().to<float>();
  float end = stack.at(1).toScalar().to<float>();
  SharedMetaDataVector metaVec;
  metaVec.reserve(3);
  SharedMetaTensor commonTensor = {1, outType};
  if (start != end && len != 1) {
    SharedMetaData rangeSharedMeta{"range"};
    rangeSharedMeta.outputs_data.push_back(commonTensor);
    metaVec.push_back(rangeSharedMeta);
  }

  SharedMetaData powSharedMeta{"pow_fwd"};
  powSharedMeta.inputs_data = {commonTensor, commonTensor};
  powSharedMeta.outputs_data.push_back(commonTensor);
  metaVec.push_back(powSharedMeta);

  return metaVec;
}

std::shared_ptr<void> RangeParams(const at::Stack& stack, size_t& size) {
  float start = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();
  int64_t step = stack[2].toScalar().to<int64_t>();

  float endValueModification = 0.000001;
  int64_t arange_step = step;

  float delta = (end - start);
  if (1.0 != arange_step) {
    delta /= (arange_step - 1.0);
  }
  if (arange_step != 1) {
    endValueModification = delta / 2.0;
  }

  end += endValueModification;
  PARAMS_STUB(ns_RangeKernel::Params);

  get<float>(params->start) = start;
  get<float>(params->limit) = end;
  get<float>(params->delta) = delta;

  return params;
}

void LogSpace::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  float start = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();
  int64_t len = stack[2].toScalar().to<int64_t>();
  float base = stack[3].toScalar().to<float>();

  auto castNeeded = c10::isIntegralType(meta.dtype, true);
  auto outType = castNeeded ? at::kFloat : meta.dtype;
  c10::optional<int> finalIndex =
      castNeeded ? c10::nullopt : c10::make_optional<int>(0);

  if (len == 0) {
    auto result = habana::OpBackend::BuildOp(
        graph, "memset", {}, {{meta.shape, outType, 0}});
    syn_out(0) = std::move(result[0]);
  } else if (base == 1.f) {
    auto result = ConstantHelper(
        graph, 1.f, castNeeded ? at::kInt : outType, meta.shape, 0);
    syn_out(0) = std::move(result);
  } else {
    std::vector<synapse_helpers::tensor> range;
    if (start != end && len != 1) {
      size_t size = 0;
      auto params = RangeParams(stack, size);

      range = BuildOp(
          graph,
          get_guid_with_precision("range", outType),
          {},
          {{meta.shape, outType}},
          params.get(),
          size);
    } else {
      range.push_back(ConstantHelper(graph, start, outType, meta.shape));
    }

    auto constant = ConstantHelper(graph, stack[3].toScalar(), outType);

    auto pow = BuildOp(
        graph,
        get_guid_with_precision("pow_fwd", outType),
        {constant.get(), range[0].get()},
        {{meta.shape, outType, finalIndex}});

    auto result = castNeeded
        ? BuildCast(
              this, graph, pow[0].get(), meta.shape, outType, meta.dtype, 0)
        : std::move(pow[0]);

    syn_out(0) = std::move(result);
  }
}
} // namespace habana
