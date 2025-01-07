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

#include "generated/backend/linspace.h"

namespace habana {

OutputMetaDataVector LinspaceMeta(const at::Stack& stack) {
  OutputMetaData meta;
  auto ival = stack.at(3);
  if (ival.isTensor()) {
    auto out_tensor = ival.toTensor();
    meta.dtype = out_tensor.scalar_type();
  } else {
    auto end_ival = stack.at(1);
    if (end_ival.isTensor()) {
      meta.dtype = end_ival.toTensor().scalar_type();
    } else if (end_ival.isScalar()) {
      meta.dtype = end_ival.toScalar().type();
    }
  }
  meta.shape = {stack.at(2).toInt()};
  return {meta};
}

std::shared_ptr<void> LinspaceRangeParams(
    const at::Stack& stack,
    size_t& size) {
  float start = stack[0].isScalar() ? stack[0].toScalar().to<float>()
                                    : stack[0].toTensor().item<float>();
  float end = stack[1].isScalar() ? stack[1].toScalar().to<float>()
                                  : stack[1].toTensor().item<float>();
  int steps = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                     : stack[2].toTensor().item<int>();

  float endValueModification = 0.000001;
  int arange_step = steps;

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

SharedMetaDataVector LinspaceOutSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto end = stack.at(1);
  auto steps = stack.at(2);
  int stepsVal = steps.isScalar() ? steps.toScalar().to<int>()
                                  : steps.toTensor().item<int>();
  auto out = stack.at(3);
  auto isOutTensor = out.isTensor();
  auto isEndScalar = end.isScalar();
  c10::ScalarType dtype;
  if (isOutTensor)
    dtype = out.toTensor().scalar_type();
  else if (!isEndScalar)
    dtype = end.toTensor().scalar_type();
  else
    dtype = end.toScalar().type();

  if (!isOutTensor &&
      (dtype == c10::ScalarType::Long || dtype == c10::ScalarType::Int))
    dtype = c10::ScalarType::Float;

  if (stepsVal == 0) {
    SharedMetaData memsetSharedMeta{"memset"};
    memsetSharedMeta.outputs_data.emplace_back(1, dtype);
    return {memsetSharedMeta};
  }

  auto start = stack.at(0);
  auto isStartScalar = start.isScalar();
  float startVal = isStartScalar ? start.toScalar().to<float>()
                                 : start.toTensor().item<float>();
  float endVal =
      isEndScalar ? end.toScalar().to<float>() : end.toTensor().item<float>();
  if (startVal != endVal && stepsVal != 1) {
    if (dtype == c10::ScalarType::Float &&
        habana::HPUDeviceContext::get_device().type() !=
            synDeviceType::synDeviceGaudi) {
      SharedMetaData linspaceSharedMeta{"linspace"};
      if (!isStartScalar && !isEndScalar)
        linspaceSharedMeta.inputs_data = {
            {start.toTensor().dim(), dtype}, {end.toTensor().dim(), dtype}};
      linspaceSharedMeta.outputs_data.emplace_back(1, dtype);

      return {linspaceSharedMeta};
    }

    SharedMetaData rangeSharedMeta{"range"};
    rangeSharedMeta.outputs_data.emplace_back(1, dtype);

    return {rangeSharedMeta};
  }
  // [SW-205149] return empty vector because shape tensor validation will block
  // shape agnostic flow
  return {};
}

void LinspaceOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  OutputMetaData meta = LinspaceMeta(stack)[0];
  auto outshape = meta.shape;

  float start = stack[0].isScalar() ? stack[0].toScalar().to<float>()
                                    : stack[0].toTensor().item<float>();
  float end = stack[1].isScalar() ? stack[1].toScalar().to<float>()
                                  : stack[1].toTensor().item<float>();
  int steps = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                     : stack[2].toTensor().item<int>();

  // For Scalar_Tensor/Tensor_Scalar variants, int/int64 need to be cast to float32
  bool is_tensor_variant = !stack.at(3).isTensor();
  auto dtype = is_tensor_variant &&
          (ScalarType() == c10::ScalarType::Long ||
           ScalarType() == c10::ScalarType::Int)
      ? c10::ScalarType::Float
      : ScalarType();

  if (steps == 0) {
    // return empty tensor if zero steps
    auto result =
        habana::OpBackend::BuildOp(graph, "memset", {}, {{outshape, dtype, 0}});
    syn_out(0) = std::move(result[0]);
  } else {
    if (start != end && steps != 1) {
      size_t size = 0;
      auto params = LinspaceRangeParams(stack, size);
      auto guid = get_guid_with_precision("range", dtype);
      std::vector<synTensor> syn_inputs;
      if (dtype == c10::ScalarType::Float &&
          habana::HPUDeviceContext::get_device().type() !=
              synDeviceType::synDeviceGaudi) {
        guid = "linspace_f32";
        if (stack.at(0).isTensor() && stack.at(1).isTensor()) {
          syn_inputs.emplace_back(syn_in(0));
          syn_inputs.emplace_back(syn_in(1));
        }
         auto linspaceParams = std::make_shared<ns_LinspaceKernel::Params>();
         linspaceParams->start = start;
         linspaceParams->end = end;
         linspaceParams->steps = steps;
         params = linspaceParams;
         size = sizeof(ns_LinspaceKernel::Params);
      }

      auto range = BuildOp(
          graph,
          guid,
          std::move(syn_inputs),
          {{outshape, dtype, 0}},
          params.get(),
          size);

      syn_out(0) = std::move(range[0]);
    } else {
      // return start when start == end or steps == 1
      auto result = ConstantHelper(graph, start, dtype, outshape, 0);
      syn_out(0) = std::move(result);
    }
  }
}
} // namespace habana
