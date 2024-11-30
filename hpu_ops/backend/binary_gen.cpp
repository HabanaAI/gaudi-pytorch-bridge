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

#include "backend/helpers/cast_sequence.h"
#include "generated/backend/_foreach_add.h"
#include "generated/backend/_foreach_div.h"
#include "generated/backend/add.h"
#include "generated/backend/mul.h"
#include "generated/backend/rsub.h"
#include "generated/backend/sub.h"
#include "hpu_ops/backend/foreach.h"
#include "hpu_ops/common/scalar_dtype_range.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {
const unsigned SELF_INDEX = 0;
const unsigned OTHER_INDEX = 1;
const unsigned ALPHA_INDEX = 2;

sizes_vec BinaryOutputShape(const at::Stack& stack) {
  if (stack.at(0).isScalar() && stack.at(OTHER_INDEX).isTensor()) {
    return {stack_tensor(stack, OTHER_INDEX).sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, SELF_INDEX);
  if (stack.at(OTHER_INDEX).isScalar()) {
    return {self.sizes().vec()};
  }
  const torch::Tensor& other = stack_tensor(stack, OTHER_INDEX);
  return {at::infer_size(self.sizes(), other.sizes())};
}

sizes_vec BinaryOutputShapeInplace(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, SELF_INDEX);
  return {self.sizes().vec()};
}

std::shared_ptr<void> FillBinaryWithAlphaParams(
    const at::Stack& stack,
    size_t& size,
    BinaryWithAlphaMode_t mode) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);
  auto self = stack.at(SELF_INDEX);
  auto other = stack.at(OTHER_INDEX);
  at::ScalarType selfType =
      self.isScalar() ? self.toScalar().type() : self.toTensor().scalar_type();
  at::ScalarType otherType = other.isScalar() ? other.toScalar().type()
                                              : other.toTensor().scalar_type();
  auto alpha = stack.at(ALPHA_INDEX).toScalar();

  if ((c10::isIntegralType(selfType, true) &&
       c10::isIntegralType(otherType, true))) {
    HABANA_ASSERT(
        !alpha.isFloatingPoint(),
        "For integral input tensors, argument alpha must not be a floating",
        "point number.");
    params->alpha.i = alpha.to<int>();
  } else
    params->alpha.f = static_cast<float>(alpha.to<double>());

  params->mode = mode;
  return params;
}

SharedMetaDataVector BinaryWithAlphaAddSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return BinaryWithAlphaSharedMeta(stack, "add");
}

SharedMetaDataVector BinaryWithAlphaSubSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return BinaryWithAlphaSharedMeta(stack, "sub");
}
SharedMetaDataVector BinaryWithAlphaRSubSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return BinaryWithAlphaSharedMeta(stack, "rsub");
}

std::shared_ptr<void> FillBinaryRSubParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(
      stack, size, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_RSUB);
}

std::shared_ptr<void> FillBinarySubParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(
      stack, size, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_SUB);
}

std::shared_ptr<void> FillBinaryAddParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(
      stack, size, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_ADD);
}

static auto BuildBinary(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::string& guid,
    std::vector<synTensor> inputs,
    sizes_vec sizes,
    const std::vector<at::ScalarType>& dtypes,
    at::ScalarType result_type,
    at::optional<at::Scalar> alpha,
    int out_index,
    bool add_casts,
    bool update_guid = true) {
  std::unique_ptr<synapse_helpers::tensor> constant;
  std::vector<synapse_helpers::tensor> mul, cast;

  if (add_casts) {
    auto result_cast_type = habana_helpers::DataTypeToCastType(result_type);
    for (auto i = 0u; i < inputs.size(); ++i) {
      if (result_cast_type == habana_helpers::DataTypeToCastType(dtypes[i])) {
        continue;
      }
      cast.push_back(OpBackend::BuildCast(
          op, graph, inputs[i], sizes[i], dtypes[i], result_type));
      inputs[i] = cast.back().get();
    }
  }

  if (alpha.has_value() and alpha.value().toFloat() != 1.) {
    constant = std::make_unique<synapse_helpers::tensor>(
        OpBackend::BuildConstant(op, graph, *alpha, result_type));
    mul = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("mult", result_type),
         {inputs[OTHER_INDEX], constant->get()},
         {{sizes[OTHER_INDEX], result_type}}});
    inputs[OTHER_INDEX] = mul[0].get();
  }

  auto outshape = at::infer_size(sizes[0], sizes[1]);

  if (update_guid) {
    guid = update_guid_dtype(guid, result_type);
  }
  return OpBackend::BuildNode(
      op, graph, {guid, inputs, {{outshape, result_type, out_index}}});
}

static void update_result_type(
    at::ScalarType& result_type,
    std::string& guid,
    bool cast_int_to_float,
    bool support_int8,
    bool support_int16) {
  bool dtype_changed = false;

  if (cast_int_to_float && isIntegralType(result_type, true)) {
    result_type = torch::kFloat32;
    dtype_changed = true;
  } else {
    if (!support_int8 &&
        (result_type == torch::kInt8 || result_type == torch::kUInt8)) {
      result_type = torch::kInt16;
      dtype_changed = true;
    }
    if (!support_int16 && result_type == torch::kInt16) {
      result_type = torch::kInt32;
      dtype_changed = true;
    }
  }

  if (dtype_changed) {
    guid = update_guid_dtype(guid, result_type);
  }
}

static synapse_helpers::tensor createForeachBinaryNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::string& guid_,
    const std::vector<synTensor>& syn_inputs,
    const std::vector<at::IValue>& pt_inputs,
    int out_index,
    bool cast_int_to_float = false,
    bool support_int8 = true,
    bool support_int16 = true,
    bool mul_or_div_guid = false) {
  const at::Tensor& self = pt_inputs[0].toTensor();
  sizes_vec sizes = {self.sizes().vec()};
  std::vector<synTensor> inputs = syn_inputs;
  std::vector<at::ScalarType> dtypes = {self.scalar_type()};

  at::optional<at::Scalar> alpha = c10::nullopt;
  at::ScalarType result_type;
  at::optional<synapse_helpers::tensor> scalar = c10::nullopt;
  bool update_guid = true;

  if (pt_inputs[1].isTensor()) {
    const at::Tensor& other = pt_inputs[1].toTensor();
    if (pt_inputs.size() > 2) {
      alpha = pt_inputs[2].toScalar();
    }
    result_type = at::result_type(self, other);
    update_result_type(
        result_type, guid_, cast_int_to_float, support_int8, support_int16);

    sizes.push_back(other.sizes().vec());
    dtypes.push_back(other.scalar_type());
  } else {
    const at::Scalar& other = pt_inputs[1].toScalar();
    result_type = at::result_type(self, other);
    update_result_type(
        result_type, guid_, cast_int_to_float, support_int8, support_int16);

    at::ScalarType scalar_type = result_type;
    const float value = other.toFloat();
    if (mul_or_div_guid && is_value_out_of_scalar_range(value, scalar_type)) {
      guid_ = update_guid_dtype(guid_, c10::kFloat);
      scalar_type = c10::kFloat;
      update_guid = false;
    }

    scalar = OpBackend::BuildConstant(op, graph, other, scalar_type);
    inputs.push_back(scalar.value().get());

    sizes.push_back({});
    dtypes.push_back(result_type);
  }

  return std::move(BuildBinary(
      op,
      graph,
      guid_,
      inputs,
      sizes,
      dtypes,
      result_type,
      alpha,
      out_index,
      true,
      update_guid)[0]);
}

static SharedMetaDataVector ForeachBinaryOneIterationSharedMeta(
    const at::Stack& stack,
    const std::string& guid,
    bool castIntToFloat,
    bool supportI8,
    bool supportI16,
    bool mulOrDiv) {
  const auto& self = stack_tensor(stack, SELF_INDEX);
  const auto other = stack.at(OTHER_INDEX);
  auto selfRank = self.dim();
  int64_t otherRank = 1;
  at::optional<at::Scalar> alpha = c10::nullopt;
  bool autocastToF32 = false;
  at::ScalarType resultType;
  std::string updatedGuid = guid;

  SharedMetaDataVector metaVec;
  if (other.isTensor()) {
    const auto& otherTensor = other.toTensor();
    if (stack.size() > 2) {
      alpha = stack.at(ALPHA_INDEX).toScalar();
    }

    otherRank = otherTensor.dim();
    resultType = at::result_type(self, otherTensor);
    update_result_type(
        resultType, updatedGuid, castIntToFloat, supportI8, supportI16);
  } else {
    const auto& otherScalar = stack.at(OTHER_INDEX).toScalar();
    resultType = at::result_type(self, otherScalar);
    update_result_type(
        resultType, updatedGuid, castIntToFloat, supportI8, supportI16);
    const float value = otherScalar.toFloat();
    if (mulOrDiv && is_value_out_of_scalar_range(value, resultType))
      autocastToF32 = true;
  }

  if (alpha.has_value() && alpha.value().toFloat() != 1.) {
    SharedMetaData multSharedMeta{"mult"};
    multSharedMeta.inputs_data = {{otherRank, resultType}, {1, resultType}};
    multSharedMeta.outputs_data.emplace_back(otherRank, resultType);
    metaVec.push_back(multSharedMeta);
  }

  if (autocastToF32)
    resultType = torch::kFloat32;

  auto outputRank = std::max(selfRank, otherRank);
  SharedMetaData sharedMetaBinary{guid};
  sharedMetaBinary.inputs_data = {
      {selfRank, resultType}, {otherRank, resultType}};
  sharedMetaBinary.outputs_data.emplace_back(outputRank, resultType);
  metaVec.push_back(sharedMetaBinary);
  return metaVec;
}

SharedMetaDataVector AddForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack, habana_helpers::HabanaExecutionMode) {
        const bool castIntToFloat = false;
        const bool supportI8 = true;
        const bool supportI16 = true;
        const bool mulOrDiv = false;
        return ForeachBinaryOneIterationSharedMeta(
            stack, "add_fwd", castIntToFloat, supportI8, supportI16, mulOrDiv);
      };

  return CommonForeachBinarySharedMeta(stack, executionMode, sharedMetaCreator);
}

SharedMetaDataVector DivForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack, habana_helpers::HabanaExecutionMode) {
        const bool castIntToFloat = true;
        const bool supportI8 = true;
        const bool supportI16 = true;
        const bool mulOrDiv = true;
        return ForeachBinaryOneIterationSharedMeta(
            stack, "div_fwd", castIntToFloat, supportI8, supportI16, mulOrDiv);
      };

  return CommonForeachBinarySharedMeta(stack, executionMode, sharedMetaCreator);
}

SharedMetaDataVector MaxForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack, habana_helpers::HabanaExecutionMode) {
        const bool castIntToFloat = false;
        const bool supportI8 = false;
        const bool supportI16 = false;
        const bool mulOrDiv = false;
        return ForeachBinaryOneIterationSharedMeta(
            stack, "max_fwd", castIntToFloat, supportI8, supportI16, mulOrDiv);
      };

  return CommonForeachBinarySharedMeta(stack, executionMode, sharedMetaCreator);
}

SharedMetaDataVector MinForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack, habana_helpers::HabanaExecutionMode) {
        const bool castIntToFloat = false;
        const bool supportI8 = false;
        const bool supportI16 = false;
        const bool mulOrDiv = false;
        return ForeachBinaryOneIterationSharedMeta(
            stack, "min_fwd", castIntToFloat, supportI8, supportI16, mulOrDiv);
      };

  return CommonForeachBinarySharedMeta(stack, executionMode, sharedMetaCreator);
}

SharedMetaDataVector MultForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack, habana_helpers::HabanaExecutionMode) {
        const bool castIntToFloat = false;
        const bool supportI8 = true;
        const bool supportI16 = true;
        const bool mulOrDiv = true;
        return ForeachBinaryOneIterationSharedMeta(
            stack, "mult_fwd", castIntToFloat, supportI8, supportI16, mulOrDiv);
      };

  return CommonForeachBinarySharedMeta(stack, executionMode, sharedMetaCreator);
}

SharedMetaDataVector SubForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack, habana_helpers::HabanaExecutionMode) {
        const bool castIntToFloat = false;
        const bool supportI8 = false;
        const bool supportI16 = true;
        const bool mulOrDiv = false;
        return ForeachBinaryOneIterationSharedMeta(
            stack, "sub_fwd", castIntToFloat, supportI8, supportI16, mulOrDiv);
      };

  return CommonForeachBinarySharedMeta(stack, executionMode, sharedMetaCreator);
}

void ForeachBinary::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const bool div_guid = guid_.find("div") != std::string::npos;
  const bool cast_int_to_float = div_guid;
  const bool not_min_or_max = guid_.find("min") == std::string::npos &&
      guid_.find("max") == std::string::npos;
  const bool support_int8 =
      guid_.find("sub") == std::string::npos && not_min_or_max;
  const bool support_int16 = not_min_or_max;
  const bool mul_or_div_guid =
      div_guid || guid_.find("mul") != std::string::npos;

  NodeCreateFunction node_creator =
      [cast_int_to_float, support_int8, support_int16, mul_or_div_guid](
          OpBackend* op,
          synapse_helpers::graph& graph,
          std::string& guid_,
          const std::vector<synTensor>& syn_inputs,
          const std::vector<at::IValue>& pt_inputs,
          int out_index) {
        return createForeachBinaryNode(
            op,
            graph,
            guid_,
            syn_inputs,
            pt_inputs,
            out_index,
            cast_int_to_float,
            support_int8,
            support_int16,
            mul_or_div_guid);
      };

  const size_t size = computeInputsNumber(stack);
  std::vector<synTensor> inputs(size);
  for (size_t i = 0; i < size; i++) {
    inputs[i] = syn_in(i);
  }
  auto results =
      CommonForeachBinary(this, guid_, inputs, graph, stack, node_creator);
  for (size_t i = 0; i < results.size(); i++) {
    syn_out(i) = std::move(results[i]);
  }
}

bool BinarySTMeta(
    habana_helpers::IShapeList&,
    habana_helpers::IShapeList& outputs) {
  static_cast<void>(outputs);

  return true;
}

void BinaryWithAlpha::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, SELF_INDEX);
  auto other = stack.at(OTHER_INDEX);
  at::ScalarType result_type;
  size_t size = 0;

  auto params = FillParams(stack, size);
  const auto outputShape = IsInplace() ? BinaryOutputShapeInplace(stack)[0]
                                       : BinaryOutputShape(stack)[0];

  bool isAlphaIntegralType{false};
  if (other.isTensor()) {
    isAlphaIntegralType =
        c10::isIntegralType(other.toTensor().scalar_type(), true) &&
        c10::isIntegralType(self.scalar_type(), true);

    const at::Tensor& other_tensor = stack_tensor(stack, OTHER_INDEX);
    result_type = at::result_type(self, other_tensor);
  } else {
    isAlphaIntegralType = c10::isIntegralType(other.toScalar().type(), true) &&
        c10::isIntegralType(self.scalar_type(), true);

    const auto& other_scalar = stack.at(OTHER_INDEX).toScalar();
    result_type = at::result_type(self, other_scalar);
  }

  const auto& filledParams =
      std::reinterpret_pointer_cast<ns_BinaryWithAlphaKernel::Params>(params);
  const auto& alpha = filledParams->alpha;
  const auto& mode = filledParams->mode;

  std::vector<synTensor> inputs{syn_in(SELF_INDEX), syn_in(OTHER_INDEX)};
  std::string guid{guid_};

  if (GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER) {
    if ((isAlphaIntegralType ? alpha.i : alpha.f) == 1) {
      std::string opName;
      switch (mode) {
        case BINARY_WITH_ALPHA_MODE_ADD:
          opName = "add";
          break;
        case BINARY_WITH_ALPHA_MODE_RSUB:
          // RSUB uses SUB kernel, but with reversed inputs
          inputs = {syn_in(OTHER_INDEX), syn_in(SELF_INDEX)};
          [[fallthrough]];
        case BINARY_WITH_ALPHA_MODE_SUB:
          opName = "sub";
          break;
        default:
          opName = {};
      }
      guid = get_guid_with_precision(opName, result_type);
    }
  }
  auto op = BuildOp(
      graph,
      std::move(guid),
      std::move(inputs),
      {{outputShape, result_type, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(op[0]);
}

// The same as the native aten.foreach_add_, but we need to disable
// eager compiler for its usage only in accumulate_grads_ op.
struct CustomForeachAdd : ForeachBinary {
  CustomForeachAdd(int device_id, c10::ScalarType scalar_type)
      : ForeachBinary(device_id, "add_fwd", scalar_type, {}, {0}, {}, false) {}
};

OutputMetaDataVector BinaryMeta(
    const at::Stack& stack,
    std::string& guid,
    bool castIntToFloat,
    bool supportI8,
    bool supportI16) {
  const auto& self = stack_tensor(stack, 0);
  const auto other = stack.at(1);
  const bool out_is_available = (stack.size() > 2 && stack.at(2).isTensor());
  at::ScalarType resultType;
  OutputMetaData meta;
  if (other.isTensor()) {
    const auto& otherTensor = other.toTensor();
    resultType = out_is_available ? stack.at(2).toTensor().scalar_type()
                                  : at::result_type(self, otherTensor);
  } else {
    const auto& otherScalar = stack.at(1).toScalar();
    resultType = out_is_available ? stack.at(2).toTensor().scalar_type()
                                  : at::result_type(self, otherScalar);
  }
  update_result_type(resultType, guid, castIntToFloat, supportI8, supportI16);
  meta.dtype = resultType;
  meta.shape = BinaryOutputShape(stack)[0];
  return {meta};
}

OutputMetaDataVector MulMeta(const at::Stack& stack) {
  const bool castIntToFloat = false;
  const bool supportI8 = true;
  const bool supportI16 = true;
  std::string guid = "mult_fwd";
  return BinaryMeta(stack, guid, castIntToFloat, supportI8, supportI16);
}

} // namespace habana

static const auto& ForeachKernelRegistry = habana::KernelRegistry().add(
    "hpu::custom_foreach_add_",
    KERNEL_FN_GLOBAL(habana::CustomForeachAdd));
