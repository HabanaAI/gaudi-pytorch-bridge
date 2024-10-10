/******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "generated/backend/_foreach_abs.h"
#include "generated/backend/_foreach_add.h"
#include "generated/backend/_foreach_div.h"
#include "generated/backend/_foreach_zero.h"
#include "hpu_ops/backend/foreach.h"
#include "hpu_ops/shared_meta_common.h"

#define UNARY_FOREACH_SHARED_META(name, guid)          \
  SharedMetaDataVector UnaryForeach##name##SharedMeta( \
      const at::Stack& stack) {                        \
    return UnaryForeachSharedMeta(stack, guid);        \
  }

namespace habana {
const unsigned SELF_INDEX = 0;
const unsigned OTHER_INDEX = 1;
const unsigned ALPHA_INDEX = 2;

OutputMetaDataVector CommonForeachMeta(
    const at::Stack& stack,
    const bool cast_int_to_float) {
  auto tensors = stack[0].toTensorList();
  OutputMetaDataVector meta;
  meta.resize(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    meta[i].dtype = tensor.scalar_type();
    meta[i].shape = tensor.sizes().vec();
    if (cast_int_to_float && isIntegralType(meta[i].dtype, true)) {
      meta[i].dtype = torch::kFloat32;
    }
  }

  return meta;
}

OutputMetaDataVector ForeachMeta(const at::Stack& stack) {
  return CommonForeachMeta(stack, false);
}

OutputMetaDataVector NonIntegerForeachMeta(const at::Stack& stack) {
  return CommonForeachMeta(stack, true);
}

UNARY_FOREACH_SHARED_META(Abs, "abs_fwd")
UNARY_FOREACH_SHARED_META(Acos, "acos_fwd")
UNARY_FOREACH_SHARED_META(Asin, "asin_fwd")
UNARY_FOREACH_SHARED_META(Atan, "atan_fwd")
UNARY_FOREACH_SHARED_META(Ceil, "ceil_fwd")
UNARY_FOREACH_SHARED_META(Cos, "cos_fwd")
UNARY_FOREACH_SHARED_META(Cosh, "cosh_fwd")
UNARY_FOREACH_SHARED_META(Erf, "erf_fwd")
UNARY_FOREACH_SHARED_META(Exp, "exp_fwd")
UNARY_FOREACH_SHARED_META(Expm1, "expm1_fwd")
UNARY_FOREACH_SHARED_META(Floor, "floor_fwd")
UNARY_FOREACH_SHARED_META(Lgamma, "gammaln_fwd")
UNARY_FOREACH_SHARED_META(Log, "log_fwd")
UNARY_FOREACH_SHARED_META(Log10, "log10_fwd")
UNARY_FOREACH_SHARED_META(Log1p, "log1p_fwd")
UNARY_FOREACH_SHARED_META(Log2, "log2_fwd")
UNARY_FOREACH_SHARED_META(Neg, "neg_fwd")
UNARY_FOREACH_SHARED_META(Reciprocal, "reciprocal_fwd")
UNARY_FOREACH_SHARED_META(Round, "round_fwd")
UNARY_FOREACH_SHARED_META(Sign, "sign_fwd")
UNARY_FOREACH_SHARED_META(Sigmoid, "sigmoid_fwd")
UNARY_FOREACH_SHARED_META(Sin, "sin_fwd")
UNARY_FOREACH_SHARED_META(Sinh, "sinh_fwd")
UNARY_FOREACH_SHARED_META(Sqrt, "sqrt_fwd")
UNARY_FOREACH_SHARED_META(Tan, "tan_fwd")
UNARY_FOREACH_SHARED_META(Tanh, "tanh_fwd")
UNARY_FOREACH_SHARED_META(Trunc, "trunc_fwd")
UNARY_FOREACH_SHARED_META(Zero, "constant")

static OutputMetaData MetaForSingleOutput(
    const at::Tensor& self,
    const at::IValue& other,
    const bool cast_int_to_float) {
  OutputMetaData meta;
  if (other.isTensor()) {
    const auto& other_tensor = other.toTensor();
    meta.dtype = at::result_type(self, other_tensor);
    meta.shape = at::infer_size(self.sizes(), other_tensor.sizes());
  } else {
    meta.dtype = at::result_type(self, other.toScalar());
    meta.shape = self.sizes().vec();
  }
  if (cast_int_to_float && isIntegralType(meta.dtype, true)) {
    meta.dtype = torch::kFloat32;
  }
  return meta;
}

OutputMetaDataVector CommonForeachBinaryMeta(
    const at::Stack& stack,
    const bool cast_int_to_float) {
  OutputMetaDataVector meta;
  if (stack.at(0).isList()) {
    auto list1 = stack.at(0).toTensorList();
    meta.reserve(list1.size());

    // Second arg could be tensorlist, scalarlist, tensor or scalar
    if (stack.at(1).isList()) {
      const auto& list2 = stack.at(1).toList();
      TORCH_CHECK(
          list1.size() == list2.size(),
          "List1 size: ",
          list1.size(),
          ", != List2 size: ",
          list2.size());
      for (size_t i = 0; i < list1.size(); ++i) {
        meta.push_back(
            MetaForSingleOutput(list1[i], list2[i], cast_int_to_float));
      }
    } else {
      for (const auto& element : list1) {
        meta.push_back(
            MetaForSingleOutput(element, stack.at(1), cast_int_to_float));
      }
    }
  } else {
    auto list = stack[1].toTensorList();
    meta.reserve(list.size());
    for (size_t i = 0; i < list.size(); ++i) {
      meta.push_back(
          MetaForSingleOutput(list[i], stack.at(0), cast_int_to_float));
    }
  }
  return meta;
}

OutputMetaDataVector DivForeachBinaryMeta(const at::Stack& stack) {
  return CommonForeachBinaryMeta(stack, true);
}

OutputMetaDataVector ForeachBinaryMeta(const at::Stack& stack) {
  return CommonForeachBinaryMeta(stack, false);
}

static std::string get_guid(
    const std::string& guid,
    const at::ScalarType dtype) {
  using namespace std::literals;
  static std::unordered_map<std::string_view, std::set<at::ScalarType>>
      map_of_unsupported_dtypes = {
          {"abs_fwd"sv, {torch::kInt16, torch::kInt8}},
          {"ceil_fwd"sv,
           {torch::kInt64, torch::kInt32, torch::kInt16, torch::kInt8}},
          {"floor_fwd"sv,
           {torch::kInt64, torch::kInt32, torch::kInt16, torch::kInt8}},
          {"neg_fwd"sv, {torch::kInt16, torch::kInt8}},
          {"round_fwd"sv,
           {torch::kInt64, torch::kInt32, torch::kInt16, torch::kInt8}},
          {"trunc_fwd"sv,
           {torch::kInt64, torch::kInt32, torch::kInt16, torch::kInt8}},
          {"acos_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
          {"asin_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
          {"atan_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
          {"cosh_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
          {"erf_fwd"sv, {torch::kFloat16}},
          {"expm1_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
          {"gammaln_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
          {"tan_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
          {"sinh_fwd"sv, {torch::kFloat16, torch::kBFloat16}},
      };

  auto it = map_of_unsupported_dtypes.find(guid);
  if (it == map_of_unsupported_dtypes.end() ||
      it->second.find(dtype) == it->second.end()) {
    return get_guid_with_precision(guid, dtype);
  } else if (
      isIntegralType(dtype, true) &&
      it->second.find(torch::kInt32) == it->second.end()) {
    return get_guid_with_precision(guid, torch::kInt32);
  } else {
    return get_guid_with_precision(guid, torch::kFloat32);
  }
}

void Foreach::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  HABANA_ASSERT(
      HPURegistrar::get_device().type() != synDeviceGaudi ||
          guid_.find("gammaln") == std::string::npos,
      "foreach_lgamma is not supported on Gaudi");
  size_t params_size = 0;
  auto params = FillParams(stack, params_size);
  const OutputMetaDataVector output_meta = GetOutputMetaData();
  const auto& tensors = stack[0].toTensorList();
  const std::string guid = guid_.substr(0, guid_.find_last_of('_'));
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildOp(
        graph,
        get_guid(guid, output_meta[i].dtype),
        {syn_in(i)},
        {{tensor.sizes(), tensor.scalar_type(), i}},
        params.get(),
        params_size);
    syn_out(i) = std::move(out[0]);
  }
}

void ForeachZero::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out =
        ConstantHelper(graph, 0, tensor.scalar_type(), tensor.sizes(), i);
    syn_out(i) = std::move(out);
  }
}

size_t computeInputsNumber(const at::Stack& stack) {
  const size_t self_size =
      stack[SELF_INDEX].isTensorList() ? stack[SELF_INDEX].toList().size() : 0;
  const size_t other_size = stack[OTHER_INDEX].isTensorList()
      ? stack[OTHER_INDEX].toList().size()
      : stack[OTHER_INDEX].isTensor() ? 1 : 0;
  return self_size + other_size;
}

SharedMetaDataVector CommonForeachBinarySharedMeta(
    const at::Stack& stack,
    SharedMetaCreateFunction sharedMetaCreator) {
  SharedMetaDataVector metaVec;
  if (stack.at(SELF_INDEX).isTensorList()) {
    const auto& selfs = stack[SELF_INDEX].toList();
    auto selfsSize = selfs.size();
    metaVec.reserve(selfsSize);
    const auto& others = stack[OTHER_INDEX];
    c10::optional<c10::List<c10::IValue>> othersList = c10::nullopt;
    if (others.isList())
      othersList = others.toList();

    for (size_t i = 0; i < selfsSize; i++) {
      c10::IValue other =
          othersList.has_value() ? othersList.value()[i] : others;
      at::Stack oneIterationStack = {selfs[i], other};
      if (stack.size() > 2)
        oneIterationStack.push_back(stack.at(ALPHA_INDEX));

      auto oneIterationSharedMeta = sharedMetaCreator(oneIterationStack);
      metaVec.insert(
          std::end(metaVec),
          std::begin(oneIterationSharedMeta),
          std::end(oneIterationSharedMeta));
    }
  } else {
    const auto& self = stack.at(SELF_INDEX);
    const auto others = stack.at(OTHER_INDEX).toList();
    auto othersSize = others.size();
    for (size_t i = 0; i < othersSize; i++) {
      at::Stack oneIterationStack = {self, others[i]};
      auto oneIterationSharedMeta = sharedMetaCreator(oneIterationStack);
      metaVec.insert(
          std::end(metaVec),
          std::begin(oneIterationSharedMeta),
          std::end(oneIterationSharedMeta));
    }
  }

  return metaVec;
}

std::vector<synapse_helpers::tensor> CommonForeachBinary(
    OpBackend* op,
    std::string& guid,
    const std::vector<synTensor>& inputs,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    NodeCreateFunction node_creator) {
  std::vector<synapse_helpers::tensor> outputs;

  if (stack[SELF_INDEX].isTensorList()) {
    const auto& selfs = stack[SELF_INDEX].toTensorList();
    const auto& others = stack[OTHER_INDEX];
    at::optional<at::Scalar> alpha;

    if (stack.at(1).isTensorList() || stack.at(1).isTensor()) {
      if (stack.size() > 2) {
        alpha = stack[ALPHA_INDEX].toScalar();
      }
      for (size_t i = 0; i < selfs.size(); ++i) {
        const auto& self = selfs[i];
        const auto& other = others.isList() ? others.toList()[i] : others;
        const size_t other_syn_index =
            others.isTensorList() ? i + selfs.size() : selfs.size();

        std::vector<at::IValue> pt_inputs = {self, other};
        if (alpha.has_value()) {
          pt_inputs.push_back(alpha.value());
        }

        outputs.push_back(node_creator(
            op,
            graph,
            guid,
            {inputs[i], inputs[other_syn_index]},
            pt_inputs,
            i));
      }
    } else {
      for (size_t i = 0; i < selfs.size(); ++i) {
        const auto& self = selfs[i];
        const auto& other = others.isList() ? others.toList()[i] : others;

        outputs.push_back(
            node_creator(op, graph, guid, {inputs[i]}, {self, other}, i));
      }
    }
  } else { // ScalarAndTensor variant
    const auto& self = stack[SELF_INDEX].toScalar();
    const auto& others = stack[OTHER_INDEX].toList();
    for (size_t i = 0; i < others.size(); ++i) {
      outputs.push_back(
          node_creator(op, graph, guid, {inputs[i]}, {self, others[i]}, i));
    }
  }

  return outputs;
}

} // namespace habana
