/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/lazy_cast.h"
#include "backend/helpers/cast_sequence.h"
#include "habana_kernels/kernel_utils.h"

namespace habana {

LazyCast::LazyCast(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "lazy_cast_guid", scalar_type, {0}, {}, {}, false) {}

void LazyCast::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() >= 2 && stack.size() <= 4,
      "Incorrect size of inputs expected for cast operator");

  TORCH_CHECK(
      stack[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  auto self = stack_tensor(stack, 0);
  auto src_type = self.scalar_type();
  auto dst_type = stack.at(1).toScalarType();
  auto sizes = self.sizes();

  if (stack.size() > 2) {
    TORCH_CHECK(
        stack[2].isBool(), "Input arg2 expected to be Bool for cast operator");
  }

  if (stack.size() > 3) {
    TORCH_CHECK(
        stack[3].isInt(), "Input arg3 expected to be Int for cast operator");
  }

  auto src_type_cast_type = habana_helpers::DataTypeToCastType(src_type);
  auto dst_type_cast_type = habana_helpers::DataTypeToCastType(dst_type);

  if (src_type_cast_type == dst_type_cast_type) {
    auto out =
        BuildOp(graph, "identity", {syn_in(0)}, {{sizes, ScalarType(), 0}});
    syn_out(0) = std::move(out.at(0));
  } else {
    auto out = BuildCast(this, graph, syn_in(0), sizes, src_type, dst_type, 0);
    syn_out(0) = std::move(out);
  }
}

static void copy_impl(
    const at::Tensor& src,
    const at::IntArrayRef& dst_size,
    const c10::ScalarType& dst_type,
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    synapse_helpers::tensor& output) {
  auto src_type = src.scalar_type();

  auto src_type_cast_type = habana_helpers::DataTypeToCastType(src_type);
  auto dst_type_cast_type = habana_helpers::DataTypeToCastType(dst_type);

  auto shape = src.sizes().vec();
  c10::optional<synapse_helpers::tensor> broadcast;

  if (src.sizes() != dst_size) {
    shape = at::infer_size(src.sizes(), dst_size);
    HABANA_ASSERT(
        shape == dst_size or
            // broadcast with src [1] to dst [] should be valid
            (shape.size() == 1 and dst_size.size() == 0),
        "Cannot broadcast src ",
        src.sizes(),
        " to dst ",
        dst_size);
    broadcast = OpBackend::BuildBroadcast(op, graph, input, shape, src_type);
    input = broadcast.value().get();
  }

  if ((src_type_cast_type == dst_type_cast_type) &&
      !(dst_type == at::ScalarType::Bool && src_type == at::ScalarType::Char)) {
    NodeAttr::NodeOutputAttr out_attr{shape, src_type, 0};
    out_attr.exp_bias = habana_helpers::get_tensor_exp_bias(src);
    output = std::move(
        OpBackend::BuildNode(op, graph, {"identity", {input}, {out_attr}})[0]);
  } else {
    output =
        OpBackend::BuildCast(op, graph, input, shape, src_type, dst_type, 0);
  }
}

static void copy_impl(
    const at::Tensor& src,
    const at::Tensor& dst,
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    synapse_helpers::tensor& output) {
  auto dst_size = dst.sizes();
  auto dst_type = dst.scalar_type();

  copy_impl(src, dst_size, dst_type, op, graph, input, output);
}

CopyFrom::CopyFrom(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {}, {}, {}, true) {}

void CopyFrom::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto src = stack_tensor(stack, 0);
  auto dst = stack_tensor(stack, 1);

  copy_impl(src, dst, this, graph, syn_in(0), syn_out(0));
}

template <bool is_inplace>
struct Copy : OpBackend {
  Copy(int device_id, c10::ScalarType scalar_type);

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
    auto dst = stack_tensor(stack, 0);
    auto src = stack_tensor(stack, 1);

    copy_impl(src, dst, this, graph, syn_in(1), syn_out(0));
  }
};

template <>
Copy<true>::Copy(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {}, {0}, {}, false) {}
template <>
Copy<false>::Copy(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetHwScalingIds({0});
}

struct ToCopy : OpBackend {
  ToCopy(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
  static bool ToCopySTMeta(
      habana_helpers::IShapeList& inputs,
      habana_helpers::IShapeList& outputs);
  static OutputMetaDataVector ToCopyMeta(const at::Stack& stack);
};

ToCopy::ToCopy(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(ToCopy::ToCopyMeta);
  SetSTMetaFn(ToCopy::ToCopySTMeta);
}

bool ToCopy::ToCopySTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  static_cast<void>(inputs);
  static_cast<void>(outputs);

  auto src_type = inputs[0].getScalarType();
  auto dst_type = inputs[1].toScalarType();
  auto src_type_cast_type = habana_helpers::DataTypeToCastType(src_type);
  auto dst_type_cast_type = habana_helpers::DataTypeToCastType(dst_type);
  PT_BRIDGE_DEBUG("Performing cast from:\t", src_type, "\t\tto:\t", dst_type);
  if (!(src_type_cast_type == dst_type_cast_type &&
        !(dst_type == at::ScalarType::Bool &&
          src_type == at::ScalarType::Char))) {
    bool handle_from_bool =
        src_type == at::kBool && GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0;
    if ((handle_from_bool || dst_type == at::kBool)) {
      std::vector<int64_t> out_shape = {1};
      PT_BRIDGE_DEBUG("ToCopySTMeta constant shape ", out_shape);
      habana_helpers::UpdateSTShapeInfo(out_shape);
    }
  }

  return true;
}

OutputMetaDataVector ToCopy::ToCopyMeta(const at::Stack& stack) {
  OutputMetaData meta;
  auto src = stack_tensor(stack, 0);
  auto dtype =
      stack.at(1).isNone() ? src.scalar_type() : stack.at(1).toScalarType();
  auto mem_format = stack.at(6).isNone() ? at::MemoryFormat::Preserve
                                         : stack.at(6).toMemoryFormat();

  if (mem_format == at::MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      meta.strides = src.strides().vec();
    } else {
      mem_format = src.suggest_memory_format();
    }
  }

  meta.shape = src.sizes().vec();
  meta.dtype = dtype;
  meta.mem_format = mem_format;
  PT_BRIDGE_DEBUG(
      "ToCopyMeta dtype:",
      dtype,
      ", meta.shape:",
      meta.shape,
      ", src dtype:",
      src.scalar_type());
  return {meta};
}

void ToCopy::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto src = stack_tensor(stack, 0);
  auto meta = GetOutputMetaData(0);
  copy_impl(src, meta.shape, meta.dtype, this, graph, syn_in(0), syn_out(0));
}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add("aten::copy", KERNEL_FN_GLOBAL(habana::Copy<false>))
        .add("aten::copy_", KERNEL_FN_GLOBAL(habana::Copy<true>))
        .add("aten::_to_copy", KERNEL_FN_GLOBAL(habana::ToCopy))
        .add("hpu::_copy_from", KERNEL_FN_GLOBAL(habana::CopyFrom))
        .add("hpu::cast", KERNEL_FN_GLOBAL(habana::LazyCast))
        .add("hpu::habana_cast_sr_mode", KERNEL_FN_GLOBAL(habana::LazyCast));
