/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "hpu_ops/lazy_cast.h"
#include "backend/helpers/cast_sequence.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

LazyCast::LazyCast(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "lazy_cast_guid", scalar_type, {0}, {}, {}, false) {}

void LazyCast::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  HABANA_ASSERT(
      stack.size() >= 2 && stack.size() <= 4,
      "Incorrect size of inputs expected for cast operator");

  HABANA_ASSERT(
      stack[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  auto self = stack_tensor(stack, 0);
  auto src_type = self.scalar_type();
  auto dst_type = stack.at(1).toScalarType();
  auto sizes = self.sizes();

  if (stack.size() > 2) {
    HABANA_ASSERT(
        stack[2].isBool(), "Input arg2 expected to be Bool for cast operator");
  }

  if (stack.size() > 3) {
    HABANA_ASSERT(
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

CopyFrom::CopyFrom(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "copy_guid", scalar_type, {}, {}, {}, true) {
  SetOutputMetaFn(CopyFrom::CopyFromMeta);
}

OutputMetaDataVector CopyFrom::CopyFromMeta(const at::Stack& stack) {
  const auto& src = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = src.sizes().vec();
  meta.dtype = src.scalar_type();

  return {meta};
}

void CopyFrom::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "CopyFrom::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto dst = stackGetter.getNextInput<TensorsPair>();
  const auto meta = CopyFromMeta(stack);
  auto out = CopyHelper(
      self.pt_t.sizes(),
      self.pt_t.scalar_type(),
      dst.pt_t.sizes(),
      dst.pt_t.scalar_type(),
      graph,
      {dst.syn_t, self.syn_t},
      meta,
      0);
  syn_out(0) = std::move(out);
}

OutputMetaDataVector CopyMeta(const at::Stack& stack) {
  const auto& src = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = src.sizes().vec();
  meta.dtype = src.scalar_type();

  return {meta};
}

template <bool is_inplace>
struct Copy : OpBackend {
  Copy(int device_id, c10::ScalarType scalar_type);

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
    const auto meta = CopyMeta(stack);
    StackGetter stackGetter(this, stack, "Copy::AddNode");
    auto self = stackGetter.getNextInput<TensorsPair>();
    auto src = stackGetter.getNextInput<TensorsPair>();
    auto out = CopyHelper(
        src.pt_t.sizes(),
        src.pt_t.scalar_type(),
        self.pt_t.sizes(),
        self.pt_t.scalar_type(),
        graph,
        {self.syn_t, src.syn_t},
        meta,
        0);
    syn_out(0) = std::move(out);
  }
};

template <>
Copy<true>::Copy(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "copy_guid", scalar_type, {}, {0}, {}, false) {}
template <>
Copy<false>::Copy(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "copy_guid", scalar_type, {0}, {}, {}, false) {}

struct ToCopy : OpBackend {
  ToCopy(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph& /*graph*/, const at::Stack& /*stack*/)
      override;
  static bool ToCopySTMeta(
      habana_helpers::IShapeList& inputs,
      habana_helpers::IShapeList& outputs);
  static OutputMetaDataVector ToCopyMeta(const at::Stack& stack);
};

ToCopy::ToCopy(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "copy_guid", scalar_type, {0}, {}, {}, false) {
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
  StackGetter stackGetter(this, stack, "ToCopy::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  const auto meta = GetOutputMetaData();
  auto out = CopyHelper(
      self.pt_t.sizes(),
      self.pt_t.scalar_type(),
      meta[0].shape,
      meta[0].dtype,
      graph,
      {self.syn_t, self.syn_t},
      meta,
      0);
  syn_out(0) = std::move(out);
}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .REGISTER_HPU_BACKEND("aten::copy", habana::Copy<false>)
        .REGISTER_HPU_BACKEND("aten::copy_", habana::Copy<true>)
        .REGISTER_HPU_BACKEND("aten::_to_copy", habana::ToCopy)
        .REGISTER_HPU_BACKEND("hpu::_copy_from", habana::CopyFrom)
        .REGISTER_HPU_BACKEND("hpu::cast", habana::LazyCast)
        .REGISTER_HPU_BACKEND("hpu::habana_cast_sr_mode", habana::LazyCast);
