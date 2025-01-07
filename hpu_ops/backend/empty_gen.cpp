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

#include "backend/habana_operator.h"
#include "hpu_ops/empty.h"

namespace {

auto empty_meta(
    std::vector<int64_t>&& sizes,
    std::vector<int64_t>&& strides,
    const at::IValue& dtype_opt,
    const at::IValue& layout_opt,
    const at::IValue& device_opt,
    const at::IValue& pin_memory_opt,
    const at::IValue& memory_format_opt) {
  c10::Device device = device_opt.toOptional<at::Device>().value_or(at::kHPU);
  TORCH_CHECK(device.is_hpu(), "Expected hpu device but got ", device);

  bool pin_memory = pin_memory_opt.toOptional<bool>().value_or(false);
  TORCH_CHECK(!pin_memory, "Only dense CPU tensors can be pinned");

  auto dtype = dtype_opt.toOptional<at::ScalarType>().value_or(
      at::get_default_dtype_as_scalartype());
  auto layout =
      layout_opt.toOptional<at::Layout>().value_or(at::Layout::Strided);
  auto mem_format = memory_format_opt.toOptional<at::MemoryFormat>().value_or(
      at::MemoryFormat::Contiguous);

  return habana::OutputMetaData{dtype, sizes, strides, layout, mem_format};
}

habana::OutputMetaDataVector EmptyMeta(const at::Stack& stack) {
  auto dtype = stack.at(1);
  auto layout = stack.at(2);
  auto device = stack.at(3);
  auto pin_memory = stack.at(4);
  auto memory_format = stack.at(5);
  // convert tensor to shape vector
  std::vector<int64_t> size;
  if (stack.at(0).isTensor()) { // shape tensor for DS
    size = stack.at(0).toTensor().sizes().vec();
  } else {
    size = stack.at(0).toIntVector();
  }
  return {empty_meta(
      std::move(size),
      {} /*strides*/,
      dtype,
      layout,
      device,
      pin_memory,
      memory_format)};
}

habana::OutputMetaDataVector EmptyStridedMeta(const at::Stack& stack) {
  auto size = stack.at(0).toIntVector();
  auto strides = stack.at(1).toIntVector();
  auto dtype = stack.at(2);
  auto layout = stack.at(3);
  auto device = stack.at(4);
  auto pin_memory = stack.at(5);
  return {empty_meta(
      std::move(size),
      std::move(strides),
      dtype,
      layout,
      device,
      pin_memory,
      c10::nullopt)};
}

habana::OutputMetaDataVector EmptyLikeMeta(const at::Stack& stack) {
  const at::Tensor& self = habana::stack_tensor(stack, 0);
  auto size = self.sizes().vec();
  auto dtype =
      stack.at(1).toOptional<at::ScalarType>().value_or(self.scalar_type());
  auto layout = stack[2].toOptional<at::Layout>().value_or(self.layout());
  auto device = stack.at(3).toOptional<at::Device>().value_or(at::kHPU);
  auto pin_memory = stack.at(4);
  auto memory_format = stack[5].toOptional<at::MemoryFormat>().value_or(
      self.suggest_memory_format());
  std::vector<int64_t> strides;
  switch (memory_format) {
    case at::MemoryFormat::ChannelsLast:
      strides = at::get_channels_last_strides_2d(size);
      break;
    case at::MemoryFormat::ChannelsLast3d:
      strides = at::get_channels_last_strides_3d(size);
      break;
    default:
      strides = self.strides().vec();
      break;
  }

  return {empty_meta(
      std::move(size),
      std::move(strides),
      dtype,
      layout,
      device,
      pin_memory,
      memory_format)};
}

auto empty_impl(
    habana::OpBackend* op,
    synapse_helpers::graph& graph,
    const habana::OutputMetaData& md) {
  std::vector<synTensor> inputs;
  op->CreateShapeTensorInput(graph, op->ScalarType(), md.shape, inputs);
  return std::move(habana::OpBackend::BuildNode(
      op, graph, {"memset", inputs, {{md.shape, md.dtype, 0}}})[0]);
}
} // namespace

namespace habana {
struct EmptyBackendDs : OpBackend {
  EmptyBackendDs(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

void Empty::AddNode(
    synapse_helpers::graph& graph,
    [[maybe_unused]] const at::Stack& stack) {
  syn_out(0) = empty_impl(this, graph, GetOutputMetaData(0));
}

void EmptyStrided::AddNode(
    synapse_helpers::graph& graph,
    [[maybe_unused]] const at::Stack& stack) {
  syn_out(0) = empty_impl(this, graph, GetOutputMetaData(0));
}

void EmptyLike::AddNode(
    synapse_helpers::graph& graph,
    [[maybe_unused]] const at::Stack& stack) {
  syn_out(0) = empty_impl(this, graph, GetOutputMetaData(0));
}

Empty::Empty(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(EmptyMeta);
}

EmptyStrided::EmptyStrided(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(EmptyStridedMeta);
}

EmptyLike::EmptyLike(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(EmptyLikeMeta);
}

EmptyBackendDs::EmptyBackendDs(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(EmptyMeta);
}

void EmptyBackendDs::AddNode(
    synapse_helpers::graph& graph,
    [[maybe_unused]] const at::Stack& stack) {
  auto meta = GetOutputMetaData(0);
  syn_out(0) = empty_impl(this, graph, GetOutputMetaData(0));
}

} // namespace habana

static const auto& EmptyKernelRegistry =
    habana::KernelRegistry()
        .add("aten::empty_like", KERNEL_FN_GLOBAL(habana::EmptyLike))
        .add("aten::empty.memory_format", KERNEL_FN_GLOBAL(habana::Empty))
        .add("aten::empty_strided", KERNEL_FN_GLOBAL(habana::EmptyStrided))
        .add("hpu::empty_ds", KERNEL_FN_GLOBAL(habana::EmptyBackendDs));
