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
#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "backend/backend_meta.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/logging_pt.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "hpu_ops/hpu_op_helper.h"

using namespace torch;
using namespace habana;

namespace {
void print_stride_warning(const Tensor& src, const Tensor& dst) {
  if (src.strides() != dst.strides())
    PT_KERNEL_DEBUG(
        "src device: ",
        src.device(),
        " src.strides(): ",
        src.strides(),
        " src.sizes(): ",
        src.sizes(),
        "\ndst device: ",
        dst.device(),
        " dst.strides(): ",
        dst.strides(),
        " dst.sizes(): ",
        dst.sizes(),
        "\nData will be copied with with basic memcopy so you can expect wrong results");
}

// Add new src->dst cast mappings to this
std::unordered_map<c10::ScalarType, std::vector<c10::ScalarType>> const
    d2d_copy_supported_casts{
        {c10::ScalarType::Byte, {c10::ScalarType::Int, c10::ScalarType::Float}},
        {c10::ScalarType::Float,
         {c10::ScalarType::BFloat16,
          c10::ScalarType::Int,
          c10::ScalarType::Bool}},
        {c10::ScalarType::BFloat16, {c10::ScalarType::Float}},
        {c10::ScalarType::Char,
         {c10::ScalarType::Float, c10::ScalarType::BFloat16}},
        {c10::ScalarType::Bool,
         {c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Int}},
        {c10::ScalarType::Int, {c10::ScalarType::Float}}};

bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  auto is_valid_4d =
      self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast &&
      src.numel() != 0 && self.dim() == 4 &&
      self.scalar_type() == src.scalar_type() &&
      src.is_contiguous(c10::MemoryFormat::Contiguous);

  auto is_valid_5d =
      self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d &&
      src.numel() != 0 && self.dim() == 5 &&
      self.scalar_type() == src.scalar_type() &&
      src.is_contiguous(c10::MemoryFormat::Contiguous);

  return is_valid_4d || is_valid_5d;
}

void adjustPTSizes(Tensor& t) {
  // PT expects metadata like sizes and strides same as in NCHW,
  // but data permuted for channel last, so change the size and stride
  // NCHW
  auto sizes = t.sizes().vec();
  std::vector<int> out_pos = {
      LayoutFormatDims::N,
      LayoutFormatDims::W,
      LayoutFormatDims::C,
      LayoutFormatDims::H};
  std::vector<int> out_pos_5d = {
      LayoutFormatWithDepthDims::N,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C,
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H};
  std::vector<long int> swapped_sizes = {
      sizes[out_pos[0]],
      sizes[out_pos[1]],
      sizes[out_pos[2]],
      sizes[out_pos[3]]};
  std::vector<long int> swapped_sizes_5d = {
      sizes[out_pos_5d[0]],
      sizes[out_pos_5d[1]],
      sizes[out_pos_5d[2]],
      sizes[out_pos_5d[3]],
      sizes[out_pos_5d[4]]};
  if (t.dim() == 5) {
    t.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes_5d);
  } else {
    t.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);
  }
  // For 4D tensors we need to make sure that we generate the PT channel last
  // strides
  if (t.dim() == 4) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast);
  }
  if (t.dim() == 5) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast3d);
  }
}

void do_copy_transpose(Tensor& dst, const Tensor& src) {
  int64_t dim_chl_pos[] = {
      LayoutFormatDims::N,
      LayoutFormatDims::H,
      LayoutFormatDims::W,
      LayoutFormatDims::C};
  at::IntArrayRef chl_pos = dim_chl_pos;
  dst = src.permute(chl_pos);
  adjustPTSizes(dst);
}

void do_d2d_copy(Tensor& dst, const Tensor& src_in, bool non_blocking) {
  // Nothing to do if copy is triggered with same src & dst addresses
  // it actually triggers an assert on func_sim if we trigger this DMA
  // therefore return without doing anything. (this case seen with Mask R-CNN
  // Detectron2 model when copying check-point weights)
  if (dst.data_ptr() == src_in.data_ptr()) {
    return;
  }

  // No direct support for Long in device
  bool same_type = (src_in.scalar_type() == dst.scalar_type());
  auto src =
      (!habana_helpers::is_downcast_to_int_needed(src_in.scalar_type()) ||
       same_type)
      ? src_in
      : habana_helpers::cast_tensor_to_integer(src_in);
  auto src_iter = d2d_copy_supported_casts.find(src.scalar_type());
  auto src_scalar_type = src.scalar_type();
  auto dst_scalar_type = dst.scalar_type();
  bool cast_supported = false;
  // cast is possible if src and dst type mapping present in
  // d2d_copy_supported_casts
  if ((src_iter != d2d_copy_supported_casts.end()) &&
      (std::find(
           src_iter->second.begin(), src_iter->second.end(), dst_scalar_type) !=
       src_iter->second.end()))
    cast_supported = true;

  if (cast_supported) { // if supported src->dst mapping
    dst = habana_helpers::hpu_cast_tensor(
        src, at::scalarTypeToTypeMeta(dst_scalar_type));

  } else { // special cases
    if (dst_scalar_type == c10::ScalarType::Long &&
        src_scalar_type == c10::ScalarType::Int) {
      TORCH_CHECK(
          dst.nbytes() >= src.nbytes(),
          "Unsupported device to device copy: dst size needs to be >= src size");
    } else {
      TORCH_CHECK(
          dst.nbytes() == src.nbytes(), "Unsupported device to device copy");
    }
    if (copy_transpose_valid(dst, src)) {
      do_copy_transpose(dst, src);
    } else {
      habana_helpers::copy_data_within_device(src, dst, non_blocking);
    }
  }
}

} // namespace

// cpu->hpu and hpu->cpu copy implementation
Tensor& copy_hpu_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking,
    synapse_helpers::hpuStream_t hpu_stream) {
  PT_OTHER_OPS_BEGIN; // this macro is used because this kernel is used from
                      // other Lazy kernels
  Tensor& dst = self;
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");
  const auto src_device = src.device().type();
  const auto dst_device = dst.device().type();

  Tensor src_contiguous;
  if (src_device == c10::DeviceType::CPU &&
      dst_device == c10::DeviceType::HPU) {
    // CPU/source tensor should have same dtype as dst & should be contiguous
    // before H2D DMA is triggered
    // Backend kernels should not trigger contiguous call
    TORCH_CHECK(src.is_contiguous(src.suggest_memory_format()));
    src_contiguous = src.to(dst.scalar_type());

    TORCH_CHECK(dst.nbytes() >= src_contiguous.nbytes());
    habana_helpers::copy_data_to_device(
        src_contiguous, dst, non_blocking, hpu_stream);
    print_stride_warning(src_contiguous, dst);
  } else if (
      src_device == c10::DeviceType::HPU &&
      dst_device == c10::DeviceType::CPU) {
    // HPU/source tensor should be contiguous before D2H DMA is triggered
    // Backend kernels should not trigger contiguous call
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      TORCH_CHECK(src.is_contiguous(src.suggest_memory_format()));
      src_contiguous = src;
    } else {
      src_contiguous = src.contiguous(src.suggest_memory_format());
    }

    if (src_contiguous.scalar_type() != dst.scalar_type()) {
      // if src & dst dtypes different, create an intermediate CPU tensor of
      // same dtype as src
      // Note this also covers special handling for int -> long or float ->
      // double casts. Needed in saving checkpoints for RN50 lazy The long
      // integer tensor that is used by PT for BN exp averaging
      // (num_batches_tracked) is converted into int in lazy mode. It needs to
      // be converted back to long while saving the checkpoint
      auto dst_intermediate = at::empty_like(
          dst,
          dst.options().dtype(src_contiguous.scalar_type()),
          dst.suggest_memory_format());
      // Is there any reason why this check cannot be strict equality?
      TORCH_CHECK(dst_intermediate.nbytes() >= src_contiguous.nbytes());
      habana_helpers::copy_data_to_host(
          src_contiguous, dst_intermediate, false, hpu_stream);
      dst.copy_(dst_intermediate.to(dst.scalar_type()));
    } else {
      // Is there any reason why this check cannot be strict equality?
      TORCH_CHECK(dst.nbytes() >= src_contiguous.nbytes());
      habana_helpers::copy_data_to_host(
          src_contiguous, dst, non_blocking, hpu_stream);
    }
    print_stride_warning(src_contiguous, dst);
  } else if (
      src_device == c10::DeviceType::HPU &&
      dst_device == c10::DeviceType::HPU) {
    do_d2d_copy(dst, src, non_blocking);
    print_stride_warning(src, dst);
  } else {
    PT_KERNEL_FATAL(
        "copy_hpu_ doesn't support ", src_device, " to ", dst_device, "copy");
  }

  PT_OTHER_OPS_END;
  return dst;
}

InferOutputMetaRetType MemCopyOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto output = inputs[(inputs.size() == 2) ? 1 : 0].toTensor();
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format()));
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for memcpy, used for D2D mem transfers
 * @param self - input which needs to be transferred
 * @param dest - Destination tensor
 ************************************************************************/
void MemCopyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  at::Tensor output;
  if (inputs.size() == 2) {
    output = inputs[1].toTensor();
    if (output.dim() > 0 && self.dim() > 0) {
      TORCH_CHECK(
          self.sizes() == output.sizes(),
          "incorrect input sizes for Copy Opertion",
          " self.sizes(): ",
          self.sizes(),
          " output.sizes(): ",
          output.sizes());
    }
    // Important:
    // Conditions for calling 'duplicate_tensor_in_memory_section' below
    // must match conditions in 'inplaceInputId' function in
    // jitgraph_utils.cpp
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(output);
  } else {
    output = habana::createPTTensor(self, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  }
  p_context_->params_size_ = 0;
  AddNodeToSynapseGraph(graph, NULL, 0);
}

InferOutputMetaRetType IdentityOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
  }
  auto output = inputs[(inputs.size() == 2) ? 1 : 0].toTensor();
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format()));
  return out;
}

void IdentityOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
  }
  at::Tensor output;

  if (inputs.size() == 2) {
    output = inputs[1].toTensor();
  } else if (
      !graph.is_dry_run() &&
      output_metadata.at(0).allocated_tensor.has_value()) {
    output = output_metadata.at(0).allocated_tensor.value();
  } else {
    output = habana::createPTTensor(self, output_metadata.at(0).persistent);
  }

  p_context_->params_size_ = 0;
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, NULL, 0);
}

/*************************************************************************
 * @brief Kernel implementation for dummy, used for graph ordering
 ************************************************************************/
InferOutputMetaRetType DummyOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  int out_index = inputs.size() - 1;
  auto output = inputs[out_index].toTensor();
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format()));
  return out;
}

void DummyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  static_cast<void>(graph);
  static_cast<void>(output_metadata);
  at::Tensor output;
  int out_index = inputs.size() - 1;
  output = inputs[out_index].toTensor();
  p_context_->params_size_ = 0;
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[out_index],
          graph,
          output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(output);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> AsStridedOperator::
    compute_output_shape(IntArrayRef size, IntArrayRef stride) {
  std::vector<int64_t> out_size_vec;
  std::vector<int64_t> out_stride_vec;

  for (size_t idx = 0; idx < size.size(); idx++) {
    out_size_vec.emplace_back(size[idx]);
    out_stride_vec.emplace_back(stride[idx]);
  }

  return std::make_tuple(out_size_vec, out_stride_vec);
}

/*************************************************************************
 * @brief Kernel implementation for As strided, used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void AsStridedOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  static_cast<void>(graph);
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs[1].isIntList(), "Input arg 1 needs to be of Int List type");
  TORCH_CHECK(
      inputs[2].isIntList(), "Input arg 2 needs to be of Int List type");
  TORCH_CHECK(inputs[3].isScalar(), "Input arg 3 fneeds to be of scalar type");
  auto size = inputs[1].toIntVector();
  auto strides = inputs[2].toIntVector();
  auto offset = inputs[3].toInt();
  auto opt_offset = c10::make_optional(offset);
  at::Tensor output;

  output = at::as_strided(self, size, strides, opt_offset);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section_with_size(
          p_context_->syn_inputs_[0],
          graph,
          size,
          strides,
          offset * self.itemsize(),
          output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(output);
}

/*************************************************************************
 * @brief Kernel implementation for As strided, used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void AsStridedLayoutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  static_cast<void>(graph);
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs[1].isIntList(), "Input arg 1 needs to be of Int List type");

  at::Tensor output;
  int64_t offset = 0;
  c10::optional<int64_t> opt_offset = c10::make_optional((int64_t)0);
  auto sizes = self.sizes().vec();
  auto dims = inputs[1].toIntVector();
  auto is_5d_layout = dims.size() == 5 ? true : false;
  std::vector<int64_t> swapped_sizes = {
      sizes[dims[0]], sizes[dims[1]], sizes[dims[2]], sizes[dims[3]]};
  if (is_5d_layout) {
    swapped_sizes.push_back(sizes[dims[4]]);
  }

  std::vector<long int> new_strides = {
      swapped_sizes[1] * swapped_sizes[2] * swapped_sizes[3],
      swapped_sizes[3] * swapped_sizes[2],
      swapped_sizes[3],
      1};

  if (is_5d_layout) {
    new_strides.clear();
    new_strides.push_back(
        swapped_sizes[4] * swapped_sizes[3] * swapped_sizes[2] *
        swapped_sizes[1]);
    new_strides.push_back(
        swapped_sizes[4] * swapped_sizes[3] * swapped_sizes[2]);
    new_strides.push_back(swapped_sizes[4] * swapped_sizes[3]);
    new_strides.push_back(swapped_sizes[4]);
    new_strides.push_back(1);
  }

  output = at::as_strided(self, swapped_sizes, new_strides, opt_offset);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section_with_size(
          p_context_->syn_inputs_[0],
          graph,
          swapped_sizes,
          new_strides,
          offset * self.itemsize(),
          output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(output);
}

namespace {
static inline Device ensure_has_index(at::Device device) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}
} // namespace

bool is_pinned_hpu(const Tensor& self, at::Device device) {
  ensure_has_index(device);
  return habana::PinnedMemoryAllocator_is_pinned(self.data_ptr());
}

Tensor pin_memory_hpu(const at::Tensor& self, at::Device device) {
  ensure_has_index(device);
  auto* allocator = habana::PinnedMemoryAllocator_get();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

InferOutputMetaRetType SliceInsertOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> shape = self.sizes().vec();

  bool have_shape_tensor = inputs[2].isTensor();
  auto metaData = TensorMetaData(
      shape,
      HabanaOperator::CalculateStrides(shape, self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format());
  InferOutputMetaRetType out;
  out.AddOutputTensor(metaData);

  if (!have_shape_tensor) {
    out.AddShapeTensor(metaData);
  }

  return out;
}

void SliceInsertOperator::ReuseMemoryAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() >= 4, "Incorrect number of arguments for slice insert op");
  // orig, insert, offset, graph_input
  auto graph_input = inputs.back().toTensor();
  auto self = inputs[0].toTensor();
  TORCH_CHECK(graph_input.sizes() == self.sizes(), "incorrect graph input");
  bool have_shape_tensor = inputs[2].isTensor();

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          syn_t_vec[0], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(graph_input);

  if (have_shape_tensor) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    auto paramsList = inputs[2].toIntList();
    synSliceParamsV2 params;
    ComputeParams(params, self, paramsList, graph);

    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void SliceInsertOperator::FixSliceParams(
    at::Tensor self,
    int64_t& dim,
    int64_t& start,
    int64_t& end,
    int64_t& step) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());

  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");

  // INT64_MAX stands for default value.
  if (start == INT64_MAX) {
    start = 0;
  }
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
}

void SliceInsertOperator::ComputeParams(
    synSliceParamsV2& params,
    at::Tensor self,
    c10::List<int64_t> paramsList,
    const synapse_helpers::graph& graph) {
  // set defaults
  std::fill_n(params.axes, HABANA_DIM_MAX, 0);
  std::fill_n(params.starts, HABANA_DIM_MAX, 0);
  std::fill_n(params.ends, HABANA_DIM_MAX, 0);
  std::fill_n(params.steps, HABANA_DIM_MAX, 1);

  int num_slice_params = paramsList.size() / 4;
  for (int i = 0; i < num_slice_params; i++) {
    int64_t dim = paramsList[i * 4];
    int64_t start = paramsList[i * 4 + 1];
    int64_t end = paramsList[i * 4 + 2];
    int64_t step = paramsList[i * 4 + 3];
    FixSliceParams(self, dim, start, end, step);
    params.axes[i] = get_dim_in_tpc_order(dim, self.dim());
    params.starts[i] = start;
    params.ends[i] = end;
    params.steps[i] = step;
    bool needs_params_handling = false;
    if (graph.is_dynamic_graph() && (!graph.is_dry_run()) &&
        end > self.sizes().vec()[dim]) {
      needs_params_handling = true;
    }
    if (needs_params_handling) {
      synapse_helpers::tensor& syn_input_tensor = p_context_->syn_inputs_[0];
      auto tensor_id = syn_input_tensor.id();
      std::vector<int64_t> min, max;
      std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
      params.ends[i] = max[dim];
    }
  }
}

void SliceInsertOperator::ValidateSliceInsertInputs(
    std::vector<int64_t>& inp_shape,
    std::vector<int64_t>& out_shape,
    std::vector<int64_t>& step,
    std::vector<int64_t>& start) {
  for (unsigned i = 0; i < inp_shape.size(); i++) {
    if (inp_shape[i]) {
      TORCH_CHECK(
          (start[i] < inp_shape[i]),
          "SliceInsert starts param, which is greater or equal to the dimension");
    }

    // original equation as per at::native::slice
    // sizes[dim] = (end_val - start_val + step - 1) / step; // round-up
    // inverse to find end
    // end_val = sizes[dim]*step + 1 - step + start_val
    auto end_val = out_shape[i] * step[i] + 1 - step[i] + start[i];

    TORCH_CHECK(
        (end_val <= inp_shape[i]),
        "SliceInsert invalid end param, which is greater or equal to the dimension ",
        end_val,
        " ",
        inp_shape[i]);
  }
}

void SliceInsertOperator::UpdateMaxPassSliceInputs(
    std::vector<int64_t>& inp_shape,
    std::vector<int64_t>& out_shape,
    std::vector<int64_t>& step,
    std::vector<int64_t>& start,
    std::vector<int64_t>& min,
    std::vector<int64_t>& max) {
  for (uint64_t i = 0; i < inp_shape.size(); i++) {
    out_shape[i] = out_shape[i] < inp_shape[i] ? out_shape[i] : inp_shape[i];
    // output = slice to be inserted
    // (end - start)/step = output
    // Assuming end max is input
    // (input - start)/step = output
    // input - start = output * step
    // start = input - output * step

    // start shape tensor is updated, so change the buckets as well if
    // start is there in bucket
    if (out_shape[i] != 0) {
      auto old_start = start[i] * step[i];
      start[i] = inp_shape[i] - (out_shape[i] * step[i]);
      if (old_start != start[i]) {
        // If the calculated value is less than current value, keep the
        // current value.
        HABANA_ASSERT(min.size() == max.size());
        if (min.size() && (min[i] != max[i]) && old_start == 0) {
          auto curr_val = max[i] /
              habana_helpers::DynamicBucketInfo::default_max_multiplier_;
          if (start[i] < curr_val) {
            start[i] = curr_val;
          }
        }
      }
    }
  }
  TORCH_CHECK(
      min <= start, "SliceInsertOperator Start tensor min is greater than max");
}

void SliceInsertOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  auto self = inputs[0].toTensor();
  bool has_shape_tensor = inputs[2].isTensor();
  if (has_shape_tensor && inputs.size() == 4) {
    TORCH_CHECK(
        inputs.size() == 4,
        "Incorrect size of inputs expected for slice_insert operator");
    TORCH_CHECK(
        p_context_->syn_inputs_[2].ref().is_shape_tensor(),
        "Synapse input3 type expected to be shape tensor");
    TORCH_CHECK(
        p_context_->syn_inputs_[3].ref().is_shape_tensor(),
        "Synapse input4 type expected to be shape tensor");

    std::vector<int64_t> shape = p_context_->syn_inputs_[1].ref().pt_shape();
    auto inp_shape = self.sizes().vec();
    // out_shape = shape of the slice to be inserted
    auto out_shape = inputs[1].toTensor().sizes().vec();
    auto step = inputs[2].toTensor().sizes().vec();
    auto start = inputs[3].toTensor().sizes().vec();

    if ((habana::ShapeInference::GetCurrentPass() ==
         habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
        (habana::ShapeInference::GetMaxPolicyInUse() ==
         habana_helpers::DynamicDimsPolicy::CALCULATED)) {

      std::vector<int64_t> min, max;
      synapse_helpers::tensor& syn_tensor_start = p_context_->syn_inputs_[3];
      std::tie(min, max) =
          habana::ShapeInference::GetMinMaxShape(syn_tensor_start.id());

      SliceInsertOperator::UpdateMaxPassSliceInputs(inp_shape, out_shape, step, start, min, max);

      // Modify the start and output shape in name shape map to create valid ranges
      synapse_helpers::tensor& syn_tensor_output = p_context_->syn_inputs_[1];
      habana::ShapeInference::UpdateShapeInfo(
          graph, syn_tensor_output.id(), out_shape);
      habana::ShapeInference::UpdateShapeInfo(
          graph, syn_tensor_start.id(), start);
      shape = out_shape;
    }

    ValidateSliceInsertInputs(inp_shape, out_shape, step, start);
  } else if (has_shape_tensor && inputs.size() == 3) {
    TORCH_CHECK(
        p_context_->syn_inputs_[2].ref().is_host_to_device_tensor(),
        "Synapse input3 type expected to be host to device tensor");
    auto inp_shape = self.sizes().vec();
    auto out_shape = inputs[1].toTensor().sizes().vec();
    auto host_tensor = inputs[2].toTensor();
    auto params_vec = SliceOperator::ComputeParamsfromH2DTensor(host_tensor);

    std::vector<int64_t> start, step;
    start = SliceOperator::get_start_tensor(params_vec);
    step = SliceOperator::get_step_tensor(params_vec);
    ValidateSliceInsertInputs(inp_shape, out_shape, step, start);
  } else {
    TORCH_CHECK(
        inputs.size() == 3,
        "Incorrect size of inputs expected for slice operator");
    TORCH_CHECK(
        inputs[2].isIntList(),
        "Input slice params type expected to be integer list");
  }
  std::vector<int64_t> shape = self.sizes().vec();

  auto& mdata = output_metadata.at(0);
  Tensor output;
  if (!graph.is_dry_run() && mdata.allocated_tensor.has_value()) {
    output = mdata.allocated_tensor.value();
    AllocateSynapseOutput(graph, output, mdata);
  } else {
    output = habana::createPTTensor(
        self,
        shape,
        self.options(),
        self.suggest_memory_format(),
        output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  }

  if (has_shape_tensor) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    // Allocate Shape tensor
    // DS for select_scatter and slice_scatter op leverages
    // DS support for slice_insert with shape tensor only
    // Adds a new shape tensor to graph builder context with context params
    // Returns address of created syn_tensor (no need to explicitly capture this)
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, output, SHAPE_TENSOR);
    }
    auto paramsList = inputs[2].toIntList();

    synSliceParamsV2 params;
    ComputeParams(params, self, paramsList, graph);
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}
// DS slice_scatter(Tensor self, Tensor src, Tensor dim = None, Tensor?
// start=None, Tensor? end = None, Tensor step) -> Tensor
void SliceScatterOperatorDSUtil::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {

  // slice_scatter op with dynamic shape enabled has 6 parameters:
  // input: Tensor
  // src: Tensor
  // dim: Shared tensor
  // start: Shared tensor
  // end: Shared tensor
  // step: Shared tensor
  auto insert_t = inputs[1].toTensor();
  int64_t dim = inputs[2].toTensor().sizes().vec()[0];
  int64_t start = inputs[3].toTensor().sizes().vec()[0];
  int64_t end = inputs[4].toTensor().sizes().vec()[0];
  int64_t step = inputs[5].toTensor().sizes().vec()[0];
  Stack inputs_mod = {
      inputs[0],
      inputs[1],
      IValue(dim),
      IValue(start),
      IValue(end),
      IValue(step)};
  // Use original slice_scatter op with extracted scalar values
  auto slicescatterOp = make_operator<SliceScatterOperator>(
      insert_t.device().index(), insert_t.scalar_type());
  slicescatterOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  slicescatterOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  slicescatterOp->AllocateAndAddSynapseNode(graph, inputs_mod, output_metadata);
  synapse_helpers::tensor& slice_scatter_out =
      slicescatterOp->GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(slice_scatter_out);
  p_context_->pt_outputs_.emplace_back(slicescatterOp->GetOutputs()[0]);

}
// slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt?
// end=None, SymInt step=1) -> Tensor
void SliceScatterOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto dim = inputs[2].toInt();
  auto start_opt = inputs[3].to<c10::optional<int64_t>>();
  auto end_opt = inputs[4].to<c10::optional<int64_t>>();
  int64_t start = start_opt.value_or(0);
  int64_t end = end_opt.value_or(INT64_MAX);
  auto step = inputs[5].toInt();
  c10::List params = {dim, start, end, step};
  Stack inputs_mod = {inputs[0], inputs[1], IValue(params)};
  SliceInsertOperator::AllocateAndAddSynapseNode(
      graph, inputs_mod, output_metadata);
}

// func: select_scatter(Tensor self, Tensor src, SymInt? dim, SymInt index) -> Tensor
// select_scatter supports dynamic shape (DS) using shape tensors
void SelectScatterOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  // select = slice -> squeeze
  // consequently select_scatter = unsqueeze -> slice_scatter
  // slice_scatter = slice_insert
  // There are 4 inputs: input, src, dim, index
  auto insert_t = inputs[1].toTensor();

  // dim is the third input
  // dim is a shape tensor if dynamic shape is enabled
  bool is_dim_shape_tensor = inputs[2].isTensor();
  int dim = 0;
  if (is_dim_shape_tensor)
      dim = inputs[2].toTensor().sizes().vec()[0];
  else
      // dim is integer if dynamic shape is disabled or
      // the first iteration if DS is enabled
      dim = inputs[2].toInt();

  // Unsqueeze requires src and dim
  auto unsqueezeOp = make_operator<UnsqueezeOperator>(
      insert_t.device().index(), insert_t.scalar_type());
  unsqueezeOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  Stack unsqueeze_inputs = {IValue(insert_t), IValue(dim)};
  unsqueezeOp->AllocateAndAddSynapseNode(
      graph, unsqueeze_inputs, OutputMetaDataVector(1));

  // index is the fourth input
  // index is a shape tensor if dynamic shape is enabled
  int index = 0;
  bool is_index_shape_tensor = inputs[3].isTensor();
  if (is_index_shape_tensor)
      index = inputs[3].toTensor().sizes().vec()[0];
  else
      // index is integer if dynamic shape is disabled or
      // the first iteration if DS is enabled
      index = inputs[3].toInt();
  int64_t start = index;
  int64_t end = index + 1;
  int64_t step = 1;
  Stack inputs_mod = {
      inputs[0],
      IValue(unsqueezeOp->GetOutputs()[0]),
      IValue(dim),
      IValue(start),
      IValue(end),
      IValue(step)};

  // slice_scatter requires input, output of unsqueeze op,
  // start, end, step
  auto slicescatterOp = make_operator<SliceScatterOperator>(
      insert_t.device().index(), insert_t.scalar_type());
  slicescatterOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  slicescatterOp->SetSynapseInput(unsqueezeOp->GetSynOutputs()[0]);
  slicescatterOp->AllocateAndAddSynapseNode(graph, inputs_mod, output_metadata);
  synapse_helpers::tensor& slice_scatter_out =
      slicescatterOp->GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(slice_scatter_out);
  p_context_->pt_outputs_.emplace_back(slicescatterOp->GetOutputs()[0]);
}

bool StridedInsertOperator::verifyViewMemoryAccess(
    const at::Tensor& real,
    const at::Tensor& view,
    IntArrayRef& strides,
    int64_t& offset) {
  auto rv = real.sizes().vec();
  const uint64_t realTensorElements =
      std::accumulate(rv.begin(), rv.end(), 1, std::multiplies<unsigned>());
  if (realTensorElements == 0) {
    return true;
  }
  uint64_t lastElementOffset = 0;
  for (unsigned d = 0; d < view.dim(); d++) {
    if (view.sizes()[d] == 0) {
      return true;
    }
    lastElementOffset += strides[d] * (view.sizes()[d] - 1);
  }
  if (offset + lastElementOffset >= realTensorElements) {
    return false;
  }
  return true;
}

namespace {
// For Memory Reuse case even though stack size if 4
// would not mean tensor[3] is strides, in that case
// need to check if tensor[3] is shape tensor to be sure
bool HasFrontendStrides(const torch::jit::Stack& inputs) {
  bool frontend_stride = false;
  if (inputs.size() >= 4) {
    auto tensor = inputs[3].toTensor();
    auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
    if (impl && impl->isShapeTensor()) {
      frontend_stride = true;
    }
  }
  PT_DYNAMIC_SHAPE_DEBUG("HasFrontendStrides returning ", frontend_stride);
  return frontend_stride;
}

bool IsStridesRatioUsed(const torch::jit::Stack& inputs) {
  bool stride_ratio_used = false;
  if (inputs.size() >= 4) {
    auto offset_t = inputs[3].toTensor();

    // Offset shape tensor is created only in case the ratio is used
    auto tmeta_offset{get_tensor_extra_meta(offset_t)};
    auto stride_ratios = tmeta_offset->get_shape_struct().get_stride_ratios();
    if (stride_ratios.size() > 0) {
      stride_ratio_used = true;
    }
  }

  return stride_ratio_used;
}

size_t GetMInMaxSifOffset(bool dry_run, size_t data_size) {
  size_t sif_offset = 0;

  if (dry_run &&
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    sif_offset = data_size;
  }

  return sif_offset;
}

std::vector<int64_t> GetAsStridedOperatorStrideData(
    at::Tensor& stride_t,
    bool dry_run) {
  std::vector<int64_t> strides;
  size_t data_size = stride_t.sizes()[0];

  auto tmeta{get_tensor_extra_meta(stride_t)};
  habana::HostDataType h2d_dt_type = tmeta->get_host_dt_type();
  void* host_ptr = nullptr;
  if (dry_run) {
    host_ptr = tmeta->get_compile_host_ptr();
  } else {
    host_ptr = tmeta->get_host_ptr();
  }

  if (h2d_dt_type == habana::HostDataType::INT32_T) {
    int32_t* h2d_data = static_cast<int32_t*>(host_ptr);
    size_t sif_offset = GetMInMaxSifOffset(dry_run, data_size);
    h2d_data = h2d_data + sif_offset;
    for (size_t i = 0; i < data_size; i++) {
      strides.push_back(static_cast<int64_t>(*h2d_data++));
    }
  } else if (h2d_dt_type == habana::HostDataType::UINT32_T) {
    uint32_t* h2d_data = static_cast<uint32_t*>(host_ptr);
    size_t sif_offset = GetMInMaxSifOffset(dry_run, data_size);
    h2d_data = h2d_data + sif_offset;
    for (size_t i = 0; i < data_size; i++) {
      strides.push_back(static_cast<int64_t>(*h2d_data++));
    }
  } else if (h2d_dt_type == habana::HostDataType::UINT64_T) {
    uint64_t* h2d_data = static_cast<uint64_t*>(host_ptr);
    data_size = data_size / 2;
    size_t sif_offset = GetMInMaxSifOffset(dry_run, data_size);
    h2d_data = h2d_data + sif_offset;
    for (size_t i = 0; i < data_size; i++) {
      uint64_t h2d_elem = *h2d_data++;
      TORCH_CHECK(
          h2d_elem < LONG_MAX,
          "H2D data ",
          h2d_elem,
          " exceeds the int64 limit");
      strides.push_back(static_cast<int64_t>(h2d_elem));
    }
  } else {
    PT_DYNAMIC_SHAPE_DEBUG("Host datatype Not Supported");
  }

  return strides;
}

std::vector<int64_t> GetStridedInsertOperatorH2DStrides(
    const torch::jit::Stack& inputs,
    bool is_dry_run) {
  std::vector<int64_t> strides;
  at::Tensor stride_t = inputs[2].toTensor();

  if (!IsStridesRatioUsed(inputs)) {
    strides = GetAsStridedOperatorStrideData(stride_t, is_dry_run);
  } else {
    auto offset_tensor = inputs[3].toTensor();
    auto impl = habana_lazy::GetHbInternalTensorImpl(offset_tensor);
    HABANA_ASSERT(impl, "impl is invalid");
    // if it is MIN or MAX pass we need to manipulate the srides
    // otherwise pass the strides coming from frontend.
    if (is_dry_run &&
        (habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MIN_SHAPE ||
         habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
      auto orig_t = inputs[0].toTensor();
      auto orig_strides = HabanaOperator::CalculateStrides(
          orig_t.sizes(), orig_t.suggest_memory_format());
      auto stride_ratios = impl->get_shape_struct().get_stride_ratios();
      auto len = stride_ratios.size();
      for (uint64_t i = 0; i < len; i++) {
        strides.push_back(orig_strides[i] * stride_ratios[i]);
      }
    } else {
      strides = impl->get_shape_struct().get_stride_shape();
    }
  }

  return strides;
}

std::vector<int64_t> GetStridedViewOperatorH2DStrides(
    torch::jit::Stack& inputs,
    bool graph_dry_run) {
  std::vector<int64_t> strides;
  at::Tensor stride_t = inputs[2].toTensor();

  if (!IsStridesRatioUsed(inputs)) {
    strides = GetAsStridedOperatorStrideData(stride_t, graph_dry_run);
  } else {
    auto offset_t = inputs[3].toTensor();
    auto input_t = inputs[0].toTensor();
    auto impl = get_tensor_extra_meta(offset_t);
    if (graph_dry_run &&
        (habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MIN_SHAPE ||
         habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
      auto self_strides = input_t.strides().vec();
      auto stride_ratios = impl->get_shape_struct().get_stride_ratios();
      auto len = stride_ratios.size();
      for (uint64_t i = 0; i < len; i++) {
        strides.push_back(self_strides[i] * stride_ratios[i]);
      }
    } else {
      strides = impl->get_shape_struct().get_stride_shape();
    }
  }

  return strides;
}

std::vector<int64_t> GetStridedInsertOperatorStrides(
    const torch::jit::Stack& inputs,
    bool is_dry_run) {
  std::vector<int64_t> strides;

  if (HasFrontendStrides(inputs)) {
    strides = inputs[2].toTensor().sizes().vec();
  } else {
    auto offset_st = inputs[2].toTensor();
    auto impl = habana_lazy::GetHbInternalTensorImpl(offset_st);
    HABANA_ASSERT(impl, "impl is invalid");
    // if it is MIN or MAX pass we need to manipulate the srides
    // otherwise pass the strides coming from frontend.
    if (is_dry_run &&
        (habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MIN_SHAPE ||
         habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
      auto orig_t = inputs[0].toTensor();
      auto insert_t = inputs[1].toTensor();
      auto orig_strides = HabanaOperator::CalculateStrides(
          orig_t.sizes(), orig_t.suggest_memory_format());
      auto stride_ratios = impl->get_shape_struct().get_stride_ratios();
      auto len = stride_ratios.size();
      for (uint64_t i = 0; i < len; i++) {
        strides.push_back(orig_strides[i] * stride_ratios[i]);
      }
    } else {
      strides = impl->get_shape_struct().get_stride_shape();
    }
  }
  return strides;
}
} // namespace

void StridedInsertOperator::compute_params_h2d(
    HabanaOperator& hop,
    synStridedOpParams& params,
    const Stack& inputs,
    synapse_helpers::graph& graph) {
  std::fill_n(params.strides, HABANA_DIM_MAX, 0);
  auto orig_t = inputs[0].toTensor();
  auto insert_t = inputs[1].toTensor();
  std::vector<int64_t> strides;
  int64_t offset = 0;

  std::vector<int64_t> stride_values;
  strides = GetStridedInsertOperatorH2DStrides(inputs, graph.is_dry_run());
  if (!IsStridesRatioUsed(inputs)) {
    uint64_t num_stride = strides[0];
    offset = strides[1];

    for (uint64_t i = 0; i < num_stride; i++) {
      stride_values.push_back(static_cast<int64_t>(strides[i + 2]));
    }
    PT_DYNAMIC_SHAPE_DEBUG(
        "Backend orig tensor = ",
        orig_t.sizes().vec(),
        " insert tensor = ",
        insert_t.sizes().vec(),
        "strides = ",
        strides,
        " offset = ",
        offset);

    params.baseOffset = static_cast<uint64_t>(offset);
    size_t idx = 0;
    for (auto it = stride_values.begin(); it != stride_values.end(); ++it) {
      params.strides[idx] = static_cast<uint64_t>(*it);
      idx++;
    }

    std::reverse(stride_values.begin(), stride_values.end());
  } else {
    auto stride_tensor = inputs[2].toTensor();
    auto offset_tensor = inputs[3].toTensor();
    offset = offset_tensor.sizes()[0];
    std::vector<uint64_t> stride_data_vec;
    stride_data_vec.push_back(static_cast<uint64_t>(strides.size()));
    stride_data_vec.push_back(static_cast<uint64_t>(offset));
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
      stride_data_vec.push_back(static_cast<uint64_t>(*it));
    }
    size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - strides.size();
    for (size_t i = 0; i < fill_dim; i++) {
      stride_data_vec.push_back(static_cast<uint64_t>(0));
    }

    auto tmeta{get_tensor_extra_meta(stride_tensor)};
    if (habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MIN_SHAPE) {
      tmeta->set_min<uint64_t>(stride_data_vec);
    } else if (
        habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MAX_SHAPE) {
      tmeta->set_max<uint64_t>(stride_data_vec);
    }

    const auto& end = hop.GetSynInputs().end();
    hop.GetSynInputs().erase(end - 1, end);

    params.baseOffset = static_cast<uint64_t>(offset);
    size_t idx = 0;
    // synapse expects strides in reverse order
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
      params.strides[idx] = static_cast<uint64_t>(*it);
      idx++;
    }

    for (auto it = strides.begin(); it != strides.end(); ++it) {
      stride_values.push_back(static_cast<int64_t>(*it));
    }
  }
  // For dynamic min-max inference, validate the mem access of
  // elements. If the calculation dosen't match, fail here for inference
  // fallback to kick in. if GC compile fails, the fallback penalty is huge.
  // TODO: Currently GC have strict boundary check min and max shapes
  // in case of H2D
  if (!graph.is_dry_run() ||
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::OUTPUT_SHAPE) {
    IntArrayRef strides_ref(stride_values.data(), stride_values.size());
    bool memAccessCheck = verifyViewMemoryAccess(
        inputs[0].toTensor(), inputs[1].toTensor(), strides_ref, offset);
    TORCH_CHECK(
        inputs[0].toTensor().numel() == 0 || memAccessCheck,
        "Strided Insert will access memory outside of original tensor range!");
  }
}

void StridedInsertOperator::compute_params(
    HabanaOperator& hop,
    synStridedOpParams& params,
    const Stack& inputs,
    synapse_helpers::graph& graph) {
  std::fill_n(params.strides, HABANA_DIM_MAX, 0);
  auto orig_t = inputs[0].toTensor();
  auto insert_t = inputs[1].toTensor();
  std::vector<int64_t> strides;
  int64_t offset = 0;

  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
      StridedInsertOperator::compute_params_h2d(hop, params, inputs, graph);
      return;
    }

    TORCH_CHECK(hop.GetSynInputs()[2].ref().is_shape_tensor());
    strides = GetStridedInsertOperatorStrides(inputs, graph.is_dry_run());
    IntArrayRef strides_ref(strides.data(), strides.size());
    if (HasFrontendStrides(inputs)) {
      auto offset_tensor = inputs[3].toTensor();
      offset = offset_tensor.sizes()[0];
    } else {
      auto offset_tensor = inputs[2].toTensor();
      offset = offset_tensor.sizes()[0];
      auto syn_shape_input = habana_helpers::create_shape_tensor_backend(
          strides_ref,
          orig_t.device().index(),
          graph,
          false,
          SHAPE_TENSOR,
          hop.GetOpDynamicity(),
          "",
          nullptr);
      syn_shape_input.set_intermediate_shape_tensor();
      // Need to insert strides before offset
      // Before: orig, insert, offset
      // After : orig, insert, strides, offset
      hop.GetSynInputs().emplace(
          hop.GetSynInputs().begin() + 2, std::move(syn_shape_input));
    }
    PT_DYNAMIC_SHAPE_DEBUG(
        "Backend orig tensor = ",
        orig_t.sizes().vec(),
        " insert tensor = ",
        insert_t.sizes().vec(),
        "strides = ",
        strides,
        " offset = ",
        offset);
    // For dynamic min-max inference, validate the mem access of
    // elements. If the calculation dosen't match, fail here for inference
    // fallback to kick in. if GC compile fails, the fallback penalty is huge.
    // Since GC has relaxed memory access check for min/max only have the check
    // for actual
    if (!graph.is_dry_run() ||
        habana::ShapeInference::GetCurrentPass() ==
            habana::ShapeInfo::InferencePass::OUTPUT_SHAPE) {
      bool memAccessCheck = verifyViewMemoryAccess(
          inputs[0].toTensor(), inputs[1].toTensor(), strides_ref, offset);
      TORCH_CHECK(
          inputs[0].toTensor().numel() == 0 || memAccessCheck,
          "Strided Insert will access memory outside of original tensor range!");
    }
  } else {
    strides = inputs[2].toIntVector();
    offset = inputs[3].toInt();
  }

  if (!have_shape_tensors) {
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      hop.AllocateSynapseShapeTensor(graph, orig_t);
      // For Dynamic case fill strides/offset params with max size
      if (!graph.is_dry_run()) {
        synapse_helpers::tensor& stride_tensor = hop.GetSynInputs()[2];
        std::vector<int64_t> min, max;
        std::tie(min, max) =
            habana::ShapeInference::GetMinMaxShape(stride_tensor.id());
        strides = max;
        synapse_helpers::tensor& offset_tensor = hop.GetSynInputs()[3];
        std::tie(min, max) =
            habana::ShapeInference::GetMinMaxShape(offset_tensor.id());
        if (max.size()) {
          offset = max[0];
        }
      }
    }
  }

  params.baseOffset = static_cast<uint64_t>(offset);

  size_t idx = 0;
  // synapse expects strides in reverse order
  for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
    params.strides[idx++] = static_cast<uint64_t>(*it);
  }
}

static void populateStridedOpParams(
    torch::jit::Stack& inputs,
    InferOutputMetaRetType& out) {
  std::vector<int64_t> strides = inputs[2].toIntVector();
  int64_t offset = inputs[3].isNone() ? 0 : inputs[3].toInt();
  synStridedOpParams params;
  std::fill_n(params.strides, HABANA_DIM_MAX, 0);
  params.baseOffset = static_cast<uint64_t>(offset);
  size_t idx = 0;
  // synapse expects strides in reverse order
  for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
    params.strides[idx++] = static_cast<uint64_t>(*it);
  }
  PT_BRIDGE_DEBUG("strides - ", strides, " offset - ", offset);
  PT_BRIDGE_DEBUG("params add - ", &params, " size - ", sizeof(params));
  out.AddNodeParams(&params, sizeof(params));
}

InferOutputMetaRetType StridedInsertOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto orig_t = inputs[0].toTensor();

  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      orig_t.sizes().vec(),
      HabanaOperator::CalculateStrides(
          orig_t.sizes(), orig_t.suggest_memory_format()),
      orig_t.scalar_type(),
      orig_t.suggest_memory_format()));
  bool have_shape_tensors = inputs[2].isTensor();
  bool is_as_strided_h2d =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED);
  if (have_shape_tensors && !HasFrontendStrides(inputs) && !is_as_strided_h2d) {
    auto strides = GetStridedInsertOperatorStrides(inputs, true);
    auto stride_meta_data = TensorMetaData(
        strides, strides, orig_t.scalar_type(), orig_t.suggest_memory_format());
    out.AddShapeTensor(stride_meta_data);
  }

  // Node params for eager mode
  if (GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER) {
    populateStridedOpParams(inputs, out);
  }

  return out;
}

void StridedInsertOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() >= 3,
      "Incorrect number of arguments for strided insert op");

  synStridedOpParams params;
  compute_params(*this, params, inputs, graph);

  auto orig_t = inputs[0].toTensor();

  PT_EAGER_INFO("Input: ", habana_helpers::DebugString(orig_t));

  auto& mdata = output_metadata.at(0);
  if (!graph.is_dry_run() && mdata.allocated_tensor.has_value()) {
    AllocateSynapseOutput(graph, mdata.allocated_tensor.value(), mdata);
  } else {
    auto output = habana::createPTTensor(
        orig_t,
        orig_t.sizes(),
        orig_t.options(),
        orig_t.suggest_memory_format(),
        mdata.persistent);
    AllocateSynapseOutput(graph, output, mdata);
  }
  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void StridedInsertOperator::ReuseMemoryAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() >= 4,
      "Incorrect number of arguments for strided insert op");
  // orig, insert, offset, graph_input
  auto graph_input = inputs.back().toTensor();
  auto orig_t = inputs[0].toTensor();
  TORCH_CHECK(graph_input.sizes() == orig_t.sizes(), "incorrect graph input");

  struct synStridedOpParams params;
  compute_params(*this, params, inputs, graph);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          syn_t_vec[0], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(graph_input);

  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

// Non-DS op has 5 input arguments
// DS op has either 4 or 3 input arguments (2 variants)
// as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride,
// SymInt? storage_offset=None) -> Tensor
// OR
// DS as_strided_scatter(Tensor self, Tensor src, Tensor stride, Tensor? storage_offset = None) -> Tensor
// OR
// DS as_strided_scatter_orig(Tensor self, Tensor src, Tensor stride) -> Tensor
void AsStridedScatterOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() >= 3,
      "Incorrect number of arguments for AsStridedScatterOperator op");

  // DS variant
  // Directly pass input to strided_insert op
  if (inputs.size() <= 4 && inputs[2].isTensor()) {
      StridedInsertOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
      return;
  }

  // Non-DS variant
  auto storage_offset_opt = inputs[4].to<c10::optional<int64_t>>();
  auto storage_offset =
      storage_offset_opt.value_or(inputs[0].toTensor().storage_offset());
  Stack inputs_mod = {inputs[0], inputs[1], inputs[3], IValue(storage_offset)};

  StridedInsertOperator::AllocateAndAddSynapseNode(
      graph, inputs_mod, output_metadata);
}

// TODO CPU backend is not reusing the memory for as_strided_scatter. Is PT ok
// with memory reuse for other backends?
// AsStridedScatterOperator::ReuseMemoryAndAddSynapseNode()

bool StridedViewOperator::verifyViewMemoryAccess(
    at::Tensor& real,
    at::Tensor& view,
    IntArrayRef& strides,
    int64_t& offset) {
  auto rv = real.sizes().vec();
  const uint64_t realTensorElements =
      std::accumulate(rv.begin(), rv.end(), 1, std::multiplies<unsigned>());
  if (realTensorElements == 0) {
    return true;
  }
  uint64_t lastElementOffset = 0;
  for (unsigned d = 0; d < view.dim(); d++) {
    if (view.sizes()[d] == 0) {
      return true;
    }
    lastElementOffset += strides[d] * (view.sizes()[d] - 1);
  }
  if (offset + lastElementOffset >= realTensorElements) {
    return false;
  }
  return true;
}

namespace {
std::vector<int64_t> GetStridedViewOperatorStrides(
    torch::jit::Stack& inputs,
    bool graph_dry_run) {
  std::vector<int64_t> size, strides;
  auto self = inputs[0].toTensor();
  auto size_st = inputs[1].toTensor();
  if (HasFrontendStrides(inputs)) {
    strides = inputs[2].toTensor().sizes().vec();
  } else {
    auto impl = habana_lazy::GetHbInternalTensorImpl(size_st);
    if (graph_dry_run &&
        (habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MIN_SHAPE ||
         habana::ShapeInference::GetCurrentPass() ==
             habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
      auto self_strides = self.strides().vec();
      auto stride_ratios = impl->get_shape_struct().get_stride_ratios();
      auto len = stride_ratios.size();
      for (uint64_t i = 0; i < len; i++) {
        strides.push_back(self_strides[i] * stride_ratios[i]);
      }
    } else {
      strides = impl->get_shape_struct().get_stride_shape();
    }
  }
  return strides;
}
} // namespace

InferOutputMetaRetType StridedViewOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> size;
  std::vector<int64_t> strides;

  bool have_shape_tensors = inputs[1].isTensor();
  bool is_as_strided_h2d =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED);
  if (have_shape_tensors) {
    size = inputs[1].toTensor().sizes().vec();
    if (is_as_strided_h2d) {
      std::vector<int64_t> h2d_strides;
      h2d_strides = GetStridedViewOperatorH2DStrides(inputs, true);
      if (!IsStridesRatioUsed(inputs)) {
        size_t num_strides = h2d_strides[0];
        for (size_t i = 0; i < num_strides; i++) {
          strides.push_back(h2d_strides[2 + i]);
        }
      } else {
        strides = h2d_strides;
      }
    } else {
      strides = GetStridedViewOperatorStrides(inputs, true);
    }
  } else {
    size = inputs[1].toIntVector();
    strides = inputs[2].toIntVector();
  }

  InferOutputMetaRetType out;
  auto tensor_meta_data = TensorMetaData(
      size, strides, self.scalar_type(), self.suggest_memory_format());
  out.AddOutputTensor(tensor_meta_data);

  if (!have_shape_tensors) {
    out.AddShapeTensor(tensor_meta_data);
  } else if (
      have_shape_tensors && !HasFrontendStrides(inputs) && !is_as_strided_h2d) {
    auto stride_meta_data = TensorMetaData(
        strides, strides, self.scalar_type(), self.suggest_memory_format());
    out.AddShapeTensor(stride_meta_data);
  }

  // skip meta op as it does not need to add syn node and node params
  int meta_op = 0;
  if (inputs.size() == 5) {
    meta_op = inputs[4].toInt();
  }

  // Node params for eager mode
  if (!meta_op &&
      GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER) {
    populateStridedOpParams(inputs, out);
  }

  return out;
}

void StridedViewOperator::compute_params_h2d(
    synStridedOpParams& params,
    Stack& inputs,
    synapse_helpers::graph& graph,
    std::vector<int64_t>& size,
    std::vector<int64_t>& strides,
    int64_t& offset) {
  params = static_cast<synStridedOpParams>(params);
  std::fill_n(params.strides, HABANA_DIM_MAX, 0);
  std::vector<int64_t> stride_values;
  TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_shape_tensor());
  size = p_context_->syn_inputs_[1].ref().pt_shape();
  strides = GetStridedViewOperatorH2DStrides(inputs, graph.is_dry_run());

  if (!IsStridesRatioUsed(inputs)) {
    auto stride_tensor = inputs[2].toTensor();
    uint64_t num_stride = strides[0];
    offset = strides[1];

    for (uint64_t i = 0; i < num_stride; i++) {
      stride_values.push_back(static_cast<int64_t>(strides[i + 2]));
    }

    params.baseOffset = static_cast<uint64_t>(offset);
    size_t idx = 0;
    // synapse expects strides in reverse order
    for (auto it = stride_values.begin(); it != stride_values.end(); ++it) {
      params.strides[idx] = static_cast<uint64_t>(*it);
      idx++;
    }
  } else {
    auto stride_tensor = inputs[2].toTensor();
    auto offset_tensor = inputs[3].toTensor();
    offset = offset_tensor.sizes()[0];
    std::vector<uint64_t> stride_data_vec;
    stride_data_vec.push_back(static_cast<uint64_t>(strides.size()));
    stride_data_vec.push_back(static_cast<uint64_t>(offset));
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
      stride_data_vec.push_back(static_cast<uint64_t>(*it));
    }
    size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - strides.size();
    for (size_t i = 0; i < fill_dim; i++) {
      stride_data_vec.push_back(static_cast<uint64_t>(0));
    }

    auto tmeta{get_tensor_extra_meta(stride_tensor)};
    if (habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MIN_SHAPE) {
      tmeta->set_min<uint64_t>(stride_data_vec);
    } else if (
        habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MAX_SHAPE) {
      tmeta->set_max<uint64_t>(stride_data_vec);
    }

    const auto& end = p_context_->syn_inputs_.end();
    p_context_->syn_inputs_.erase(end - 1, end);

    params.baseOffset = static_cast<uint64_t>(offset);
    size_t idx = 0;
    // synapse expects strides in reverse order
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
      params.strides[idx] = static_cast<uint64_t>(*it);
      idx++;
    }
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
      stride_values.push_back(static_cast<int64_t>(*it));
    }
  }
  // For dynamic min-max inference, validate the mem access of
  // elements. If the calculation dosen't match, fail here for inference
  // fallback to kick in. if GC compile fails, the fallback penalty is huge.
  // Since GC has relaxed memory access check for min/max only have the check
  // for actual
  if (!graph.is_dry_run() ||
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::OUTPUT_SHAPE) {
    std::reverse(stride_values.begin(), stride_values.end());
    IntArrayRef strides_ref(stride_values.data(), stride_values.size());
    bool memAccessCheck = verifyViewMemoryAccess(
        inputs[0].toTensor(), inputs[1].toTensor(), strides_ref, offset);
    TORCH_CHECK(
        inputs[0].toTensor().numel() == 0 || memAccessCheck,
        "Strided View will access memory outside of original tensor range!");
  }

  // TODO  For Dynamic case fill strides/offset params with max size
  strides.clear();
  for (auto it = stride_values.rbegin(); it != stride_values.rend(); ++it) {
    strides.push_back(static_cast<uint64_t>(*it));
  }
}

/*************************************************************************
 * @brief Kernel implementation for strided view , used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void StridedViewOperator::compute_params(
    synStridedOpParams& params,
    Stack& inputs,
    synapse_helpers::graph& graph,
    std::vector<int64_t>& size,
    std::vector<int64_t>& strides,
    int64_t& offset) {
  std::fill_n(params.strides, HABANA_DIM_MAX, 0);
  auto self = inputs[0].toTensor();
  offset = 0;

  bool have_shape_tensors = inputs[1].isTensor();
  if (have_shape_tensors) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
      StridedViewOperator::compute_params_h2d(
          params, inputs, graph, size, strides, offset);
      return;
    }

    TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_shape_tensor());
    size = p_context_->syn_inputs_[1].ref().pt_shape();
    strides = GetStridedViewOperatorStrides(inputs, graph.is_dry_run());
    IntArrayRef strides_ref(strides.data(), strides.size());
    if (HasFrontendStrides(inputs)) {
      auto offset_tensor = inputs[3].toTensor();
      offset = offset_tensor.sizes()[0];
    } else {
      auto offset_tensor = inputs[2].toTensor();
      offset = offset_tensor.sizes()[0];
      auto syn_shape_input = habana_helpers::create_shape_tensor_backend(
          strides_ref,
          self.device().index(),
          graph,
          false,
          SHAPE_TENSOR,
          is_op_dynamic,
          "",
          nullptr);
      syn_shape_input.set_intermediate_shape_tensor();
      // Need to insert strides before offset
      // Before: orig, insert, offset
      // After : orig, insert, strides, offset
      p_context_->syn_inputs_.emplace(
          p_context_->syn_inputs_.begin() + 2, std::move(syn_shape_input));
    }
    // For dynamic min-max inference, validate the mem access of
    // elements. If the calculation dosen't match, fail here for inference
    // fallback to kick in. if GC compile fails, the fallback penalty is huge.
    // Since GC has relaxed memory access check for min/max only have the
    // check for actual
    if (!graph.is_dry_run() ||
        habana::ShapeInference::GetCurrentPass() ==
            habana::ShapeInfo::InferencePass::OUTPUT_SHAPE) {
      bool memAccessCheck = verifyViewMemoryAccess(
          inputs[0].toTensor(), inputs[1].toTensor(), strides_ref, offset);
      TORCH_CHECK(
          self.numel() == 0 || memAccessCheck,
          "Strided View will access memory outside of original tensor range!");
    }
  } else {
    TORCH_CHECK(
        inputs[1].isIntList(), "Input arg 1 needs to be of Int List type");
    TORCH_CHECK(
        inputs[2].isIntList(), "Input arg 2 needs to be of Int List type");
    size = inputs[1].toIntVector();
    strides = inputs[2].toIntVector();
    offset = inputs[3].isNone() ? 0 : inputs[3].toInt();
  }

  // For Dynamic case fill strides/offset params with max size
  if (graph.is_dynamic_graph() && have_shape_tensors) {
    synapse_helpers::tensor& stride_tensor = p_context_->syn_inputs_[2];
    std::vector<int64_t> min, max;
    std::tie(min, max) =
        habana::ShapeInference::GetMinMaxShape(stride_tensor.id());
    strides = max;
    synapse_helpers::tensor& offset_tensor = p_context_->syn_inputs_[3];
    std::tie(min, max) =
        habana::ShapeInference::GetMinMaxShape(offset_tensor.id());
    if (max.size()) {
      offset = max[0];
    }
  }

  // If shape tensors are not created at frontend we need to create
  // Shape tensor at backend and also pass the params. Otherwise no params are
  // required.
  if (!have_shape_tensors) {
    params.baseOffset = static_cast<uint64_t>(offset);
    size_t idx = 0;
    // synapse expects strides in reverse order
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
      params.strides[idx++] = static_cast<uint64_t>(*it);
    }
  }
}

void StridedViewOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  synStridedOpParams params;
  std::vector<int64_t> size, strides;
  int64_t offset;
  compute_params(params, inputs, graph, size, strides, offset);
  auto self = inputs[0].toTensor();
  int meta_op = 0;
  if (inputs.size() == 5) {
    meta_op = inputs[4].toInt();
  }

  at::Tensor output;
  auto& mdata = output_metadata.at(0);
  if (!graph.is_dry_run() && mdata.allocated_tensor.has_value()) {
    output = mdata.allocated_tensor.value();
  } else {
    output = habana::createPTTensor(
        self,
        size,
        self.options(),
        self.suggest_memory_format(),
        mdata.persistent);
  }
  AllocateSynapseOutput(graph, output, mdata);

  if (!meta_op) {
    // If shape tensors are not created at frontend we need to create
    // Shape tensor at backend and also pass the params. Otherwise no params are
    // required.
    bool have_shape_tensors = inputs[1].isTensor();
    if (!have_shape_tensors) {
      // Allocate Shape tensor
      if (graph.is_dynamic_graph()) {
        AllocateSynapseShapeTensor(graph, output);
      }
      AddNodeToSynapseGraph(graph, &params, sizeof(params));
    } else {
      AddNodeToSynapseGraph(graph, nullptr, 0);
    }
  }
}

void StridedViewOperator::ReuseMemoryAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
    const habana::OutputMetaDataVector& output_metadata) {
  synStridedOpParams params;
  std::vector<int64_t> sizes, strides;
  int64_t offset;
  compute_params(params, inputs, graph, sizes, strides, offset);
  auto self = inputs[0].toTensor();
  auto graph_input = inputs[inputs.size() - 1].toTensor();

  // params can have non-contiguous strides but tensors will have contiguous
  // strides as synapse node densifies
  auto strides_contig = strides;
  habana_helpers::recalc_strides(strides_contig, sizes);

  auto output = at::as_strided(graph_input, sizes, strides_contig, offset);

  auto syn_tensor_output =
      habana_helpers::duplicate_tensor_in_memory_section_with_size(
          syn_t_vec[0],
          graph,
          sizes,
          strides_contig,
          offset * graph_input.itemsize(),
          output_metadata.at(0).external);

  // This flag can be enabled in model scripts only if DDP
  // gradient_as_bucket_view = True
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRADIENT_VIEW_LAYOUT_OPT)) {
    syn_tensor_output.set_dont_allow_permute(true);
    auto smeta{habana::get_storage_extra_meta(output)};
    HABANA_ASSERT(smeta);
    smeta->set_dont_allow_permutation(true);
  }

  p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_output));
  p_context_->pt_outputs_.emplace_back(output);

  // If shape tensors are not created at frontend we need to create
  // Shape tensor at backend and also pass the params. Otherwise no params are
  // required.
  bool have_shape_tensors = inputs[1].isTensor();
  if (!have_shape_tensors) {
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, output);
    }
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  } else {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  }
}

static auto& BasicKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::habana_d2d_memcpy_other", KERNEL_FN_GLOBAL(MemCopyOperator))
        .add("hpu::control_edge_", KERNEL_FN_GLOBAL(DummyOperator))
        .add("hpu::as_strided_lazy_", KERNEL_FN_GLOBAL(AsStridedOperator))
        .add("hpu::strided_view", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_cl", KERNEL_FN_GLOBAL(StridedViewClOperator))
        .add("hpu::strided_view_ds", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_ds_h2d", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_cl_ds", KERNEL_FN_GLOBAL(StridedViewClOperator))
        .add("hpu::strided_view_out", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_orig_ds", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add(
            "hpu::strided_view_orig_ds_h2d",
            KERNEL_FN_GLOBAL(StridedViewOperator))
        .add(
            "hpu::strided_view_out_orig_ds_h2d",
            KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_out_ds", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add(
            "hpu::strided_view_out_ds_h2d",
            KERNEL_FN_GLOBAL(StridedViewOperator))
        .add(
            "hpu::strided_view_out_orig_ds",
            KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::slice_insert", KERNEL_FN_GLOBAL(SliceInsertOperator))
        .add("hpu::slice_insert_ds", KERNEL_FN_GLOBAL(SliceInsertOperator))
        .add("hpu::slice_insert_ds_ht", KERNEL_FN_GLOBAL(SliceInsertOperator))
	.add("hpu::slice_scatter_ds", KERNEL_FN_GLOBAL(SliceInsertOperator))
        .add("hpu::strided_insert", KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add("hpu::strided_insert_ds", KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add(
            "hpu::strided_insert_orig_ds",
            KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add(
            "hpu::strided_insert_orig_ds_h2d",
            KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add(
            "hpu::strided_insert_cl",
            KERNEL_FN_GLOBAL(StridedInsertClOperator))
        .add(
            "hpu::strided_insert_cl_ds",
            KERNEL_FN_GLOBAL(StridedInsertClOperator))
        .add(
            "hpu::as_strided_layout",
            KERNEL_FN_GLOBAL(AsStridedLayoutOperator))
        .add("hpu::identity", KERNEL_FN_GLOBAL(IdentityOperator))
        .add("aten::alias", KERNEL_FN_GLOBAL(IdentityOperator))
        .add("aten::as_strided", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("aten::slice_scatter", KERNEL_FN_GLOBAL(SliceScatterOperator))
        .add("hpu::slice_scatter", KERNEL_FN_GLOBAL(SliceScatterOperatorDSUtil))
        .add("aten::select_scatter", KERNEL_FN_GLOBAL(SelectScatterOperator))
        .add("hpu::select_scatter", KERNEL_FN_GLOBAL(SelectScatterOperator))
        .add("hpu::as_strided_scatter", KERNEL_FN_GLOBAL(AsStridedScatterOperator))
        .add("hpu::as_strided_scatter_orig", KERNEL_FN_GLOBAL(AsStridedScatterOperator))
        .add(
            "aten::as_strided_scatter",
            KERNEL_FN_GLOBAL(AsStridedScatterOperator));
