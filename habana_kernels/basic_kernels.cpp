/*******************************************************************************
 * Copyright (C) 2020-2022 Habana Labs, Ltd. an Intel Company
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
#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <synapse_api.h>
#include <torch/script.h>

#include <habana_device/PinnedMemoryAllocator.h>
#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_lazy/aten_lazy_bridge.h"

using namespace torch;
using namespace habana;

static void print_stride_warning(const Tensor& src, const Tensor& dst) {
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

static void do_d2d_copy(Tensor& dst, const Tensor& src_in, bool non_blocking) {
  // Nothing to do if copy is triggered with same src & dst addresses
  // it actually triggers an assert on func_sim if we trigger this DMA
  // therefore return without doing anything. (this case seen with Mask R-CNN
  // Detectron2 model when copying check-point weights)
  if (dst.data_ptr() == src_in.data_ptr()) {
    return;
  }

  // No direct support for Long in device
  bool same_type = (src_in.scalar_type() == dst.scalar_type());
  auto src = ((src_in.scalar_type() != c10::ScalarType::Long) || same_type)
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

// cpu->hpu and hpu->cpu copy implementation
Tensor& copy_hpu_(Tensor& self, const Tensor& src, bool non_blocking) {
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
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      TORCH_CHECK(src.is_contiguous(src.suggest_memory_format()));
      src_contiguous = src.to(dst.scalar_type());
    } else {
      src_contiguous =
          src.to(dst.scalar_type()).contiguous(src.suggest_memory_format());
    }

    TORCH_CHECK(dst.nbytes() >= src_contiguous.nbytes());
    habana_helpers::copy_data_to_device(src_contiguous, dst, non_blocking);
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
          src_contiguous, dst_intermediate, non_blocking);
      dst = dst_intermediate.to(dst.scalar_type());
    } else {
      // Is there any reason why this check cannot be strict equality?
      TORCH_CHECK(dst.nbytes() >= src_contiguous.nbytes());
      habana_helpers::copy_data_to_host(src_contiguous, dst, non_blocking);
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

void ToDtypeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  // This function can handle following 2 schemas only:
  // (1) to.device(Tensor self, Device device, ScalarType dtype, bool
  // non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) ->
  // (2) Tensor to.dtype(Tensor self, ScalarType dtype, bool
  // non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) ->
  // Tensor
  TORCH_CHECK(
      inputs.size() >= 5,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  if (inputs.size() == 6) {
    // Erase device information to unify subsequent code for both schemas.
    // Should be ok since we come here only for Habana device
    inputs.erase(inputs.cbegin() + 1);
  }

  auto self = inputs[0].toTensor();
  auto type = inputs[1].toScalarType();

  // Determine cast node_type to use based on src & dst dtypes
  std::string node_type;

  if ((type != self.scalar_type()) &&
      !((type == c10::ScalarType::Char &&
         self.scalar_type() == c10::ScalarType::Bool) ||
        (type == c10::ScalarType::Bool &&
         self.scalar_type() == c10::ScalarType::Char))) {
    std::pair<c10::ScalarType, c10::ScalarType> type_key{
        self.scalar_type(), type};

    auto node_type_opt{habana_helpers::direct_cast_guid(type_key)};
    HABANA_ASSERT(
        node_type_opt.has_value() &&
            "Unsupported Cast operation requested in ToDtypeOperator::AllocateAndAddSynapseNode",
        self.scalar_type(),
        " -> ",
        type);
    node_type = std::move(node_type_opt.value());
  } else {
    // Cases where a simple copy is being done (input_new = input) come as .to
    // call with same input & output data types. we add a identity node to
    // graph to handle this
    auto memcopyOp = make_operator<IdentityOperator>(
        self.device().index(), self.scalar_type());
    memcopyOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    torch::jit::Stack stack = {IValue(self)};
    memcopyOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(memcopyOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(memcopyOp->GetOutputs()[0]));
    return;
  }

  // we do not care about last 3 entries dtype conversion, so throw them away
  inputs.pop_back();
  inputs.pop_back();
  inputs.pop_back();

  auto Op = make_operator<CastOperator>(self.device().index(), node_type);
  Op->SetSynapseInput(p_context_->syn_inputs_[0]);
  Op->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  p_context_->syn_outputs_.emplace_back(std::move(Op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(Op->GetOutputs()[0]));
}

OutputShapeInfRetType MemCopyOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto output = inputs[(inputs.size() == 2) ? 1 : 0].toTensor();
  OutputShapeInfRetType out;
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
    // Important:
    // Conditions for calling 'duplicate_tensor_in_memory_section' below
    // must match conditions in 'inplaceInputId' function in
    // jitgraph_utils.cpp
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(output);
  } else {
    output =
        habana_helpers::createPTTensor(self, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  }
  p_context_->params_size_ = 0;
  AddNodeToSynapseGraph(graph, NULL, 0);
}

OutputShapeInfRetType IdentityOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto output = inputs[(inputs.size() == 2) ? 1 : 0].toTensor();
  OutputShapeInfRetType out;
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
  } else {
    output =
        habana_helpers::createPTTensor(self, output_metadata.at(0).persistent);
  }
  p_context_->params_size_ = 0;
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, NULL, 0);
}

/*************************************************************************
 * @brief Kernel implementation for dummy, used for graph ordering
 ************************************************************************/
OutputShapeInfRetType DummyOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  int out_index = inputs.size() - 1;
  auto output = inputs[out_index].toTensor();
  OutputShapeInfRetType out;
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
    compute_output_shape(
        const Tensor& self,
        IntArrayRef size,
        IntArrayRef stride) {
  std::vector<int64_t> out_size_vec;
  std::vector<int64_t> out_stride_vec;

  if (((self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast) ||
       (self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d)) &&
      (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING))) {
    if (size.size() == 4) {
      // NCHW -> NHWC
      const int64_t dim_pos_in[4] = {
          LayoutFormatDims::N,
          LayoutFormatDims::H,
          LayoutFormatDims::W,
          LayoutFormatDims::C};
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[dim_pos_in[idx]]);
        out_stride_vec.emplace_back(stride[dim_pos_in[idx]]);
      }
    } else if (size.size() == 5) {
      // NCDHW -> NDHWC
      const int64_t dim_pos_in[5] = {
          LayoutFormatWithDepthDims::N,
          LayoutFormatWithDepthDims::D,
          LayoutFormatWithDepthDims::H,
          LayoutFormatWithDepthDims::W,
          LayoutFormatWithDepthDims::C};
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[dim_pos_in[idx]]);
        out_stride_vec.emplace_back(stride[dim_pos_in[idx]]);
      }
    } else {
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[idx]);
        out_stride_vec.emplace_back(stride[idx]);
      }
    }
  } else {
    for (size_t idx = 0; idx < size.size(); idx++) {
      out_size_vec.emplace_back(size[idx]);
      out_stride_vec.emplace_back(stride[idx]);
    }
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

Tensor as_strided_hpu(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  // DeviceGuard omitted
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

Tensor view_hpu(const Tensor& self, IntArrayRef size) {
  // DeviceGuard omitted
  return at::native::view(self, size);
}

static inline Device ensure_has_index(c10::optional<at::Device> device) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl((*device).type());
  return impl->getDevice();
}

bool is_pinned_hpu(const Tensor& self, c10::optional<at::Device> device) {
  ensure_has_index(device);
  return habana::PinnedMemoryAllocator_is_pinned(self.storage().data());
}

Tensor pin_memory_hpu(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  ensure_has_index(device);
  auto* allocator = habana::getPinnedMemoryAllocator();
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

OutputShapeInfRetType SliceInsertOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> shape = self.sizes().vec();

  bool have_shape_tensor = inputs[2].isTensor();
  auto metaData = TensorMetaData(
      shape,
      HabanaOperator::CalculateStrides(shape, self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format());
  OutputShapeInfRetType out;
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
    synSliceParamsNDims params;
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
    synSliceParamsNDims& params,
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
    params.axes[i] = self.dim() - dim - 1;
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
      params.ends[i] = static_cast<int>(max[dim]);
    }
  }
}

void SliceInsertOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  auto self = inputs[0].toTensor();
  bool have_shape_tensor = inputs[2].isTensor();
  if (have_shape_tensor) {
    TORCH_CHECK(
        inputs.size() == 4,
        "Incorrect size of inputs expected for slice_insert operator");
    TORCH_CHECK(
        p_context_->syn_inputs_[2].ref().is_shape_tensor(),
        "Synapse input3 type expected to be shape tensor");
    TORCH_CHECK(
        p_context_->syn_inputs_[3].ref().is_shape_tensor(),
        "Synapse input4 type expected to be shape tensor");
  } else {
    TORCH_CHECK(
        inputs.size() == 3,
        "Incorrect size of inputs expected for slice operator");
    TORCH_CHECK(
        inputs[2].isIntList(),
        "Input slice params type expected to be integer list");
  }
  std::vector<int64_t> shape = self.sizes().vec();
  Tensor output = habana_helpers::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  if (have_shape_tensor) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, output);
    }
    auto paramsList = inputs[2].toIntList();

    synSliceParamsNDims params;
    ComputeParams(params, self, paramsList, graph);
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

bool StridedInsertOperator::verifyViewMemoryAccess(
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

// For Memory Reuse case even though stack size if 4
// would not mean tensor[3] is strides, in that case
// need to check if tensor[3] is shape tensor to be sure
bool HasFrontendStrides(torch::jit::Stack& inputs) {
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

std::vector<int64_t> GetStridedInsertOperatorStrides(
    torch::jit::Stack& inputs,
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

void StridedInsertOperator::compute_params(
    synStridedOpParams& params,
    Stack& inputs,
    synapse_helpers::graph& graph) {
  auto orig_t = inputs[0].toTensor();
  auto insert_t = inputs[1].toTensor();
  std::vector<int64_t> strides;
  int64_t offset = 0;

  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors) {
    TORCH_CHECK(p_context_->syn_inputs_[2].ref().is_shape_tensor());
    strides = GetStridedInsertOperatorStrides(inputs, graph.is_dry_run());
    IntArrayRef strides_ref(strides.data(), strides.size());
    if (HasFrontendStrides(inputs)) {
      auto offset_tensor = inputs[3].toTensor();
      offset = offset_tensor.sizes()[0];
    } else {
      auto offset_tensor = inputs[2].toTensor();
      offset = offset_tensor.sizes()[0];
      auto syn_shape_input = habana_helpers::create_shape_tensor(
          strides_ref,
          orig_t.device().index(),
          graph,
          false,
          SHAPE_TENSOR,
          "",
          nullptr);
      syn_shape_input.set_intermediate_shape_tensor();
      // Need to insert strides before offset
      // Before: orig, insert, offset
      // After : orig, insert, strides, offset
      p_context_->syn_inputs_.emplace(
          p_context_->syn_inputs_.begin() + 2, std::move(syn_shape_input));
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
      AllocateSynapseShapeTensor(graph, orig_t);
      // For Dynamic case fill strides/offset params with max size
      if (!graph.is_dry_run()) {
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
    }
  }

  params.baseOffset = static_cast<uint64_t>(offset);

  size_t idx = 0;
  // synapse expects strides in reverse order
  for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
    params.strides[idx++] = static_cast<uint64_t>(*it);
  }
}

OutputShapeInfRetType StridedInsertOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto orig_t = inputs[0].toTensor();

  OutputShapeInfRetType out;
  out.AddOutputTensor(TensorMetaData(
      orig_t.sizes().vec(),
      HabanaOperator::CalculateStrides(
          orig_t.sizes(), orig_t.suggest_memory_format()),
      orig_t.scalar_type(),
      orig_t.suggest_memory_format()));
  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors && !HasFrontendStrides(inputs)) {
    auto strides = GetStridedInsertOperatorStrides(inputs, true);
    auto stride_meta_data = TensorMetaData(
        strides, strides, orig_t.scalar_type(), orig_t.suggest_memory_format());
    out.AddShapeTensor(stride_meta_data);
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
  compute_params(params, inputs, graph);

  auto orig_t = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(
      orig_t,
      orig_t.sizes(),
      orig_t.options(),
      orig_t.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
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
  compute_params(params, inputs, graph);

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

OutputShapeInfRetType StridedViewOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> size;
  std::vector<int64_t> strides;

  bool have_shape_tensors = inputs[1].isTensor();
  if (have_shape_tensors) {
    size = inputs[1].toTensor().sizes().vec();
    strides = GetStridedViewOperatorStrides(inputs, true);
  } else {
    size = inputs[1].toIntVector();
    strides = inputs[2].toIntVector();
  }

  OutputShapeInfRetType out;
  auto tensor_meta_data = TensorMetaData(
      size, strides, self.scalar_type(), self.suggest_memory_format());
  out.AddOutputTensor(tensor_meta_data);

  if (!have_shape_tensors) {
    out.AddShapeTensor(tensor_meta_data);
  } else if (have_shape_tensors && !HasFrontendStrides(inputs)) {
    auto stride_meta_data = TensorMetaData(
        strides, strides, self.scalar_type(), self.suggest_memory_format());
    out.AddShapeTensor(stride_meta_data);
  }
  return out;
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
  auto self = inputs[0].toTensor();
  offset = 0;
  bool have_shape_tensors = inputs[1].isTensor();
  if (have_shape_tensors) {
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
      auto syn_shape_input = habana_helpers::create_shape_tensor(
          strides_ref,
          self.device().index(),
          graph,
          false,
          SHAPE_TENSOR,
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
    // Since GC has relaxed memory access check for min/max only have the check
    // for actual
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
    TORCH_CHECK(inputs[3].isScalar(), "Input arg 3 needs to be of scalar type");
    size = inputs[1].toIntVector();
    strides = inputs[2].toIntVector();
    offset = inputs[3].toInt();
  }

  // For Dynamic case fill strides/offset params with max size
  if (graph.is_dynamic_graph()) {
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

  auto output = habana_helpers::createPTTensor(
      self,
      size,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));

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
    auto hb_impl = habana_lazy::GetHbInternalTensorImpl(output);
    syn_tensor_output.set_dont_allow_permute(true);
    hb_impl->SetDontAllowPermutation(true);
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
        .add("hpu::habana_d2d_memcpy", KERNEL_FN_GLOBAL(MemCopyOperator))
        .add("hpu::habana_d2d_memcpy_other", KERNEL_FN_GLOBAL(MemCopyOperator))
        .add("aten::to.dtype", KERNEL_FN_GLOBAL(ToDtypeOperator))
        .add("hpu::control_edge_", KERNEL_FN_GLOBAL(DummyOperator))
        .add("hpu::control_edge_other_", KERNEL_FN_GLOBAL(DummyOperator))
        .add("hpu::as_strided_lazy_", KERNEL_FN_GLOBAL(AsStridedOperator))
        .add("hpu::as_strided_lazy_cl_", KERNEL_FN_GLOBAL(AsStridedClOperator))
        .add("hpu::strided_view", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_cl", KERNEL_FN_GLOBAL(StridedViewClOperator))
        .add("hpu::strided_view_ds", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_cl_ds", KERNEL_FN_GLOBAL(StridedViewClOperator))
        .add("hpu::strided_view_out", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_orig_ds", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_out_ds", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add(
            "hpu::strided_view_out_orig_ds",
            KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::slice_insert", KERNEL_FN_GLOBAL(SliceInsertOperator))
        .add("hpu::slice_insert_ds", KERNEL_FN_GLOBAL(SliceInsertOperator))
        .add("hpu::strided_insert", KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add("hpu::strided_insert_ds", KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add(
            "hpu::strided_insert_orig_ds",
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
        .add("hpu::identity", KERNEL_FN_GLOBAL(IdentityOperator));
