/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

#ifdef PT_KERNEL_END
#undef PT_KERNEL_BEGIN
#define PT_KERNEL_BEGIN (void)(0)
#undef PT_KERNEL_END
#define PT_KERNEL_END (void)(0)
#endif

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
std::map<c10::ScalarType, std::vector<c10::ScalarType>> const
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
  std::vector<int> out_pos = {0, 3, 1, 2};
  std::vector<int> out_pos_5d = {0, 4, 1, 2, 3};
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
  int64_t dim_chl_pos[] = {0, 2, 3, 1};
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

  if (src_device == c10::DeviceType::CPU &&
      dst_device == c10::DeviceType::HPU) {
    // CPU/source tensor should have same dtype as dst & should be contiguous
    // before H2D DMA is triggered
    auto src_contiguous =
        src.to(dst.scalar_type()).contiguous(src.suggest_memory_format());
    TORCH_CHECK(dst.nbytes() >= src_contiguous.nbytes());
    habana_helpers::copy_data_to_device(src_contiguous, dst, non_blocking);
    print_stride_warning(src_contiguous, dst);
  } else if (
      src_device == c10::DeviceType::HPU &&
      dst_device == c10::DeviceType::CPU) {
    // HPU/source tensor should be contiguous before D2H DMA is triggered
    auto src_contiguous = src.contiguous(src.suggest_memory_format());
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

  PT_KERNEL_END;
  return dst;
}

Tensor& set_hpu_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  PT_KERNEL_BEGIN;
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "inconsistent size/stride sizes");
  }

  auto scalar_type = self.scalar_type();
  auto self_ = checked_dense_tensor_unwrap(
      self, "self", 1, "_th_set_", false, DeviceType::HPU, scalar_type);
// TODO: remove this commented section
// part of 1.5 migration related change - revert once not needed
#if 0
  auto source_ = checked_storage(
      source,
      "source",
      2,
      DeviceType::HPU,
      at::scalarTypeToTypeMeta(scalar_type));
#else
  auto source_ = source;
#endif
  // Code below is based on THCTensor_setStorage
  TORCH_CHECK(
      self_->storage(),
      "Cannot use PyTorch operations on a half-constructed "
      "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
      "it first; otherwise, this is a bug, please report it.");
  auto self_storage = self_->storage().unsafeGetStorageImpl();
  auto source_storage = source_.unsafeGetStorageImpl();
  if (self_storage != source_storage) {
    TORCH_CHECK(self_storage, "Invalid null storage");
    if (self_storage) {
      c10::raw::intrusive_ptr::incref(source_storage);
      THTensor_stealAndSetStoragePtr(self_, source_storage);
    } else {
      auto THHStorage_new = []() -> THStorage* {
        THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                                 c10::StorageImpl::use_byte_size_t(),
                                 0,
                                 habana::getHABANADeviceAllocator(),
                                 true)
                                 .release();
        return storage;
      };
      THTensor_stealAndSetStoragePtr(self_, THHStorage_new());
    }
  }

  TORCH_CHECK(storage_offset >= 0, "Invalid storage offset: ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  THHTensor_resizeNd(self_, stride.size(), size.data(), stride.data());

  PT_KERNEL_END;
  return self;
}

void ToDtypeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
    auto iter = habana_helpers::cast_map.find(type_key);
    if (iter != habana_helpers::cast_map.end()) {
      node_type = iter->second;
    } else {
      HABANA_ASSERT(
          0 &&
              "Unsupported Cast operation requested in ToDtypeOperator::AllocateAndAddSynapseNode",
          self.scalar_type(),
          " -> ",
          type);
    }
  } else {
    // Cases where a simple copy is being done (input_new = input) come as .to
    // call with same input & output data types. we add a identity node to
    // graph to handle this
    auto memcopyOp = make_operator<IdentityOperator>(
        self.device().index(), self.scalar_type());
    memcopyOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    torch::jit::Stack stack = {IValue(self)};
    memcopyOp->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

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
  Op->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  p_context_->syn_outputs_.emplace_back(std::move(Op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(Op->GetOutputs()[0]));
}

void CastLazyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  auto self = inputs[0].toTensor();
  // auto output = inputs[1].toTensor();
  auto type = inputs[1].toScalarType();

  std::string node_type;
  if (self.scalar_type() != type) {
    std::pair<c10::ScalarType, c10::ScalarType> type_key{
        self.scalar_type(), type};
    auto iter = habana_helpers::cast_map.find(type_key);
    if (iter != habana_helpers::cast_map.end()) {
      node_type = iter->second;
    } else {
      HABANA_ASSERT(
          0 &&
              "Unsupported Cast operation requested in CastLazyOperator::AllocateAndAddSynapseNode: ",
          self.scalar_type(),
          " -> ",
          type);
    }
  } else {
    node_type = "cast_identity";
  }

  /*
   TODO: This is the Original implementation for Cast Operator
        where we pass the output of the cast as part of the inputs,
        making this as inplace operator. For Resnet we decided
        to make the inplace cast operator as cast out operator.
        The down side of cast out operator is it cannot give us
        the effect of eager mode, cast out operator would create
        a new pytorch output and use that instead of the one that
        is altready created by the .to operator from pytorch.

        Will eventually enable this class as needed going further.

  SetGuid(node_type);

  ns_CastKernel::Params params = synapse_cast_params_builder();
  p_context_->params_.emplace<ns_CastKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  HABANA_ASSERT(p_context_->syn_inputs_.size() == 2);
  synapse_helpers::tensor_or_ref& input_tensor = p_context_->syn_inputs_.back();
  p_context_->syn_outputs_.emplace_back(std::move(input_tensor));
  p_context_->pt_outputs_.emplace_back(output);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));*/

  // Insert the cast node - in case cast is to same type alias, insert an
  // identity op
  if (node_type.compare("cast_identity")) {
    auto Op = make_operator<CastOperator>(self.device().index(), node_type);
    Op->SetSynapseInput(p_context_->syn_inputs_[0]);
    Op->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_outputs_.emplace_back(std::move(Op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(Op->GetOutputs()[0]));
  } else {
    auto identityOp = make_operator<IdentityOperator>(
        self.device().index(), self.scalar_type());
    identityOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    torch::jit::Stack stack = {IValue(self)};
    identityOp->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    p_context_->syn_outputs_.emplace_back(
        std::move(identityOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(identityOp->GetOutputs()[0]));
  }
}

/*************************************************************************
 * @brief Kernel implementation for memcpy, used for D2D mem transfers
 * @param self - input which needs to be transferred
 * @param dest - Destination tensor
 ************************************************************************/
void MemCopyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  at::Tensor output;
  if (inputs.size() == 2) {
    output = inputs[1].toTensor();
    synapse_helpers::tensor& output_tensor = p_context_->syn_inputs_[1];
    p_context_->syn_outputs_.emplace_back(output_tensor);
    p_context_->pt_outputs_.emplace_back(output);
  } else {
    output = habana_helpers::createPTTensor(self, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }
  p_context_->params_size_ = 0;
  AddNodeToSynapseGraph(graph, NULL, 0);
}

void IdentityOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  at::Tensor output;
  if (inputs.size() == 2) {
    output = inputs[1].toTensor();
  } else {
    output = habana_helpers::createPTTensor(self, is_output_persistent);
  }
  p_context_->params_size_ = 0;
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, NULL, 0);
}

/*************************************************************************
 * @brief Kernel implementation for dummy, used for graph ordering
 ************************************************************************/
void DummyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(graph);
  static_cast<void>(is_output_persistent);
  at::Tensor output;
  int out_index = inputs.size() - 1;
  output = inputs[out_index].toTensor();
  p_context_->params_size_ = 0;
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[out_index], graph));
  p_context_->pt_outputs_.emplace_back(output);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> AsStridedOperator::
    compute_output_shape(
        const Tensor& self,
        IntArrayRef size,
        IntArrayRef stride) {
  std::vector<int64_t> out_size_vec;
  std::vector<int64_t> out_stride_vec;

  if ((self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast) &&
      (size.size() == 4)) {
    // NCHW -> NHWC
    const int64_t dim_pos_in[4] = {0, 2, 3, 1};
    for (size_t idx = 0; idx < size.size(); idx++) {
      out_size_vec.emplace_back(size[dim_pos_in[idx]]);
      out_stride_vec.emplace_back(stride[dim_pos_in[idx]]);
    }
  } else if (
      (self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d) &&
      (size.size() == 5)) {
    // NCDHW -> NDHWC
    const int64_t dim_pos_in[5] = {0, 2, 3, 4, 1};
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

  return std::make_tuple(out_size_vec, out_stride_vec);
}

/*************************************************************************
 * @brief Kernel implementation for As strided, used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void AsStridedOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  static_cast<void>(graph);
  static_cast<void>(is_output_persistent);
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
          offset * self.itemsize()));
  p_context_->pt_outputs_.emplace_back(output);
}

/*************************************************************************
 * @brief Kernel implementation for As strided, used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void AsStridedLayoutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  static_cast<void>(graph);
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      inputs[1].isIntList(), "Input arg 1 needs to be of Int List type");

  at::Tensor output;
  int64_t offset = 0;
  c10::optional<int64_t> opt_offset = c10::make_optional((int64_t)0);
  auto sizes = self.sizes().vec();
  auto dims = inputs[1].toIntVector();
  std::vector<int64_t> swapped_sizes = {
      sizes[dims[0]], sizes[dims[1]], sizes[dims[2]], sizes[dims[3]]};

  std::vector<int64_t> new_strides = {
      swapped_sizes[1] * swapped_sizes[2] * swapped_sizes[3],
      swapped_sizes[3] * swapped_sizes[2],
      swapped_sizes[3],
      1};

  output = at::as_strided(self, swapped_sizes, new_strides, opt_offset);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section_with_size(
          p_context_->syn_inputs_[0],
          graph,
          swapped_sizes,
          new_strides,
          offset * self.itemsize()));
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

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habana_d2d_memcpy",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MemCopyOperator>(device_id, node_type);
            })
        .add(
            "hpu::habana_d2d_memcpy_other",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MemCopyOperator>(device_id, node_type);
            })
        .add(
            "hpu::cast",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<CastLazyOperator>(device_id, node_type);
            })
        .add(
            "aten::to.dtype",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ToDtypeOperator>(device_id, node_type);
            })
        .add(
            "hpu::control_edge_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<DummyOperator>(device_id, node_type);
            })
        .add(
            "hpu::control_edge_other_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<DummyOperator>(device_id, node_type);
            })
        .add(
            "hpu::as_strided_lazy_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<AsStridedOperator>(device_id, node_type);
            })
        .add(
            "hpu::as_strided_lazy_cl_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<AsStridedClOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::as_strided_layout_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<AsStridedLayoutOperator>(
                  device_id, node_type);
            });
