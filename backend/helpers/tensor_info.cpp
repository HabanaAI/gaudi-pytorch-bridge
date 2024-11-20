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

#include "backend/helpers/tensor_info.h"
#include "backend/backend_meta.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/get_n_bytes.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/random.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/serializers.h"

void DMAInputGenerators::populateSeedTensor(
    const PtTensorInfo& ti,
    at::Tensor& dma_tensor) {
  auto gen = torch::get_generator_or_default<torch::CPUGeneratorImpl>(
      c10::nullopt, habana::detail::getDefaultHPUGenerator());

  // Acquire lock when using random generators
  std::vector<int> seed_vec;
  std::lock_guard<std::mutex> lock(gen->mutex_);
  for (size_t i = 0; i < ti.get_numel(); i++) {
    seed_vec.push_back((int)gen->random());
  }

  auto vec_size = seed_vec.size() * sizeof(seed_vec[0]);
  TORCH_CHECK(
      vec_size == ti.get_size(),
      " cpu vec size ",
      vec_size,
      " mismatch with ti.get_size ",
      ti.get_size());

  at::IntArrayRef tshape{ti.get_shape()};
  habana_helpers::copy_scalar_to_device(
      seed_vec.data(), dma_tensor, ti.get_size());
}

void PtTensorInfo::populate_tinfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const uint64_t tensor_id,
    const synTensorType stt,
    DMAInputGeneratorType dma_gen_id) {
  ir_name_ = irn;
  syn_name_ = sn;
  tensor_id_ = tensor_id;

  is_ZST_ = habana::is_ZST(pt_tensor);

  buffer_ = pt_tensor.data_ptr();
  buffer_start_ = pt_tensor.storage().data_ptr().get();

  numel_ = pt_tensor.numel();
  size_ = habana_helpers::GetNBytes(pt_tensor);
  shape_ = pt_tensor.sizes().vec();
  strides_ = pt_tensor.strides().vec();

  mf_ = pt_tensor.suggest_memory_format();
  topts_ = pt_tensor.options();

  auto tmeta{habana::get_tensor_extra_meta(pt_tensor)};
  std::tie(hb_internal_perm_, hb_dont_allow_permute_) =
      habana_helpers::get_tensor_memory_permutation(pt_tensor);
  hb_internal_lf_ = tmeta->get_tensor_layout();
  PT_BACKEND_DEBUG_TENSOR(
      pt_tensor,
      "Saving the layout and permutation to the cache for tensor: {:d}"
      " permutation: {:s}",
      tensor_id,
      habana_helpers::FormatTokens::Permutations);

  if (get_buffer_syn() != 0) {
    // set valid offset
    offset_ = (get_buffer_syn() - get_buffer_start_syn());
  }
  is_view_tensor_ = (offset_ != 0);

  dma_gen_id_ = dma_gen_id;

  tensor_type_ = stt;

  update_shape_syn();
}

PtTensorInfo::PtTensorInfo(
    const synapse_helpers::tensor& st,
    const std::string& irn) {
  ir_name_ = irn;
  syn_name_ = st.name();
  tensor_type_ = st.tensor_type();
  // Populate the synapse shapes directly from the input syn tensor.
  numel_ = st.num_elements();
  size_ = st.size_bytes();
  shape_ = st.pt_shape();
  strides_ = st.pt_strides();
  tensor_id_ = st.id();

  update_shape_syn();
}

PtTensorInfo::PtTensorInfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const uint64_t tensor_id,
    const synTensor handle,
    const synTensorType stt,
    DMAInputGeneratorType dma_gen_id)
    : orig_syn_handle_(handle) {
  populate_tinfo(pt_tensor, sn, irn, tensor_id, stt, dma_gen_id);
}

PtTensorInfo::PtTensorInfo(
    const IValPtrShared& ivpsh,
    const std::string& sn,
    const ValPtr& vp,
    const uint64_t tensor_id,
    const synTensor handle,
    const synTensorType stt,
    DMAInputGeneratorType dma_gen_id)
    : orig_syn_handle_(handle) {
  TORCH_CHECK(ivpsh->isTensor(), "aten tensor is expected");
  std::string irn = "%" + vp->debugName();
  auto pt_tensor = ivpsh->toTensor();
  populate_tinfo(pt_tensor, sn, irn, tensor_id, stt, dma_gen_id);
}

PtTensorInfo::PtTensorInfo(
    const std::string& sn,
    const uint64_t tensor_id,
    const synTensor handle,
    const synTensorType stt,
    const std::vector<int64_t> shape)
    : syn_name_(sn),
      shape_(shape),
      tensor_type_(stt),
      tensor_id_(tensor_id),
      orig_syn_handle_(handle) {}

void PtTensorInfo::update_shape_syn() {
  switch (tensor_type_) {
    case DATA_TENSOR:
    case SHAPE_TENSOR:
    case DATA_TENSOR_DYNAMIC:
    case HOST_TO_DEVICE_TENSOR:
      HABANA_ASSERT(SYN_GAUDI_MAX_TENSOR_DIM >= shape_.size());
      for (size_t i = 0; i < shape_.size(); ++i) {
        // Reverse PyTorch shapes for synapse tensor shape patching
        if (i < shape_.size()) {
          syn_shape_[i] = shape_[shape_.size() - 1 - i];
        }
      }
      break;
    case DEVICE_SHAPE_TENSOR:
      syn_shape_ = {SYN_MAX_TENSOR_DIM, 0, 0, 0, 0};
      break;
    case TENSOR_TYPE_MAX:
    default:
      TORCH_CHECK(false, "Unreachable condition.");
  }
}

PtTensorInfo::PtTensorInfo(std::istream& is) {
  using namespace serialization;
  deserialize(is, is_ZST_);
  deserialize(is, is_view_tensor_);
  deserialize(is, is_restrided_);
  deserialize(is, is_allow_permutation_);
  deserialize(is, hb_internal_perm_);
  deserialize(is, hb_dont_allow_permute_);
  deserialize(is, offset_);
  deserialize(is, ir_name_);
  deserialize(is, syn_name_);
  deserialize(is, numel_);
  deserialize(is, size_);
  deserialize(is, is_duplicate_);
  deserialize(is, parent_index_);
  deserialize(is, output_index_);
  deserialize(is, shape_);
  deserialize(is, strides_);
  deserialize(is, mf_);
  deserialize(is, topts_);
  deserialize(is, tensor_type_);
  deserialize(is, dma_tensor_idx_);
  deserialize(is, tensor_id_);
  deserialize(is, dma_gen_id_);

  update_shape_syn(); // constructs syn_shape_ according to shape_ and
                      // tensor_type_
}

void PtTensorInfo::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, is_ZST_);
  serialize(os, is_view_tensor_);
  serialize(os, is_restrided_);
  serialize(os, is_allow_permutation_);
  serialize(os, hb_internal_perm_);
  serialize(os, hb_dont_allow_permute_);
  serialize(os, offset_);
  serialize(os, ir_name_);
  serialize(os, syn_name_);
  serialize(os, numel_);
  serialize(os, size_);
  serialize(os, is_duplicate_);
  serialize(os, parent_index_);
  serialize(os, output_index_);

  serialize(os, shape_);
  serialize(os, strides_);
  serialize(os, mf_);
  serialize(os, topts_);
  serialize(os, tensor_type_);
  serialize(os, dma_tensor_idx_);
  serialize(os, tensor_id_);
  serialize(os, dma_gen_id_);
}

std::ostream& operator<<(std::ostream& O, const PtTensorInfo& t) {
  O << '<' << t.get_ir_name();
  O << ":[" << t.get_shape() << "]:[" << t.get_strides() << "]:#"
    << t.get_numel() << ':' << '(' << t.get_size() << " b):["
    << t.getHbInternalLayoutFormat() << "]"
    << " :: " << t.get_syn_name() << ':' << t.get_buffer() << '>'
    << " tensor type:" << t.tensor_type_ << " tensor id: " << t.tensor_id_;

  if (t.get_dma_cb() != nullptr) {
    O << " dma_cb : " << (void*)t.get_dma_cb();
    O << " dma_tensor_idx : " << t.get_dma_tensor_idx();
  }
  if (t.is_duplicate()) {
    O << " duplicate of " << t.get_parent_index();
  }

  if (ULONG_MAX != t.get_output_index()) {
    O << ", output index " << t.get_output_index();
  } else {
    O << ", non output";
  }

  O << ", is_restrided " << std::boolalpha << t.is_restrided();

  O << ", <" << t.get_buffer_start() << ", +" << t.offset_ << ">";

  if (t.offset_ != 0) {
    O << " nz offset view tensor ";
  }
  return O;
}
