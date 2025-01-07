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

#include "backend/kernel/hpu_habana_cache.h"
#include <algorithm>
#include <iomanip>
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/collective_kernel_info.h"
#include "backend/helpers/tensor_info.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_meta_op_list.h"
#include "backend/synapse_helpers/devmem_logger.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/event.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_helpers/towl.h"
#include "habana_kernels/hccl_kernels.h"
#include "habana_lazy/memlog.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/recipe_cache_config.h"
#include "habana_serialization/serializers.h"

namespace habana {

HbCas::HbCas(bool with_grad, at::ArrayRef<c10::IValue> inputs) {
  p_cas = std::make_shared<torch::jit::CompleteArgumentSpec>(with_grad, inputs);
}

RecipeArgumentSpec::RecipeArgumentSpec(
    at::ArrayRef<torch::jit::IValue> input_refs,
    const size_t& graphKey,
    const std::string& op_strs)
    : cas(false, input_refs), opstrs(op_strs), graph_hash_code(graphKey) {
  hash_code = graph_hash_code;
  size_t sym_hash_code = habana::ComputeSymSizeHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, sym_hash_code);
  size_t perm_hash_code = habana::ComputePermutationHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, perm_hash_code);
  graph_with_permute_hash_code = hash_code;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    at::ArrayRef<torch::jit::IValue> input_refs,
    const size_t& graphKey,
    const std::string& op_strs,
    const uint64_t token)
    : cas(false, input_refs), opstrs(op_strs) {
  graph_hash_code = graphKey;
  hash_code = at::hash_combine(hash_code, graph_hash_code);

  token_ = token;
  hash_code = at::hash_combine(hash_code, token_);

  ComputeOffsetHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, offset_hash_code);
  size_t sym_hash_code = habana::ComputeSymSizeHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, sym_hash_code);
  size_t perm_hash_code = habana::ComputePermutationHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, perm_hash_code);
  dynamic_hash_code = hash_code;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    at::ArrayRef<torch::jit::IValue> input_refs,
    const size_t& graphKey,
    const size_t& graph_sym_hash,
    const size_t& graph_perm_hash,
    const std::string& op_strs)
    : cas(false, input_refs), opstrs(op_strs), graph_hash_code(graphKey) {
  hash_code = graph_hash_code;
  hash_code = at::hash_combine(hash_code, graph_sym_hash);
  hash_code = at::hash_combine(hash_code, graph_perm_hash);
  graph_with_permute_hash_code = hash_code;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    at::ArrayRef<torch::jit::IValue> input_refs,
    const size_t& graphKey,
    const size_t& graph_sym_hash,
    const size_t& graph_perm_hash,
    const std::string& op_strs,
    const uint64_t token)
    : cas(false, input_refs), opstrs(op_strs) {
  graph_hash_code = graphKey;
  hash_code = at::hash_combine(hash_code, graph_hash_code);
  token_ = token;
  hash_code = at::hash_combine(hash_code, token_);

  ComputeOffsetHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, offset_hash_code);
  hash_code = at::hash_combine(hash_code, graph_sym_hash);
  hash_code = at::hash_combine(hash_code, graph_perm_hash);
  dynamic_hash_code = hash_code;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    bool with_grad,
    at::ArrayRef<torch::jit::IValue> input_refs,
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const size_t& graphKey,
    const std::string& op_strs,
    size_t symhash,
    size_t permhash)
    : cas(with_grad, input_refs), opstrs(op_strs), hash_code(cas.hashCode()) {
  cargspec_hash_code = cas.hashCode();
  graph_hash_code = graphKey;
  hash_code = at::hash_combine(hash_code, graph_hash_code);
  hash_code = at::hash_combine(hash_code, irgraph->outputs().size());
  hash_code = habana_helpers::hash_combine_scalars(hash_code, input_refs);

  ComputeOffsetHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, offset_hash_code);
  ComputeH2DHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, h2d_hash_code);
  hash_code = at::hash_combine(hash_code, symhash);
  graph_with_permute_hash_code = permhash;
  hash_code = at::hash_combine(hash_code, permhash);

  for (auto* node : irgraph->nodes()) {
    auto node_qual_str = node->kind().toQualString();
    /*Ignore the const & meta nodes*/
    if (node->kind().is_prim() ||
        HabanaMetaOpList::isHabanaMetaOp(node_qual_str)) {
      continue;
    }

    hash_code =
        at::hash_combine(hash_code, node->i(torch::jit::attr::deterministic));
  }
}

void RecipeArgumentSpec::ComputeH2DHashCode(
    at::ArrayRef<torch::jit::IValue> input_refs) {
  h2d_hash_code = 0;
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      auto pt_tensor = input.toTensor();
      auto tmeta{get_tensor_extra_meta(pt_tensor, true)};
      if (tmeta && tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        size_t h2d_size = tmeta->get_host_size();

        std::vector<int64_t> h2d_vec;
        habana::HostDataType h2d_dt_type = tmeta->get_host_dt_type();
        if (h2d_dt_type == habana::HostDataType::INT32_T) {
          int32_t* h2d_data = static_cast<int32_t*>(tmeta->get_host_ptr());
          for (size_t i = 0; i < h2d_size; i++) {
            h2d_vec.push_back(static_cast<int64_t>(*h2d_data++));
          }
        } else if (h2d_dt_type == habana::HostDataType::UINT32_T) {
          uint32_t* h2d_data = static_cast<uint32_t*>(tmeta->get_host_ptr());
          for (size_t i = 0; i < h2d_size; i++) {
            h2d_vec.push_back(static_cast<int64_t>(*h2d_data++));
          }
        } else if (h2d_dt_type == habana::HostDataType::UINT64_T) {
          uint64_t* h2d_data = static_cast<uint64_t*>(tmeta->get_host_ptr());
          for (size_t i = 0; i < h2d_size; i++) {
            uint64_t h2d_elem = *h2d_data++;
            TORCH_CHECK(
                h2d_elem < LONG_MAX,
                "H2D data ",
                h2d_elem,
                " exceeds the int64 limit");
            h2d_vec.push_back(static_cast<int64_t>(h2d_elem));
          }
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Host datatype Not Supported while processing host data from bucketing");
        }

        size_t h2d_single_value = 0;
        for (size_t i = 0; i < h2d_vec.size(); i++) {
          h2d_single_value = h2d_single_value + ((i + 1) * h2d_vec[i]);
        }
        h2d_hash_code = at::hash_combine(h2d_hash_code, h2d_single_value);
      }
    }
  }
}

void RecipeArgumentSpec::ComputeOffsetHashCode(
    at::ArrayRef<torch::jit::IValue> input_refs) {
  offset_hash_code = 0;
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      auto pt_tensor = input.toTensor();
      synapse_helpers::device_ptr storage_data_ptr_ =
          reinterpret_cast<synapse_helpers::device_ptr>(
              pt_tensor.storage().data_ptr().get());
      synapse_helpers::device_ptr buffer_ptr =
          reinterpret_cast<synapse_helpers::device_ptr>(pt_tensor.data_ptr());
      auto offset = (buffer_ptr - storage_data_ptr_);
      offset_hash_code = at::hash_combine(offset_hash_code, offset);
    }
  }
}

std::ostream& operator<<(std::ostream& O, const RecipeArgumentSpec& v) {
  O << "RecipeArgumentSpec :: is graph key : " << std::boolalpha
    << (v.hashCode() == v.graphHashCode()) << std::noboolalpha << '\n';
  O << "combined hash_code : " << v.hashCode() << '\n';
  O << "graph    hash_code : " << v.graphHashCode() << '\n';
  O << "offset   hash_code : " << v.offsetHashCode() << '\n';
  O << "cArgSpec hash_code : " << v.cArgSpecHashCode() << '\n';
  O << "Dynamic  hash_code : " << v.dynamicHashCode() << '\n';

  return O;
}

RecipeValueSpec::~RecipeValueSpec() {
  PT_BRIDGE_DEBUG("Destroying recipe with key : ", key);
}

std::ostream& operator<<(std::ostream& O, const RecipeLauncher& v) {
  O << "\n----   RecipeLauncher :: begin \n";
  O << " <id : " << v.id_ << ">\n";
  O << " ntensorbytes : " << synapse_helpers::get_mem_str(v.ntensorbytes_)
    << '\n';
  O << " workspace    : " << synapse_helpers::get_mem_str(v.workspace_size_)
    << '\n';
  O << " <addr : " << v.recipe_.get() << "> "
    << " <use_count : " << v.recipe_.use_count() << "> "
    << "\n";
  O << " <num_launches : " << v.num_launches << ">\n";
  O << " #inputs                        : " << v.num_inputs_ << '\n'
    << " #outputs                       : " << v.num_outputs_ << '\n'
    << " #input_to_outduplicates        : " << v.num_input_to_outduplicates_
    << '\n'
    << " #intermediate_to_outduplicates : "
    << v.num_intermediate_to_outduplicates_ << '\n';
  O << "----   RecipeLauncher :: end";
  return O;
}

std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v) {
  O << '\n'
    << "---- recipe details :: begin" << '\n'
    << " <id : " << v.id << "> "
    << " <iteration : " << v.iter_idx << "> ";
  O << " #inputs                        : " << v.num_inputs << '\n';
  O << " #induplicates                  : " << v.num_induplicates << '\n'
    << " #dma_inputs                    : " << v.num_dma_inputs << '\n'
    << " #intermediates                 : " << v.num_intermediates << '\n'
    << " #outputs                       : " << v.num_outputs << '\n'
    << " #outduplicates                 : " << v.num_outduplicates << '\n'
    << " #input_to_outduplicates        : " << v.num_input_to_outduplicates
    << '\n'
    << " #intermediate_to_outduplicates : "
    << v.num_intermediate_to_outduplicates << '\n'
    << " #output_to_outduplicates       : " << v.num_output_to_outduplicates
    << '\n';

  O << "dtensorinfos #" << v.dtensorinfos.size() << "::";
  O << '\n';
  size_t idx{0};
  for (auto& a : v.dtensorinfos) {
    O << idx++ << " : ";
    O << *a << '\n';
  }
  if (!v.sif_tidx_to_tinfo_map.empty()) {
    O << "sif_tidx_to_tinfo_map #" << v.sif_tidx_to_tinfo_map.size() << "::";
    O << '\n';
    std::vector<size_t> tidx_vec;
    for (auto const& p : v.sif_tidx_to_tinfo_map) {
      tidx_vec.emplace_back(p.first);
    }

    std::sort(tidx_vec.begin(), tidx_vec.end());
    for (auto const& idx : tidx_vec) {
      O << "sif_tidx : " << idx << " -> " << *(v.sif_tidx_to_tinfo_map.at(idx))
        << '\n';
    }
  }
  O << "---- recipe details :: end" << '\n';

  return O;
}

std::string RecipeValueSpec::header_str() {
  if (header.empty()) {
    header = build_header_str();
  }
  return header;
}

std::string RecipeValueSpec::build_header_str() const {
  std::ostringstream O;
  O << "\n key " << key << "\n graph_key " << graph_key << "\n num_inputs "
    << num_inputs << "\n num_induplicates " << num_induplicates
    << "\n num_dma_inputs " << num_dma_inputs << "\n num_intermediates "
    << num_intermediates << "\n num_outputs " << num_outputs
    << "\n num_outduplicates " << num_outduplicates
    << "\n num_input_to_outduplicates " << num_input_to_outduplicates
    << "\n num_intermediate_to_outduplicates "
    << num_intermediate_to_outduplicates << "\n size "
    << (dynamic_graph ? "dynamic graph" : "static graph") << " - "
    << (is_refined ? "refined" : "original");
  return O.str();
}

std::string RecipeValueSpec::digest_str() {
  std::ostringstream O;
  auto& recipe_cache = HPUDeviceContext::get_device().get_recipe_handle_cache();
  O << "Recipe digest : total size of graph recipes "
    << synapse_helpers::get_mem_str(RecipeValueSpec::total_recipe_ntbytes)
    << '\n';
  auto rv_hit_count = recipe_cache.getHitCount(key);
  if (-1 != rv_hit_count) {
    // Hit count needs to be enabled with
    // PT_HABANA_MAX_RECIPE_HIT_COUNT=<positive number>
    O << " #hits " << rv_hit_count << '\n';
  }
  O << " #graph_recipes " << recipe_count << " (#static "
    << (recipe_count - dynamic_recipe_count) << ", #dynamic "
    << dynamic_recipe_count << ')' << '\n'
    << " #eager_recipes " << recipe_cache.getCount();

  return O.str();
}

int RecipeValueSpec::update_hit_count() {
  auto& device = HPUDeviceContext::get_device();
  device.get_recipe_handle_cache().increaseHitCount(key);
  auto rv_hit_count = device.get_recipe_handle_cache().getHitCount(key);

  auto max_hit_count = GET_ENV_FLAG_NEW(PT_HABANA_MAX_RECIPE_HIT_COUNT);
  if (max_hit_count && rv_hit_count >= int(max_hit_count)) {
    device.get_recipe_handle_cache().printHitCount();
    PT_BRIDGE_DEBUG(
        "Max hit count ",
        max_hit_count,
        " reached. Resetting the hit counter.");
    device.get_recipe_handle_cache().clearHitCount();
  }
  return rv_hit_count;
}

RecipeHolder::RecipeHolder(std::istream& is) {
  using namespace serialization;
  rvs_ = std::make_shared<RecipeValueSpec>(is);
  rl_ = std::make_shared<RecipeLauncher>(is, *rvs_);
}

RecipeValueSpec::RecipeValueSpec(std::istream& is) {
  using namespace serialization;
  int info_size = 0;
  deserialize(is, info_size);
  dtensorinfos.reserve(info_size);
  for (int i = 0; i < info_size; ++i) {
    dtensorinfos.emplace_back(std::make_shared<PtTensorInfo>(is));
  }
  deserialize(is, id);
  deserialize(is, iter_idx);
  deserialize(is, num_inputs);
  deserialize(is, num_induplicates);
  deserialize(is, num_dma_inputs);
  deserialize(is, num_shape_tensors);
  deserialize(is, num_intermediates);
  deserialize(is, num_outputs);
  deserialize(is, num_outduplicates);
  deserialize(is, num_input_to_outduplicates);
  deserialize(is, num_intermediate_to_outduplicates);
  deserialize(is, num_output_to_outduplicates);
  deserialize(is, key);
  deserialize(is, graph_key);
  deserialize(is, opstrs);
  deserialize(is, header);
  size_t num_tensors;
  deserialize(is, num_tensors);
  deserialize(is, graph_name);
  tensor_ids_.resize(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    deserialize(is, tensor_ids_[i]);
  }
  deserialize(is, dynamic_graph);
  deserialize(is, is_refined);
  deserialize(is, is_refined_wirt);
  deserialize(is, count);
  deserialize(is, total_recipe_ntbytes);
  // deserialize(is, get_use_flag());
  collective_kernels_info =
      std::make_shared<habana_helpers::CollectiveKernelInfos>(is, dtensorinfos);

  if (dynamic_graph) {
    std::vector<int64_t> sif_tensor_indices;
    deserialize(is, sif_tensor_indices);

    for (size_t idx = 0; idx < sif_tensor_indices.size(); ++idx) {
      sif_tidx_to_tinfo_map.insert(
          {sif_tensor_indices[idx], dtensorinfos.at(idx)});
    }
    deserialize(is, disabled_jit_ir_ops_);
    deserialize(is, st_to_tensor_idx_map);
    deserialize(is, enable_optim_output_sif_);
    deserialize(is, dynamic_nodes_with_backend_STs);
  }
}

void RecipeHolder::Serialize(std::ostream& os) const {
  using namespace serialization;
  rvs_->Serialize(os);
  rl_->Serialize(os);
}

void RecipeLauncher::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, recipe_ != nullptr);
  if (recipe_) {
    serialize(os, recipe_->recipe_name_);
    serialize(os, recipe_->graph_is_empty_);
  }
  serialize(os, workspace_size_);
  serialize(os, ntensorbytes_);
}

void RecipeValueSpec::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, static_cast<int>(dtensorinfos.size()));
  for (const PtTensorInfoShared& tInfo : dtensorinfos) {
    tInfo->Serialize(os);
  }
  serialize(os, id);
  serialize(os, iter_idx);
  serialize(os, num_inputs);
  serialize(os, num_induplicates);
  serialize(os, num_dma_inputs);
  serialize(os, num_shape_tensors);
  serialize(os, num_intermediates);
  serialize(os, num_outputs);
  serialize(os, num_outduplicates);
  serialize(os, num_input_to_outduplicates);
  serialize(os, num_intermediate_to_outduplicates);
  serialize(os, num_output_to_outduplicates);
  serialize(os, key);
  serialize(os, graph_key);
  serialize(os, opstrs);
  serialize(os, header);
  serialize(os, tensor_ids_.size());
  serialize(os, graph_name);
  for (auto el : tensor_ids_) {
    serialize(os, el);
  }

  serialize(os, dynamic_graph);
  serialize(os, is_refined);
  serialize(os, is_refined_wirt);
  serialize(os, count);
  serialize(os, total_recipe_ntbytes);

  collective_kernels_info->Serialize(os, dtensorinfos);

  if (dynamic_graph) {
    std::unordered_map<PtTensorInfoShared, int64_t> tinfo_to_sif_tidx_map;
    std::vector<int64_t> sif_tensor_indices;
    for (auto const& ele : sif_tidx_to_tinfo_map) {
      tinfo_to_sif_tidx_map[ele.second] = ele.first;
    }
    for (const PtTensorInfoShared& tinfo : dtensorinfos) {
      sif_tensor_indices.push_back(tinfo_to_sif_tidx_map[tinfo]);
    }
    serialize(os, sif_tensor_indices);
    serialize(os, disabled_jit_ir_ops_);
    serialize(os, st_to_tensor_idx_map);
    serialize(os, enable_optim_output_sif_);
    serialize(os, dynamic_nodes_with_backend_STs);
  }
}

synTensor RecipeValueSpec::get_syn_new_handle(
    std::unordered_map<uint64_t, synTensor>& synapse_tensor_id_to_tensor_handle,
    std::unordered_map<synTensor, synTensor>& synapse_orig_to_new_handle,
    size_t ridx) {
  HABANA_ASSERT(
      synapse_tensor_id_to_tensor_handle.find(ridx) !=
          synapse_tensor_id_to_tensor_handle.end(),
      "ridx : ",
      ridx,
      " not present in the tensor id to orig handle map");
  synTensor orig_handle = synapse_tensor_id_to_tensor_handle.find(ridx)->second;
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] Tensor ridx : ", ridx, " origHandle : ", orig_handle);
  if (synapse_orig_to_new_handle.find(orig_handle) ==
      synapse_orig_to_new_handle.end()) {
    // This could be changed assert if it is gauranteed that synapse
    // duplicate API returns orig/new handle for each persistent tensor
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] origHandle : ",
        orig_handle,
        " not present in the synapse_orig_to_new_handle map");
    return nullptr;
  }
  synTensor new_handle = synapse_orig_to_new_handle.find(orig_handle)->second;
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] Tensor ridx : ", ridx, " newHandle : ", new_handle);
  return new_handle;
}

void RecipeValueSpec::update_tensor_shape(
    synTensor tensor_handle,
    PtTensorInfoShared tinfo,
    std::vector<int64_t> shape) {
  PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape used for patching : ", shape);
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] tensor shape before patching : ", tinfo->get_shape());
  if (shape.size() == 0 && !tinfo->is_ZST()) {
    shape = {1};
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] settting the tensor shape to {1} for scalar");
  }
  tinfo->set_shape(shape);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 1; i > 0; i--) {
    strides[i - 1] *= shape[i] * strides[i];
  }
  tinfo->set_strides(strides);
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] tensor shape after patching : ", tinfo->get_shape());
  synapse_helpers::graph::setTensorGeometry(tensor_handle, shape);
}

inline void RecipeValueSpec::update_new_tensor(
    size_t ridx,
    std::unordered_map<synTensor, synTensor>& synapse_orig_to_new_handle,
    std::vector<int64_t> new_shape,
    std::optional<uint64_t> tensor_offset_opt,
    std::optional<PtTensorInfoShared> tinfo_opt) const {
  PtTensorInfoShared tinfo;
  // tinfo_opt is for passing sif info instead of dtensor info
  if (tinfo_opt.has_value()) {
    tinfo = tinfo_opt.value();
  } else {
    tinfo = dtensorinfos.at(ridx);
  }

  auto orig_handle = tinfo->get_orig_syn_handle();

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] Tensor ridx : ", ridx, " orig_handle : ", orig_handle);

  synTensor new_handle = nullptr;
  if (synapse_orig_to_new_handle.find(orig_handle) ==
      synapse_orig_to_new_handle.end()) {
    // This could be changed assert if it is gauranteed that synapse
    // duplicate API returns orig/new handle for each persistent tensor
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] origHandle : ",
        orig_handle,
        " not present in the synapse_orig_to_new_handle map");
  } else {
    new_handle = synapse_orig_to_new_handle.find(orig_handle)->second;
  }

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] Tensor ridx : ", ridx, " newHandle : ", new_handle);

  size_t tensorId = tinfo->get_tensor_id();
  PT_EAGER_DEBUG("[SHAPE AGNOSTIC] ridx : ", ridx, " tensor id : ", tensorId);

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] new handle : ",
      new_handle,
      " is_output : ",
      tinfo->is_output(),
      " is_duplicate : ",
      tinfo->is_duplicate(),
      " is_ZST : ",
      tinfo->is_ZST(),
      " allow perm : ",
      tinfo->get_allow_permutation(),
      " output idx : ",
      tinfo->get_output_index());

  if (new_handle) {
    update_tensor_shape(new_handle, tinfo, new_shape);
    if (tensor_offset_opt.has_value()) {
      const auto& new_offset = tensor_offset_opt.value();
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] new handle : ",
          new_handle,
          " new section offset : ",
          new_offset);
      synapse_helpers::graph::setTensorSectionOffset(new_handle, new_offset);
    }
  }
}

void RecipeValueSpec::update_patching_table(
    at::ArrayRef<torch::jit::IValue>& input_refs,
    std::shared_ptr<VecOfIValPtrSh>& intermediate_tensors_ptr,
    VecOfIValPtrSh& dma_inputs,
    VecOfIValPtrSh& aten_outputs,
    const habana::IdShapeMap& m_actual_shapes,
    std::optional<
        std::reference_wrapper<const std::unordered_map<int64_t, at::Tensor>>>
        tidx_to_tensor_map_opt,
    const std::optional<std::vector<at::Tensor>>& allocated_outputs,
    std::vector<std::vector<int64_t>> output_shapes,
    std::unordered_map<synTensor, synTensor> synapse_orig_to_new_handle,
    bool is_shape_agnostic_graph) {
  PT_BRIDGE_BEGIN;
  PT_EAGER_DEBUG("[SHAPE AGNOSTIC] dtensorinfos size : ", dtensorinfos.size());
  bool enable_fast_shape_inf =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE);
  if (dynamic_graph) {
    if (enable_fast_shape_inf && GET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF)) {
      std::vector<size_t> sif_tidx_vec;
      for (auto& idx_tensor_pair : sif_tidx_to_tinfo_map) {
        sif_tidx_vec.push_back(idx_tensor_pair.first);
      }
      std::sort(sif_tidx_vec.begin(), sif_tidx_vec.end());
      for (auto tensor_idx : sif_tidx_vec) {
        auto& ti = sif_tidx_to_tinfo_map.at(tensor_idx);
        // for (auto& tensors : sif_tidx_to_tinfo_map) {
        // auto tensor_idx = tensors.first;
        // auto& ti = tensors.second;

        PT_DYNAMIC_SHAPE_DEBUG(
            "update_patching_table :: before updating tidx : ",
            tensor_idx,
            ", tinfo: ",
            *ti);

        HABANA_ASSERT(
            tidx_to_tensor_map_opt != std::nullopt,
            "nullopt passed as tidx_to_tensor_map_opt");
        const std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map =
            tidx_to_tensor_map_opt->get();

        HABANA_ASSERT(
            tidx_to_tensor_map.count(tensor_idx),
            "Tensor index ",
            tensor_idx,
            " is missing from the computed tidx_to_tensor_map");
        auto new_sizes = tidx_to_tensor_map.at(tensor_idx).sizes().vec();

        std::vector<int64_t> strides(new_sizes.size(), 1);
        for (int64_t i = (int64_t)new_sizes.size() - 1; i > 0; i--) {
          strides[i - 1] *= new_sizes[i] * strides[i];
        }
        ti->set_shape(new_sizes);
        ti->set_strides(strides);
        PT_DYNAMIC_SHAPE_DEBUG(
            "update_patching_table :: after  updating tidx : ",
            tensor_idx,
            ", tinfo: ",
            *ti);
      }
    } else {
      for (size_t i = 0; i < dtensorinfos.size(); ++i) {
        auto& ti = *(dtensorinfos.at(i));
        auto tensor_id = ti.get_tensor_id();
        if (enable_optim_output_sif_ == true &&
            ti.tensor_type() != SHAPE_TENSOR) {
          continue;
        }
        if (!enable_optim_output_sif_) {
          // Frontend STs should be part of m_actual_shapes
          // in lazy mode or when enable_optim_output_sif_ is disabled
          HABANA_ASSERT(
              m_actual_shapes.count(tensor_id), "Tensor ID ", tensor_id);
        }
        if (m_actual_shapes.count(tensor_id)) {
          auto dims = m_actual_shapes.at(tensor_id).get_dims();
          auto syn_shape = ti.get_shape();

          // If there is no change in the new shape values, then
          // do not set the same shape, recalculate strides, etc
          if (dims == syn_shape) {
            continue;
          }

          std::vector<int64_t> strides(dims.size(), 1);
          for (int64_t i = (int64_t)dims.size() - 1; i > 0; i--) {
            strides[i - 1] *= dims[i] * strides[i];
          }
          ti.set_shape(dims);
          ti.set_strides(strides);
        }
      }
    }
  }

  // shape agnostic: count to keep track of number of dtinfos patched for shape
  size_t dtinfos_patched_count = 0;

  // Patch the input buffers
  // Running index on dtensorinfos
  size_t ridx = 0;
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] ridx : ", ridx, " num_inputs : ", num_inputs);
  std::unordered_map<size_t, IValPtrShared> inputIVpshMap;
  for (auto const& input : input_refs) {
    if (input.isTensor()) {
      auto& tensor = input.toTensor();
      auto tmeta{habana::get_tensor_extra_meta(tensor)};
      if (tmeta->has_valid_const_id()) {
        auto impl{tensor.unsafeGetTensorImpl()};
        if (impl->storage().nbytes() == 0) {
          PT_BRIDGE_DEBUG(
              "Skipping patching info for constant id: ",
              tmeta->get_const_id());
          ridx++;
          continue;
        }
      }
      if (tmeta->get_tensor_type() != synTensorType::HOST_TO_DEVICE_TENSOR) {
        auto& ti = *(dtensorinfos.at(ridx));
        ti.set_shape(tensor.sizes().vec());
        ti.set_strides(tensor.strides().vec());
        dtensorinfos.at(ridx)->patch_exact(
            input.toTensor(), is_shape_agnostic_graph);
        IValPtrShared ivpsh = std::make_shared<IVal>(input);
        inputIVpshMap.emplace(ridx, ivpsh);
        if (is_shape_agnostic_graph) {
          std::optional<uint64_t> tensor_offset_opt = std::nullopt;
          const auto& tensor = input.toTensor();
          if (tensor.data_ptr()) {
            tensor_offset_opt = tensor.storage_offset() * tensor.itemsize();
          }
          update_new_tensor(
              ridx,
              synapse_orig_to_new_handle,
              tensor.sizes().vec(),
              tensor_offset_opt);
          dtinfos_patched_count++;
        }
      } else {
        dtensorinfos.at(ridx)->set_host_ptr(tmeta->get_host_ptr());
      }
      ridx++;
    } else if (input.isTensorList()) {
      for (const at::Tensor& t : input.toTensorList()) {
        auto tmeta{habana::get_tensor_extra_meta(t)};
        if (tmeta->has_valid_const_id()) {
          auto impl{t.unsafeGetTensorImpl()};
          if (impl->storage().nbytes() == 0) {
            PT_BRIDGE_DEBUG(
                "Skipping patching info for constant id: ",
                tmeta->get_const_id());
            ridx++;
            continue;
          }
        }
        auto& ti = *(dtensorinfos.at(ridx));
        ti.set_shape(t.sizes().vec());
        ti.set_strides(t.strides().vec());
        dtensorinfos.at(ridx)->patch_exact(t, is_shape_agnostic_graph);
        IValPtrShared ivpsh = std::make_shared<IVal>(t);
        inputIVpshMap.emplace(ridx, ivpsh);
        if (is_shape_agnostic_graph) {
          std::optional<uint64_t> t_offset_opt = std::nullopt;
          if (t.data_ptr()) {
            t_offset_opt = t.storage_offset() * t.itemsize();
          }
          update_new_tensor(
              ridx, synapse_orig_to_new_handle, t.sizes().vec(), t_offset_opt);
          dtinfos_patched_count++;
        }
        ridx++;
      }
    }
  }

  TORCH_CHECK(
      ridx == num_inputs,
      "running index ",
      ridx,
      " mismatch with num_inputs ",
      num_inputs);

  // Patch the duplicates if there are any
  if (num_induplicates) {
    size_t induplicates_index_end = num_inputs + num_induplicates;
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] ridx : ",
        ridx,
        " num_induplicates : ",
        num_induplicates);
    for (; ridx < induplicates_index_end; ridx++) {
      size_t parent_idx = dtensorinfos.at(ridx)->get_parent_index();
      auto parent_ti = dtensorinfos.at(parent_idx);

      auto& ti = *(dtensorinfos.at(ridx));
      ti.set_shape(parent_ti->get_shape());
      ti.set_strides(parent_ti->get_strides());

      dtensorinfos.at(ridx)->patch(*parent_ti, is_shape_agnostic_graph);
      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: Input duplicate : parent idx ",
          parent_idx,
          ", parent buffer ptr ",
          parent_ti->get_buffer());
      if (is_shape_agnostic_graph) {
        auto tshape{parent_ti->get_shape()};
        std::optional<uint64_t> toffset_opt = std::nullopt;
        if (parent_ti->get_buffer()) {
          toffset_opt = parent_ti->get_offset();
        }
        update_new_tensor(
            ridx, synapse_orig_to_new_handle, tshape, toffset_opt);
        dtinfos_patched_count++;
      }
    }
  }

  // To Do - to patch for the shape agnostic graph
  // Patch the dma inputs if there are any
  if (num_dma_inputs) {
    // For DMA inputs patching works in reverse. The tensor is stored
    // within recipe and the corresponding index is stored in the tinfo.
    // The DMA input tensor needs to be populated.
    size_t dma_inputs_index_end =
        num_inputs + num_induplicates + num_dma_inputs;
    for (; ridx < dma_inputs_index_end; ridx++) {
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] dma_inputs ridx : ",
          ridx,
          " shape patching not done!");
      auto& ti = *(dtensorinfos.at(ridx));

      auto dma_cb = ti.get_dma_cb();
      auto tshape{ti.get_shape()};
      at::TensorOptions topts(ti.get_topts());
      TORCH_CHECK(
          topts.dtype() == c10::ScalarType::Int,
          " mismatch in seed tensor dtype, expected ",
          c10::ScalarType::Int,
          " got ",
          topts.dtype());
      auto seed_tensor = at::empty(tshape, topts, ti.get_mf());

      // TODO : The tensor creation should be part of the callback
      dma_cb(ti, seed_tensor);

      ti.patch_exact(seed_tensor);

      IValPtrShared dma_ivpsh = std::make_shared<IVal>(seed_tensor);
      PT_BRIDGE_DEBUG("Persistent tensor for DMA\n");
      dma_inputs.push_back(dma_ivpsh);

      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: DMA input : buffer ptr ",
          dtensorinfos.at(ridx)->get_buffer());
    }
  }

  // Shape tensor patching is already done from name shape map
  ridx = ridx + num_shape_tensors;

  // TODO : Creation of output tensors and associated patching should
  // be part of a member function of RecipeValueSpec

  // Patch persistent intermediates
  // The persistent intermediates are retained in the rv
  const size_t intermediates_start =
      num_inputs + num_induplicates + num_dma_inputs + num_shape_tensors;
  const size_t intermediates_end = intermediates_start + num_intermediates;
  auto intermediate_idx = 0;
  std::unordered_map<size_t, IValPtrShared> intermediateIVpshMap;
  std::vector<at::Tensor> intermediate_tensors;

  auto patch_intermediate_tensor = [&](size_t idx) {
    PtTensorInfo& ti = *(dtensorinfos.at(idx));
    auto tshape{ti.get_shape()};
    if (ti.is_duplicate()) {
      auto ti_parent_index = ti.get_parent_index();
      auto pt_parent_index = ti_parent_index - intermediates_start;
      TORCH_CHECK(
          pt_parent_index < intermediate_tensors.size(),
          "out of range duplicate intermediate tensor index ",
          pt_parent_index,
          " #intermediate_tensors ",
          intermediate_tensors.size());

      auto pt_parent = intermediate_tensors[pt_parent_index];

      auto pt_sizes{ti.get_shape()};
      auto pt_strides{ti.get_strides()};
      long pt_offset = (long)ti.get_offset() / pt_parent.itemsize();
      auto pt_opt_offset = c10::make_optional(pt_offset);

      at::Tensor pt_intermediate =
          at::as_strided(pt_parent, pt_sizes, pt_strides, pt_opt_offset);

      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: Intermediate : Duplicate with shape : ",
          tshape);

      intermediate_tensors.push_back(pt_intermediate);
    } else {
      auto pt_intermediate = habana_helpers::create_empty_tensor(ti);
      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: Intermediate : Creating new with shape : ",
          tshape);

      intermediate_tensors.push_back(pt_intermediate);
    }
    auto& rv_intermediate_tensor = intermediate_tensors.at(intermediate_idx++);

    // Theoretically the data and storage pts of an interim tinfo
    // should not change over iterations. That possibility will only
    // arise if we support freeing of intermediate_tensors after the
    // recipe execution.
    ti.patch(rv_intermediate_tensor);

    IValPtrShared ivpsh = std::make_shared<IVal>(rv_intermediate_tensor);
    intermediate_tensors_ptr->push_back(ivpsh);
    intermediateIVpshMap.emplace(idx, ivpsh);
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache hit :: Intermediate : buffer ptr ",
        rv_intermediate_tensor.data_ptr());
  };

  for (; ridx < intermediates_end; ridx++) {
    if (is_shape_agnostic_graph) {
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] intermediates ridx : ",
          ridx,
          " shape patching will be done at last !");
      continue;
    }
    patch_intermediate_tensor(ridx);
  }

  TORCH_CHECK(
      ridx == intermediates_end,
      "tensor info index ",
      ridx,
      " mismatch with intermediates_end ",
      intermediates_end);

  // The aten_output_num is the total number of outputs
  size_t aten_output_num = num_outputs + num_input_to_outduplicates +
      num_intermediate_to_outduplicates + num_output_to_outduplicates;

  if (is_shape_agnostic_graph) {
    TORCH_CHECK(
        aten_output_num == output_shapes.size(),
        "number of output shapes for patching ",
        output_shapes.size(),
        " is not equal to #outputs ",
        aten_output_num);
  }
  std::unordered_map<size_t, IValPtrShared> outputIVpshMap;
  size_t outputs_end = intermediates_end + num_outputs;
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] ridx : ", ridx, " num_outputs : ", num_outputs);

  auto create_or_use_output_tensor = !allocated_outputs.has_value()
      ? std::function<at::Tensor(const PtTensorInfo&)>(
            [](const PtTensorInfo& ti) {
              return habana_helpers::create_empty_tensor(ti);
            })
      : std::function<at::Tensor(const PtTensorInfo&)>(
            [it = allocated_outputs->begin()](const PtTensorInfo& ti) mutable {
              habana_helpers::update_tensor_layout_and_permutation(*it, ti);
              return *it++;
            });

  HABANA_ASSERT(
      !allocated_outputs.has_value() ||
      (outputs_end - ridx == allocated_outputs->size()));
  for (; ridx < outputs_end; ridx++) {
    auto output_idx = dtensorinfos.at(ridx)->get_output_index();
    TORCH_CHECK(
        output_idx < aten_output_num,
        "output index ",
        output_idx,
        " is greater than #outputs ",
        aten_output_num);
    if (is_shape_agnostic_graph) {
      update_new_tensor(
          ridx, synapse_orig_to_new_handle, output_shapes.at(output_idx));
      dtinfos_patched_count++;
    }
    PtTensorInfo& ti = *(dtensorinfos.at(ridx));
    auto pt_output = create_or_use_output_tensor(ti);
    if (is_shape_agnostic_graph) {
      PT_BACKEND_DEBUG_TENSOR(
          pt_output,
          "output HbInternal address: {:s}"
          " storage address : {:s}"
          " permute: {:s}",
          habana_helpers::FormatTokens::ImplPtr,
          habana_helpers::FormatTokens::DataPtr,
          habana_helpers::FormatTokens::Permutations);
    }
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache hit :: Creating new output with shape : ",
        pt_output.sizes());
    IValPtrShared ivpsh = std::make_shared<IVal>(pt_output);
    aten_outputs.at(output_idx) = ivpsh;

    outputIVpshMap.emplace(ridx, ivpsh);

    // Patch the buffer for the output
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] ridx : ",
        ridx,
        " output tensor storage data pointer : ",
        pt_output.storage().data_ptr().get());
    ti.set_shape(pt_output.sizes().vec());
    ti.set_strides(pt_output.strides().vec());
    ti.patch(pt_output, is_shape_agnostic_graph);
  }

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] ridx : ",
      ridx,
      " num_outduplicates : ",
      num_outduplicates);
  // Patch the duplicates of output that are going back to graph
  size_t outduplicates_end = outputs_end + num_outduplicates;
  if (num_outduplicates) {
    for (; ridx < outduplicates_end; ridx++) {
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] outduplicates ridx : ",
          ridx,
          " shape patching not done!");
      size_t parent_idx = dtensorinfos.at(ridx)->get_parent_index();
      dtensorinfos.at(ridx)->patch(*(dtensorinfos.at(parent_idx)));
    }
  }

  TORCH_CHECK(
      ridx == outduplicates_end,
      "tensor info idx",
      ridx,
      " mismatch with outduplicates_end",
      outduplicates_end);

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] ridx : ",
      ridx,
      " num_input_to_outduplicates : ",
      num_input_to_outduplicates);
  // Patch the input duplicates if there are any
  size_t input_to_outduplicates_end =
      outduplicates_end + num_input_to_outduplicates;
  if (num_input_to_outduplicates) {
    for (; ridx < input_to_outduplicates_end; ridx++) {
      if (is_shape_agnostic_graph) {
        auto output_idx = dtensorinfos.at(ridx)->get_output_index();
        TORCH_CHECK(
            output_idx < aten_output_num,
            "output index ",
            output_idx,
            " is greater than #outputs ",
            aten_output_num);
        size_t parent_idx = dtensorinfos.at(ridx)->get_parent_index();
        auto parent_ti = dtensorinfos.at(parent_idx);
        std::optional<uint64_t> offset_opt = std::nullopt;
        if (parent_ti->get_buffer()) {
          offset_opt = parent_ti->get_offset();
        }
        update_new_tensor(
            ridx,
            synapse_orig_to_new_handle,
            output_shapes.at(output_idx),
            offset_opt);
        dtinfos_patched_count++;
      }
      create_outdup(
          ridx,
          inputIVpshMap,
          "inputIVpshMap",
          aten_outputs,
          is_shape_agnostic_graph);
    }
  }

  TORCH_CHECK(
      ridx == input_to_outduplicates_end,
      "tensor info idx ",
      ridx,
      " mismatch with input_to_outduplicates_end ",
      input_to_outduplicates_end);

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] ridx : ",
      ridx,
      " num_intermediate_to_outduplicates : ",
      num_intermediate_to_outduplicates);

  // Patch the interim duplicates if there are any
  const size_t interim_to_outduplicates_start = input_to_outduplicates_end;
  const size_t interim_to_outduplicates_end =
      input_to_outduplicates_end + num_intermediate_to_outduplicates;
  if (num_intermediate_to_outduplicates) {
    constexpr bool shape_agnostic = false;
    for (; ridx < interim_to_outduplicates_end; ridx++) {
      if (is_shape_agnostic_graph) {
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] intermediate_to_outduplicates ridx : ",
            ridx,
            " shape patching will be done at last !");
        continue;
      }
      create_outdup(
          ridx,
          intermediateIVpshMap,
          "intermediateIVpshMap",
          aten_outputs,
          shape_agnostic);
    }
  }

  TORCH_CHECK(
      ridx == interim_to_outduplicates_end,
      "tensor info idx ",
      ridx,
      " mismatch with interim_to_outduplicates_end ",
      interim_to_outduplicates_end);

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] ridx : ",
      ridx,
      " num_output_to_outduplicates : ",
      num_output_to_outduplicates);
  // Patch the output duplicates if there are any
  size_t output_to_outduplicates_end =
      interim_to_outduplicates_end + num_output_to_outduplicates;
  if (num_output_to_outduplicates) {
    for (; ridx < output_to_outduplicates_end; ridx++) {
      if (is_shape_agnostic_graph) {
        auto output_idx = dtensorinfos.at(ridx)->get_output_index();
        TORCH_CHECK(
            output_idx < aten_output_num,
            "output index ",
            output_idx,
            " is greater than #outputs ",
            aten_output_num);
        update_new_tensor(
            ridx, synapse_orig_to_new_handle, output_shapes.at(output_idx));
        dtinfos_patched_count++;
      }
      create_outdup(
          ridx,
          outputIVpshMap,
          "outputIVpshMap",
          aten_outputs,
          is_shape_agnostic_graph);
    }
  }

  TORCH_CHECK(
      ridx == output_to_outduplicates_end,
      "tensor info idx ",
      ridx,
      " mismatch with output_to_outduplicates_end ",
      output_to_outduplicates_end);

  auto num_tinfos = dtensorinfos.size();
  TORCH_CHECK(
      ridx == num_tinfos,
      "tensor info idx ",
      ridx,
      ", mismatch with num_tinfos",
      num_tinfos);

  if (is_shape_agnostic_graph) {
    // These tinfos not patched
    auto tinfos_not_patched = num_dma_inputs + num_intermediates +
        num_outduplicates + num_intermediate_to_outduplicates;

    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] number of dtinfos patched count : ",
        dtinfos_patched_count);
    TORCH_CHECK(
        dtinfos_patched_count == num_tinfos - tinfos_not_patched,
        "number of dtinfos patched : ",
        dtinfos_patched_count,
        ", mismatch with num_tinfos : ",
        num_tinfos);
  }

  // Patch non-persistent info for shape agnostic flow
  //       outputs and inputs are patched separately
  //
  // Logic
  // Read sif_tidx_to_tinfo_map
  // Map of sif_tidx -> tensor info [to get original syn tensor]
  //
  // Get new shapes from tidx_to_tensor_map
  // Map of sif_tidx -> tensor shapes
  //
  // Now patch the shape information to duplicate synapse tensors
  //
  // ToDo: Patch all non-persistents + outputs + inputs
  //       at one common place for both lazy eager and new eager
  if (is_shape_agnostic_graph) {
    HABANA_ASSERT(
        tidx_to_tensor_map_opt != std::nullopt,
        "nullopt passed as tidx_to_tensor_map_opt");

    const std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map =
        tidx_to_tensor_map_opt->get();
    for (auto& t : sif_tidx_to_tinfo_map) {
      auto sif_tensor_idx = t.first;
      HABANA_ASSERT(
          tidx_to_tensor_map.count(sif_tensor_idx),
          "Tensor index ",
          sif_tensor_idx,
          " is missing from the computed tidx_to_tensor_map");

      auto new_sizes = tidx_to_tensor_map.at(sif_tensor_idx).sizes().vec();
      PT_EAGER_DEBUG(
          "SAG Patching tensor index:",
          sif_tensor_idx,
          " new shape: ",
          new_sizes);

      update_new_tensor(
          ++ridx, // dummy value
          synapse_orig_to_new_handle,
          new_sizes,
          std::nullopt,
          t.second);
    }

    // Patching is done now after updating all intermediate tensor(s) shape
    for (size_t ridx = intermediates_start; ridx < intermediates_end; ridx++) {
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] intermediates ridx : ",
          ridx,
          " patching now done !");
      patch_intermediate_tensor(ridx);
    }

    if (num_intermediate_to_outduplicates) {
      for (size_t ridx = interim_to_outduplicates_start;
           ridx < interim_to_outduplicates_end;
           ridx++) {
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] intermediate_to_outduplicates ridx : ",
            ridx,
            " patching now done !");
        create_outdup(
            ridx,
            intermediateIVpshMap,
            "intermediateIVpshMap",
            aten_outputs,
            false);
      }
    }
  }
  PT_BRIDGE_END;
}

void RecipeValueSpec::update_node_params(
    const std::unordered_map<synNodeId, synNodeId>&
        synapse_node_orig_to_new_handle,
    const std::vector<synNodeId>& syn_node_id_vec,
    const synGraphHandle duplicate_graph_handle,
    std::shared_ptr<std::vector<InferNodeParams>>& node_params_vec_ptr) {
  HABANA_ASSERT(node_params_vec_ptr != nullptr, "node params are empty !");

  const auto node_params_vec = *node_params_vec_ptr;
  HABANA_ASSERT(
      syn_node_id_vec.size() == node_params_vec.size(),
      "Mismatch, Orig graph syn node id vec size: ",
      syn_node_id_vec.size(),
      ", node params vec size: ",
      node_params_vec.size());

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] duplicate graph handle: ", duplicate_graph_handle);

  // Iterate over all orignal graph syn node ids
  // and check if present in the original -> duplicate map
  // if yes, update the node params for duplicate node ids
  for (size_t idx = 0; idx < syn_node_id_vec.size(); ++idx) {
    synNodeId orig_handle = syn_node_id_vec[idx];
    auto synapse_node = synapse_node_orig_to_new_handle.find(orig_handle);
    if (synapse_node == synapse_node_orig_to_new_handle.end()) {
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] origHandle : ",
          orig_handle,
          " not present in the synapse_node_orig_to_new_handle map");
    } else {
      const auto& params = node_params_vec[idx];
      const auto params_data = params.get_data();
      const auto params_size = params.get_size();
      if (params_data && params_size) {
        synNodeId new_handle = synapse_node->second;
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] node index: ",
            idx,
            ", new_handle: ",
            new_handle,
            ", params data: ",
            params_data,
            ", params size: ",
            params_size);
        synapse_helpers::graph::setNodeParams(
            duplicate_graph_handle, new_handle, params_data, params_size);
      }
    }
  }
}

void RecipeValueSpec::populate_syn_tensor_ids(
    const synapse_helpers::graph::recipe_handle& recipe) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_USE_SYN_TENSOR_IDS))
    return;

  HABANA_ASSERT(tensor_ids_.empty());

  auto num_tinfos = dtensorinfos.size();
  tensor_ids_.resize(num_tinfos);
  std::vector<const char*> tensor_names;
  tensor_names.reserve(num_tinfos);

  for (auto& el : dtensorinfos) {
    tensor_names.push_back(el->get_syn_namec_str());
  }

  synStatus status = synTensorRetrieveIds(
      recipe.syn_recipe_handle_,
      tensor_names.data(),
      tensor_ids_.data(),
      num_tinfos);
  if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) {
    PT_BRIDGE_FATAL(
        Logger::formatStatusMsg(status), "synTensorRetrieveIds launch failed");
  }
}

void RecipeValueSpec::patch_launch_info(
    std::vector<synLaunchTensorInfo>& syn_launch_info_vec,
    std::vector<size_t>& external_tensor_info_indexes) const {
  TORCH_CHECK(
      tensor_ids_.size() == dtensorinfos.size(),
      "syn tensor ids are not populated");

  auto& device = HPUDeviceContext::get_device();
  auto record_graph_data = GET_ENV_FLAG_NEW(PT_HPU_POOL_MEM_FRAGMENT_JSON);
  size_t tensor_idx{0};
  for (size_t i = 0; i < dtensorinfos.size(); ++i) {
    const PtTensorInfo& ti = *dtensorinfos[i];
    if (record_graph_data) {
      auto is_output = ti.is_output();
      synapse_helpers::log_synDeviceRecordGraphTensorInfo(
          ti.get_ir_name(), !is_output, is_output, i, ti.get_size());
    }

    switch (ti.tensor_type()) {
      case SHAPE_TENSOR: {
        const auto& tsv = ti.syn_shape();
        syn_launch_info_vec.emplace_back(synLaunchTensorInfo{
            ti.get_syn_namec_str(),
            0,
            ti.tensor_type(),
            {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4]},
            tensor_ids_[tensor_idx++]});
        break;
      }
      case HOST_TO_DEVICE_TENSOR: {
        if (synapse_helpers::memory_reporter_enable()) {
          synapse_helpers::MemoryReporter* reporter =
              device.get_device_memory().get_memory_reporter();
          reporter->getTensorStats()->updateTensorAddressData(
              ti.get_buffer(), ti.get_syn_name(), ti.get_size());
        }
        const auto& tsv = ti.syn_shape();
        syn_launch_info_vec.emplace_back(synLaunchTensorInfo{
            ti.get_syn_namec_str(),
            ti.get_host_ptr(),
            ti.tensor_type(),
            {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4], tsv[5], tsv[6], tsv[7]},
            tensor_ids_[tensor_idx++]});
        break;
      }
      case DATA_TENSOR:
      case DATA_TENSOR_DYNAMIC: {
        if (ti.get_external()) {
          external_tensor_info_indexes.push_back(tensor_idx);
        }
        const auto& tsv = ti.syn_shape();
        syn_launch_info_vec.emplace_back(synLaunchTensorInfo{
            ti.get_syn_namec_str(),
            ti.get_buffer_syn(),
            ti.tensor_type(),
            {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4], tsv[5], tsv[6], tsv[7]},
            tensor_ids_[tensor_idx++]});
        break;
      }
      case DEVICE_SHAPE_TENSOR: {
        const auto& tsv = ti.syn_shape();
        syn_launch_info_vec.emplace_back(synLaunchTensorInfo{
            ti.get_syn_namec_str(),
            ti.get_buffer_syn(),
            ti.tensor_type(),
            {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4]},
            tensor_ids_[tensor_idx++]});
        break;
      }
      case TENSOR_TYPE_MAX:
        TORCH_CHECK(
            false, "Patching of ", ti.tensor_type(), " is not supported yet.");
        break;
      default:
        TORCH_CHECK(false, "Unreachable condition.");
    }
  }
}

namespace {
void MaybePrintDebugInfo(
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    const std::shared_ptr<VecOfIValPtrSh>& intermediate_tensors_ptr,
    const VecOfIValPtrSh& aten_outputs,
    const RecipeLauncher& rl) {
  PT_BRIDGE_BEGIN;
  if (hl_logger::logLevelAtLeast(
          HlLogger::LoggerType::PT_BRIDGE, HLLOG_LEVEL_DEBUG)) {
    PT_BRIDGE_DEBUG(
        "Details of recipe",
        ", #inputs=",
        input_refs.size(),
        ", #intermediates=",
        intermediate_tensors_ptr->size(),
        ", #outputs=",
        aten_outputs.size());

    for (size_t idx{0}; idx < input_refs.size(); idx++) {
      PT_BRIDGE_DEBUG(
          "Input[", idx, "] -> ", habana_helpers::DebugString(input_refs[idx]));
    }
    if (intermediate_tensors_ptr) {
      size_t idx{0};
      for (auto& a : *intermediate_tensors_ptr) {
        PT_BRIDGE_DEBUG(
            "Intermediate[", idx, "] -> ", habana_helpers::DebugString(a));
        idx += 1;
      }
    }
    if (!aten_outputs.empty()) {
      size_t idx{0};
      for (auto& a : aten_outputs) {
        PT_BRIDGE_DEBUG(
            "Output[", idx, "] -> ", habana_helpers::DebugString(a));
        idx += 1;
      }
    }
    PT_BRIDGE_DEBUG(rl);
  }
  PT_BRIDGE_END;
}
} // namespace

size_t get_active_graph_unique_key(const std::string& name) {
  static uint64_t graph_key_suffix_ = -1;
  ++graph_key_suffix_;
  std::hash<std::string> str_hash;
  return (graph_key_suffix_ + str_hash(name));
}

RecipeLauncher::RecipeLauncher(
    const RecipeValueSpec& rvs,
    std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe) {
  ntensorbytes_ = rvs.CalculateNtensorbytes();
  id_ = rvs.id;
  num_inputs_ = rvs.num_inputs;
  num_outputs_ = rvs.num_outputs;
  num_input_to_outduplicates_ = rvs.num_input_to_outduplicates;
  num_intermediate_to_outduplicates_ = rvs.num_intermediate_to_outduplicates;
  graph_name_ = rvs.get_graph_name();
  collective_kernels_info_ = rvs.collective_kernels_info;

  SetRecipe(recipe);
}

RecipeLauncher::RecipeLauncher(std::istream& is, const RecipeValueSpec& rvs) {
  id_ = rvs.id;
  num_inputs_ = rvs.num_inputs;
  num_outputs_ = rvs.num_outputs;
  num_input_to_outduplicates_ = rvs.num_input_to_outduplicates;
  num_intermediate_to_outduplicates_ = rvs.num_intermediate_to_outduplicates;
  graph_name_ = rvs.get_graph_name();

  collective_kernels_info_ = rvs.collective_kernels_info;

  using namespace serialization;
  bool valid_recipe_handle = false;
  deserialize(is, valid_recipe_handle);

  if (valid_recipe_handle) {
    recipe_ = std::make_shared<synapse_helpers::graph::recipe_handle>();
    deserialize(is, recipe_->recipe_name_);
    deserialize(is, recipe_->graph_is_empty_);
    recipe_->in_execution_phase_ = false;
  }
  deserialize(is, workspace_size_);
  deserialize(is, ntensorbytes_);
}

void RecipeLauncher::Launch(
    synapse_helpers::hpuStream_t hpu_stream,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    std::shared_ptr<VecOfIValPtrSh>& intermediate_tensors_ptr,
    const VecOfIValPtrSh& aten_outputs,
    std::vector<synLaunchTensorInfo>& syn_launch_info,
    std::vector<size_t>& external_tensor_info_indexes,
    const VecOfIValPtrSh& dma_inputs) {
  PT_BRIDGE_BEGIN;
  MaybePrintDebugInfo(
      input_refs, intermediate_tensors_ptr, aten_outputs, *this);

  auto& device = HPUDeviceContext::get_device();
  auto& stream_handle = device.get_stream(hpu_stream);

  std::vector<at::Tensor> ptRefs;
  std::vector<at::Tensor> outPtRefs;
  std::vector<synapse_helpers::device_ptr> outDevPtr;

  size_t active_graph_key_ = 0;

  if (device.IsStreamASyncEnabled()) {
    // Get the reference to the tensor it is operating on to prevent
    // it from being deallocated while the operation is still in flight.
    std::vector<synapse_helpers::device_ptr> inDevPtr;
    inDevPtr.reserve(num_inputs_);
    for (auto& input : input_refs) {
      if (input.isTensor()) {
        at::Tensor tensor = input.toTensor();
        ptRefs.push_back(std::move(tensor));
        inDevPtr.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
            input.toTensor().storage().data_ptr().get()));
      }
    }
    if (dma_inputs.size() > 0) {
      for (auto& dma_input : dma_inputs) {
        TORCH_CHECK(
            dma_input->isTensor(), "Only tensor is supported as dma_input");
        at::Tensor tensor = dma_input->toTensor();
        ptRefs.push_back(std::move(tensor));
        inDevPtr.push_back(
            reinterpret_cast<uint64_t>((dma_input->toTensor()).data_ptr()));
      }
    }
    // wait for input DMA to complete before launching the compute.
    device.add_wait_events_on_stream(inDevPtr, stream_handle);

    // Hold on to the pytorch tensors for the intermediates untill the recipe
    // execution completes
    if (intermediate_tensors_ptr != nullptr &&
        intermediate_tensors_ptr->size() > 0) {
      for (auto& intermediate_tensor : *intermediate_tensors_ptr) {
        at::Tensor tensor = intermediate_tensor->toTensor();
        ptRefs.push_back(std::move(tensor));
      }
    }

    outDevPtr.reserve(
        num_inputs_ + num_outputs_ + num_input_to_outduplicates_ +
        num_intermediate_to_outduplicates_);
    for (auto& output : aten_outputs) {
      if (output && output->isTensor()) {
        at::Tensor tensor = output->toTensor();
        outDevPtr.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
            tensor.storage().data_ptr().get()));
        outPtRefs.push_back(std::move(tensor));
      }
    }

    // Write after read dependancy, add event for the inputs. So all the
    // tensors being written to will appear in the read side.
    outDevPtr.insert(outDevPtr.end(), inDevPtr.begin(), inDevPtr.end());

    std::vector<synapse_helpers::shared_event> ext_events;
    for (auto external_idx : external_tensor_info_indexes) {
      synLaunchTensorInfo& ti = syn_launch_info.at(external_idx);
      PT_BRIDGE_DEBUG("Map event to external tensor ", ti.tensorName);
      ext_events.emplace_back(device.map_event_to_tensor(
          stream_handle, recipe_->syn_recipe_handle_, &ti, []() {}));

      // Remove collective kenrel inputs from outDevPtr since they will be
      // signaled from the graph (if they are external)
      PT_BRIDGE_DEBUG(
          "Remove tensor ",
          ti.tensorName,
          " address ",
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          reinterpret_cast<void*>(ti.pTensorAddress),
          " from outDevPtr since it is an external tensor");
      outDevPtr.erase(
          std::remove(outDevPtr.begin(), outDevPtr.end(), ti.pTensorAddress),
          outDevPtr.end());
    }

    auto& recipe_counter = device.get_active_recipe_counter();

    std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
    {
      synapse_helpers::TimeScope ts(std::move(time_slot_));
      if (recipe_) {
        if (synapse_helpers::memory_reporter_enable()) {
          active_graph_key_ =
              get_active_graph_unique_key(recipe_->recipe_name_);
          if (active_graph_key_ > 0) {
            synapse_helpers::MemoryReporter* reporter =
                device.get_device_memory().get_memory_reporter();
            reporter->getGraphStats()->addGraph(
                active_graph_key_,
                id_,
                num_inputs_,
                num_outputs_,
                ntensorbytes_,
                workspace_size_,
                workspace_size_);
          }
        }

        synapse_helpers::graph::launch(
            device,
            *recipe_,
            workspace_size_,
            syn_launch_info,
            address_lock,
            ext_events,
            stream_handle,
            active_graph_key_);
        habana_lazy::log_dev_mem_stats(
            "Post-Launch", graph_name_, workspace_size_);
      } else {
        PT_BRIDGE_DEBUG("Skipping recipe launch. empty recipe");
      }
    }
    recipe_counter.increase();

    // register events for external tensors on compute
    for (size_t i = 0; i < ext_events.size(); ++i) {
      device.register_producer_on_stream(stream_handle, ext_events.at(i));
    }

    if (!(common::IsRecordStreamEnabled() &&
          GET_ENV_FLAG_NEW(PT_HPU_USE_LAUNCH_RECORD_STREAM))) {
      // Use wrapper for resources that must survive async part of the compute.
      struct ResourceHolder {
        std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_id_;
        std::vector<at::Tensor> output_tensors_;
        std::vector<at::Tensor> input_tensors_;
        std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
        synapse_helpers::active_recipe_counter* recipe_counter_ptr;
        size_t active_graph_key_;
      };
      auto resource_holder = std::shared_ptr<ResourceHolder>(
          new ResourceHolder(), [](ResourceHolder* resource_holder) {
            auto recipe_counter_ptr = resource_holder->recipe_counter_ptr;
            auto recipe_handle = resource_holder->recipe_id_;
            delete resource_holder;
            recipe_counter_ptr->decrease_and_notify();
            if (synapse_helpers::memory_reporter_enable() &&
                resource_holder->active_graph_key_ > 0) {
              auto& device = HPUDeviceContext::get_device();
              synapse_helpers::MemoryReporter* reporter =
                  device.get_device_memory().get_memory_reporter();
              reporter->getGraphStats()->removeLiveGraph(
                  resource_holder->active_graph_key_);
            }
            towl::emitRecipeFinished(recipe_handle.get());
            PT_LAZY_DEBUG("call decrease and notify of recipe_counter");
          });
      // recipe_id_ needs to be passed to done_cb to ensure its lifetime until
      // corresponding recipe is finished on stream
      const auto& recipe_ptr = recipe_;
      resource_holder->recipe_id_ = recipe_ptr;
      if (not common::IsRecordStreamNoHolderEnabled()) {
        resource_holder->output_tensors_ = outPtRefs;
        resource_holder->address_lock = std::move(address_lock);
        resource_holder->input_tensors_ = ptRefs;
      }
      resource_holder->recipe_counter_ptr = &recipe_counter;
      resource_holder->active_graph_key_ = active_graph_key_;
      // ResourceHolder could be used directly as callback, if we would only
      // implement operator(), but copying of ResourceHolder would result in
      // copying of all shared_ptr stored inside (including std::vector). To
      // make sharing more lightweight we hide ResourceHolder behind one
      // shared_ptr. This indirection allows us to maintain only one shared
      // reference.
      auto cleanup_callback = [resource_holder]() mutable {
        resource_holder.reset();
      };

      device.register_producer_on_stream(
          std::move(outDevPtr), stream_handle, cleanup_callback);
      // Launch collective ops
      collective_kernels_info_->Launch(true, cleanup_callback);
    } else {
      // Use wrapper for resources that must survive async part of the compute.
      struct ResourceHolder {
        std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_id_;
        std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
        synapse_helpers::active_recipe_counter* recipe_counter_ptr;
        size_t active_graph_key_;
      };
      auto resource_holder = std::shared_ptr<ResourceHolder>(
          new ResourceHolder(), [](ResourceHolder* resource_holder) {
            auto recipe_counter_ptr = resource_holder->recipe_counter_ptr;
            auto recipe_handle = resource_holder->recipe_id_;
            delete resource_holder;
            recipe_counter_ptr->decrease_and_notify();
            if (synapse_helpers::memory_reporter_enable() &&
                resource_holder->active_graph_key_ > 0) {
              auto& device = HPUDeviceContext::get_device();
              synapse_helpers::MemoryReporter* reporter =
                  device.get_device_memory().get_memory_reporter();
              reporter->getGraphStats()->removeLiveGraph(
                  resource_holder->active_graph_key_);
            }
            towl::emitRecipeFinished(recipe_handle.get());
            PT_LAZY_DEBUG("call decrease and notify of recipe_counter");
          });
      // recipe_id_ needs to be passed to done_cb to ensure its lifetime until
      // corresponding recipe is finished on stream
      const auto& recipe_ptr = recipe_;
      resource_holder->recipe_id_ = recipe_ptr;
      if (not common::IsRecordStreamNoHolderEnabled()) {
        resource_holder->address_lock = std::move(address_lock);
      }
      resource_holder->recipe_counter_ptr = &recipe_counter;
      resource_holder->active_graph_key_ = active_graph_key_;
      // ResourceHolder could be used directly as callback, if we would only
      // implement operator(), but copying of ResourceHolder would result in
      // copying of all shared_ptr stored inside (including std::vector). To
      // make sharing more lightweight we hide ResourceHolder behind one
      // shared_ptr. This indirection allows us to maintain only one shared
      // reference.
      auto cleanup_callback = [resource_holder]() mutable {
        resource_holder.reset();
      };

      device.register_producer_on_stream(
          std::move(outDevPtr), stream_handle, cleanup_callback);
      for (auto data_ptr : inDevPtr) {
        device.get_device_memory().recordStream(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            reinterpret_cast<void*>(data_ptr),
            hpu_stream);
      }
      for (auto data_ptr : outDevPtr) {
        device.get_device_memory().recordStream(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            reinterpret_cast<void*>(data_ptr),
            hpu_stream);
      }
      // Launch collective ops
      collective_kernels_info_->Launch(true, cleanup_callback);
    }

  } else {
    std::vector<synapse_helpers::shared_event> ext_events;
    std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
    synapse_helpers::TimeScope ts(std::move(time_slot_));
    if (recipe_) {
      if (synapse_helpers::memory_reporter_enable()) {
        active_graph_key_ = get_active_graph_unique_key(recipe_->recipe_name_);
        if (active_graph_key_ > 0) {
          synapse_helpers::MemoryReporter* reporter =
              device.get_device_memory().get_memory_reporter();
          reporter->getGraphStats()->addGraph(
              active_graph_key_,
              id_,
              num_inputs_,
              num_outputs_,
              ntensorbytes_,
              workspace_size_,
              workspace_size_);
        }
      }

      synapse_helpers::graph::launch(
          device,
          *recipe_,
          workspace_size_,
          syn_launch_info,
          address_lock,
          ext_events,
          stream_handle);

      habana_lazy::log_dev_mem_stats(
          "Post-Launch", graph_name_, workspace_size_);
    }
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");

    // Launch collective ops
    collective_kernels_info_->Launch(false, [] {});

    if (synapse_helpers::memory_reporter_enable() && active_graph_key_ > 0) {
      auto& device = HPUDeviceContext::get_device();
      synapse_helpers::MemoryReporter* reporter =
          device.get_device_memory().get_memory_reporter();
      reporter->getGraphStats()->removeLiveGraph(active_graph_key_);
    }
  }

  num_launches++;
  PT_BRIDGE_END;
}

std::shared_ptr<habana_helpers::DynamicBucketInfo> DynamicBucketInfoMap::get(
    std::shared_ptr<RecipeArgumentSpec>& key) {
  std::lock_guard<std::mutex> lg(mutex_);
  if (exists(key)) {
    return map_[key];
  }
  return {nullptr};
}

size_t DynamicBucketInfoMap::Size() const {
  size_t size = 0;
  for (auto const& p : map_) {
    size += p.second->Size();
  }
  return size;
}

size_t DynamicBucketInfoMap::HistSize() const {
  size_t size = 0;
  for (auto const& p : map_) {
    size += p.second->HistSize();
  }
  return size;
}

void DynamicBucketInfoMap::clear() {
  std::lock_guard<std::mutex> lg(mutex_);
  map_.clear();
}

void DynamicBucketInfoMap::save_ds_checkpoint(std::ofstream& ds_checkpoint) {
  std::stringstream os;
  DynamicBucketInfoMap::get_instance().Serialize(os);
  ds_checkpoint << os.rdbuf();
  ds_checkpoint.close();

  const bool is_ds_cache_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DISK_CACHE_FOR_DSD);

  if (is_ds_cache_enabled) {
    HPUDeviceContext::recipe_cache().Serialize();
  }
}

void DynamicBucketInfoMap::load_ds_checkpoint(std::ifstream& ds_checkpoint) {
  std::stringstream is;

  is << ds_checkpoint.rdbuf();
  ds_checkpoint.close();
  DynamicBucketInfoMap::get_instance().Deserialize(is);

  const bool is_ds_cache_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DISK_CACHE_FOR_DSD);
  if (is_ds_cache_enabled) {
    HPUDeviceContext::recipe_cache().Deserialize();
  }
}

void DynamicBucketInfoMap::Serialize(std::ostream& os) const {
  using namespace serialization;
  int map_size = 0;
  for (auto const& p : map_) {
    if (!p.first->hasToken()) {
      map_size++;
    }
  }
  serialize(os, static_cast<int>(map_size));
  for (auto const& p : map_) {
    if (!p.first->hasToken()) {
      p.first->Serialize(os);
      p.second->Serialize(os);
    }
  }
}

void DynamicBucketInfoMap::Deserialize(std::istream& is) {
  using namespace serialization;
  int map_size = 0;
  deserialize(is, map_size);
  for (int i = 0; i < map_size; ++i) {
    std::shared_ptr<RecipeArgumentSpec> key =
        std::make_shared<RecipeArgumentSpec>(is);
    std::shared_ptr<habana_helpers::DynamicBucketInfo> value =
        std::make_shared<habana_helpers::DynamicBucketInfo>(is);
    add(key, value);
  }
}

void DynamicBucketInfoMap::DumpBucketMemoryStat() {
  PT_HOSTSTAT_DEBUG(
      "Size of Dynamic Bucket: ",
      synapse_helpers::get_mem_str(
          DynamicBucketInfoMap::get_instance().Size()));
}

void DynamicBucketInfoMap::DumpHistoryMemoryStat() {
  PT_HOSTSTAT_DEBUG(
      "Size of Dynamic Bucket History: ",
      synapse_helpers::get_mem_str(
          DynamicBucketInfoMap::get_instance().HistSize()));
}

void RecipeCacheLRU::DumpRecipeMemoryStat() {
  PT_HOSTSTAT_DEBUG(
      "Size of Recipe LRU Cache: ",
      synapse_helpers::get_mem_str(HPUDeviceContext::recipe_cache().Size()));
}

void RecipeCacheLRU::DumpSynapseRecipeMemoryStat() {
  PT_HOSTSTAT_DEBUG(
      "Size of Synapse Recipe: ",
      synapse_helpers::get_mem_str(
          HPUDeviceContext::recipe_cache().SynapseRecipeSize()));
}

void RecipeCacheLRU::DumpDynamicShapeMemoryStat() {
  PT_HOSTSTAT_DEBUG(
      "DS MemoryStats:- Bucket::",
      synapse_helpers::get_mem_str(DynamicBucketInfoMap::get_instance().Size()),
      ", History::",
      synapse_helpers::get_mem_str(
          DynamicBucketInfoMap::get_instance().HistSize()),
      ", Recipe::",
      synapse_helpers::get_mem_str(HPUDeviceContext::recipe_cache().Size()),
      ", SynapseRecipe::",
      synapse_helpers::get_mem_str(
          HPUDeviceContext::recipe_cache().SynapseRecipeSize()));
}

void DynamicBucketInfoMap::add(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<habana_helpers::DynamicBucketInfo>& val) {
  std::lock_guard<std::mutex> lg(mutex_);
  map_.emplace(key, val);
}

void DynamicBucketInfoMap::refine_graph(size_t graph_key) {
  for (auto& p : map_) {
    auto dbipsh = p.second;
    if (dbipsh->GetGraphKey() == graph_key) {
      dbipsh->CheckForSplitBucket(dbipsh);
      return;
    }
  }
  TORCH_CHECK(
      false, "Graph key ", graph_key, " is missing from DynamicBucketInfoMap");
}

void ClearDynamicBucketRecipeInfo() {
  HPUDeviceContext::recipe_cache().clear();
  DynamicBucketInfoMap::get_instance().clear();
}
} // namespace habana
