/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_bridge/kernel/hpu_habana_cache.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "habana_device/HPUAllocator.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"

#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_helpers/tensor_info.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_serialization/cache_version.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/serializers.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/event.h"

#include "habana_kernels/hccl_kernels.h"

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

namespace {
template <typename T>
std::vector<int64_t> ptr_array_indices(
    const std::vector<std::shared_ptr<T>>& elements,
    const std::vector<std::shared_ptr<T>>& src_array) {
  std::vector<int64_t> indices;
  indices.reserve(elements.size());
  for (const auto& e : elements) {
    if (e.get() == nullptr) {
      indices.push_back(-1);
      continue;
    }
    auto iter = std::find(src_array.begin(), src_array.end(), e);
    TORCH_CHECK(iter != src_array.end(), "Failed to find element in src_array");
    indices.push_back(std::distance(src_array.begin(), iter));
  }
  return indices;
}

template <typename T>
std::vector<std::shared_ptr<T>> indices_array_to_ptr_array(
    const std::vector<int64_t>& indices,
    const std::vector<std::shared_ptr<T>>& src_array) {
  std::vector<std::shared_ptr<T>> ptr_array;
  ptr_array.resize(indices.size());
  for (const auto& idx : indices) {
    if (idx == -1) {
      continue;
    }

    TORCH_CHECK(
        idx <= (int64_t)src_array.size(),
        "idx ",
        idx,
        " out of range. src array size = ",
        src_array.size());
    ptr_array.push_back(src_array.at(idx));
  }
  return ptr_array;
}
} // namespace
namespace habana {

// static initializations
std::mutex RecipeCacheLRU::mutex_;
RecipeCacheLRU* RecipeCacheLRU::instance_ = nullptr;
size_t RecipeCacheLRU::max_size_ = PGM_LRU_MAX_LAZY_NRECIPES;

size_t RecipeValueSpec::count = 0;
size_t RecipeValueSpec::recipe_count = 0;
size_t RecipeValueSpec::dynamic_recipe_count = 0;
size_t RecipeValueSpec::total_recipe_ntbytes = 0;
size_t RecipeValueSpec::compile_count = 0;
size_t RecipeValueSpec::launch_count = 0;

HbCas::HbCas(bool with_grad, at::ArrayRef<c10::IValue> inputs) {
  p_cas = std::make_shared<torch::jit::CompleteArgumentSpec>(with_grad, inputs);
}

RecipeArgumentSpec::RecipeArgumentSpec(
    at::ArrayRef<torch::jit::IValue> input_refs,
    const size_t& graphKey,
    const std::string& op_strs)
    : cas(false, input_refs), opstrs(op_strs), graph_hash_code(graphKey) {
  hash_code = graph_hash_code;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    size_t perm_hash_code = habana_lazy::ComputePermutationHashCode(input_refs);
    hash_code = at::hash_combine(hash_code, perm_hash_code);
  }
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
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    size_t perm_hash_code = habana_lazy::ComputePermutationHashCode(input_refs);
    hash_code = at::hash_combine(hash_code, perm_hash_code);
  }
  dynamic_hash_code = hash_code;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    bool with_grad,
    at::ArrayRef<torch::jit::IValue> input_refs,
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const size_t& graphKey,
    const std::string& op_strs)
    : cas(with_grad, input_refs), opstrs(op_strs), hash_code(cas.hashCode()) {
  cargspec_hash_code = cas.hashCode();
  graph_hash_code = graphKey;
  hash_code = at::hash_combine(hash_code, graph_hash_code);
  hash_code = at::hash_combine(hash_code, irgraph->outputs().size());
  hash_code = habana_helpers::hash_combine_scalars(hash_code, input_refs);

  ComputeOffsetHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, offset_hash_code);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    size_t perm_hash_code = habana_lazy::ComputePermutationHashCode(input_refs);
    hash_code = at::hash_combine(hash_code, perm_hash_code);
  }
  /*Add deterministic flag as well here*/
  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    torch::jit::graph_node_list graph_nodes = irgraph->nodes();
    for (auto node : graph_nodes) {
      auto node_qual_str = node->kind().toQualString();
      std::string opname(node_qual_str);
      /*Ignore the const & meta nodes*/
      if (node->kind().is_prim() || HabanaMetaOpList::isHabanaMetaOp(opname)) {
        continue;
      }

      auto one = torch::jit::attr::alpha;
      hash_code = at::hash_combine(hash_code, node->f(one));
      PT_BRIDGE_DEBUG("Jit Sysnapse Cache deterministic: ", node->f(one));
      auto node_name = node->kind().toQualString();
      PT_BRIDGE_DEBUG("Node Name: ", node_name);
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

  if (htensor_wbuff) {
    synStatus status;
    auto& device = synapse_helpers::HPURegistrar::get_device();
    auto device_id = device.id();
    status = synHostFree(device_id, (void*)(htensor_wbuff), 0);
    if (status != synSuccess)
      PT_BRIDGE_DEBUG("host-free failed");
  }

  if (nullptr != tensor_names) {
    delete[] tensor_names;
  }
  if (nullptr != tensor_ids) {
    delete[] tensor_ids;
  }
}

std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v) {
  O << '\n'
    << "---- recipe details :: begin" << '\n'
    << " <id : " << v.id << "> "
    << " <iteration : " << v.iter_idx << "> "
    << " <addr : " << v.recipe.get() << "> "
    << " <use_count : " << v.recipe.use_count() << ">"
    << " <num_launches : " << v.num_launches << ">" << '\n';

  O << " ntensorbytes : " << synapse_helpers::get_mem_str(v.ntensorbytes)
    << '\n';
  O << " workspace    : " << synapse_helpers::get_mem_str(v.workspace_size)
    << '\n';

  O << " #inputs                        : " << v.num_inputs << '\n';
  O << " #num_tensors                   : " << v.num_tensors << '\n';

  O << " #aten_outputs                  : ";
  if (v.aten_outputs) {
    O << v.aten_outputs->size() << '\n';
  } else {
    O << "not populated yet" << '\n';
  }

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

  if (v.dtensorinfos) {
    O << "dtensorinfos #" << v.dtensorinfos->size() << "::";
    O << '\n';
    size_t idx{0};
    for (auto& a : *v.dtensorinfos) {
      O << idx++ << " : ";
      O << *a << '\n';
    }
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

void RecipeValueSpec::print_hbuff(
    size_t buf_idx,
    std::ofstream& out,
    size_t iteration_count,
    int numel) {
  float* wb = reinterpret_cast<float*>(htensor_wbuff);
  unsigned buf_size = dtensorinfos->at(buf_idx)->get_size();

  out << "iteration " << iteration_count << " : <"
      << ((buf_idx >= num_inputs) ? "output" : "input") << "> :: < "
      << dtensorinfos->at(buf_idx)->get_ir_name() << " : "
      << "shape [" << dtensorinfos->at(buf_idx)->get_shape() << "] : "
      << "numel " << dtensorinfos->at(buf_idx)->get_numel() << " : "
      << "size (" << buf_size << " b) >";
  out << "<buffer" << '[' << buf_idx << ']' << "@"
      << dtensorinfos->at(buf_idx)->get_buffer() << ">";

  const unsigned max_numel = buf_size / sizeof(float);
  unsigned lim{max_numel};
  if (numel >= 0) {
    lim = std::min(lim, (unsigned)numel);
  }

  size_t line_items_num = 8;
  size_t j = 0;
  for (j = 0; j < lim; j++) {
    out << (j % line_items_num ? ' ' : '\n') << std::showpoint << std::setw(10)
        << std::fixed << std::right << wb[j];
  }

  if (lim && lim < max_numel)
    out << (j % line_items_num ? ' ' : '\n') << "...";

  out << '\n';
  if (lim > 0) {
    out << "--------------------" << '\n';
  }
}

void RecipeValueSpec::d2h_dbuff(size_t buf_idx) {
  TORCH_CHECK(num_tinfos > buf_idx, "buf_idx is out of range");

  unsigned buf_size = dtensorinfos->at(buf_idx)->get_size();
  if (buf_size > htensor_wbuff_size) {
    buf_size = htensor_wbuff_size;
  }
  PT_BRIDGE_DEBUG("tensor dump will write ", htensor_wbuff_size, " bytes");

  auto& device = synapse_helpers::HPURegistrar::get_device();
  std::atomic<bool> copyDone{false};
  auto syn_error = device.copy_data_to_host(
      (uint64_t)dtensorinfos->at(buf_idx)->get_buffer(),
      (void*)htensor_wbuff,
      dtensorinfos->at(buf_idx)->get_buffer_start_syn(),
      buf_size,
      [&copyDone]() { copyDone = true; });
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }
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
    << synapse_helpers::get_mem_str(ntensorbytes) << "\n "
    << (dynamic_graph ? "dynamic graph" : "static graph") << " - "
    << (is_refined ? "refined" : "original");
  return O.str();
}

std::string RecipeValueSpec::digest_str() {
  std::ostringstream O;
  auto& device = synapse_helpers::HPURegistrar::get_device();
  O << "Recipe digest : total size of graph recipes "
    << synapse_helpers::get_mem_str(RecipeValueSpec::total_recipe_ntbytes)
    << '\n';
  auto rv_hit_count = device.get_recipe_handle_cache().getHitCount(key);
  if (-1 != rv_hit_count) {
    // Hit count needs to be enabled with
    // PT_HABANA_MAX_RECIPE_HIT_COUNT=<positive number>
    O << " #hits " << rv_hit_count << '\n';
  }
  O << " #graph_recipes " << recipe_count << " (#static "
    << (recipe_count - dynamic_recipe_count) << ", #dynamic "
    << dynamic_recipe_count << ')' << '\n'
    << " #eager_recipes " << device.get_recipe_handle_cache().getCount();

  return O.str();
}

int RecipeValueSpec::update_hit_count() {
  auto& device = synapse_helpers::HPURegistrar::get_device();
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

RecipeValueSpec::RecipeValueSpec(std::istream& is) {
  using namespace serialization;
  bool valid_recipe_handle = false;
  deserialize(is, valid_recipe_handle);

  if (valid_recipe_handle) {
    recipe = std::make_shared<synapse_helpers::graph::recipe_handle>();
    deserialize(is, recipe->recipe_name_);
    deserialize(is, recipe->graph_is_empty_);
    recipe->in_execution_phase_ = false;
  }

  int info_size = 0;
  deserialize(is, info_size);
  dtensorinfos = std::make_shared<std::vector<PtTensorInfoShared>>();
  dtensorinfos->reserve(info_size);
  for (int i = 0; i < info_size; ++i) {
    dtensorinfos->emplace_back(std::make_shared<PtTensorInfo>(is));
  }

  deserialize(is, workspace_size);
  deserialize(is, htensor_wbuff);
  deserialize(is, htensor_wbuff_size);
  deserialize(is, id);
  deserialize(is, iter_idx);
  deserialize(is, num_tinfos);
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
  deserialize(is, ntensorbytes);
  deserialize(is, key);
  deserialize(is, graph_key);
  deserialize(is, opstrs);
  deserialize(is, header);
  deserialize(is, num_tensors);
  deserialize(is, graph_name);
  tensor_ids = new uint64_t[num_tensors];
  for (size_t i = 0; i < num_tensors; i++) {
    deserialize(is, tensor_ids[i]);
  }
  tensor_names = new const char*[num_tensors];
  for (size_t i = 0; i < num_tensors; i++) {
    char* tmp;
    deserialize(is, tmp);
    tensor_names[i] = tmp;
  }
  deserialize(is, dynamic_graph);
  deserialize(is, is_refined);
  deserialize(is, is_refined_wirt);
  deserialize(is, count);
  deserialize(is, total_recipe_ntbytes);
  // deserialize(is, get_use_flag());
  size_t num_collective_kernels = 0;
  deserialize(is, num_collective_kernels);
  for (size_t i = 0; i < num_collective_kernels; i++) {
    auto kernel_info =
        std::make_shared<habana_helpers::collective_kernel_info>();

    std::vector<int64_t> input_indices;
    deserialize(is, input_indices);
    kernel_info->input_tensor_infos =
        indices_array_to_ptr_array(input_indices, *dtensorinfos);

    std::vector<int64_t> output_indices;
    deserialize(is, output_indices);
    kernel_info->output_tensor_infos =
        indices_array_to_ptr_array(output_indices, *dtensorinfos);

    std::string guid;
    int device_id;
    c10::ScalarType scalar_type;
    deserialize(is, guid);
    deserialize(is, device_id);
    deserialize(is, scalar_type);
    c10::OperatorName op_name(guid, "");
    HabanaOperatorPtr habana_kernel =
        KernelRegistry().get(device_id, op_name, scalar_type);
    auto collective_kernel =
        std::dynamic_pointer_cast<CollectiveOperator>(habana_kernel);
    TORCH_CHECK(
        collective_kernel,
        "Failed to find collective kernel for ",
        guid,
        "during recipe load from disk");

    collective_kernel->Deserialize(is);
    kernel_info->kernel = collective_kernel;
    collective_kernels_info.emplace_back(kernel_info);
  }
}

void RecipeValueSpec::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, recipe != nullptr);
  if (recipe) {
    serialize(os, recipe->recipe_name_);
    serialize(os, recipe->graph_is_empty_);
  }
  serialize(os, static_cast<int>(dtensorinfos.get()->size()));
  for (PtTensorInfoShared& tInfo : *dtensorinfos) {
    tInfo->Serialize(os);
  }
  serialize(os, workspace_size);
  serialize(os, htensor_wbuff);
  serialize(os, htensor_wbuff_size);
  serialize(os, id);
  serialize(os, iter_idx);
  serialize(os, num_tinfos);
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
  serialize(os, ntensorbytes);
  serialize(os, key);
  serialize(os, graph_key);
  serialize(os, opstrs);
  serialize(os, header);
  serialize(os, num_tensors);
  serialize(os, graph_name);
  for (size_t i = 0; i < num_tensors; i++) {
    serialize(os, tensor_ids[i]);
  }
  for (size_t i = 0; i < num_tensors; i++) {
    serialize(os, tensor_names[i]);
  }

  serialize(os, dynamic_graph);
  serialize(os, is_refined);
  serialize(os, is_refined_wirt);
  serialize(os, count);
  serialize(os, total_recipe_ntbytes);
  serialize(os, collective_kernels_info.size());
  for (const auto& collective_kernel : collective_kernels_info) {
    auto input_indices =
        ptr_array_indices(collective_kernel->input_tensor_infos, *dtensorinfos);
    serialize(os, input_indices);

    auto output_indices = ptr_array_indices(
        collective_kernel->output_tensor_infos, *dtensorinfos);
    serialize(os, output_indices);

    serialize(os, collective_kernel->kernel->GetGuid());
    serialize(os, collective_kernel->kernel->GetDeviceId());
    serialize(os, collective_kernel->kernel->GetScalarType());
    collective_kernel->kernel->Serialize(os);
  }
}

void RecipeValueSpec::update_patching_table(
    at::ArrayRef<torch::jit::IValue>& input_refs,
    std::shared_ptr<std::vector<IValPtrShared>>& intermediate_tensors_ptr,
    std::shared_ptr<std::vector<IValPtrShared>>& dma_inputs_ptr,
    const habana::IdShapeMap& m_actual_shapes,
    std::optional<
        std::reference_wrapper<const std::unordered_map<int64_t, at::Tensor>>>
        tidx_to_tensor_map_opt) {
  PT_BRIDGE_BEGIN;
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
        auto& ti = sif_tidx_to_tinfo_map[tensor_idx];
        // for (auto& tensors : sif_tidx_to_tinfo_map) {
        // auto tensor_idx = tensors.first;
        // auto& ti = tensors.second;

        PT_TEST_DEBUG(
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
        PT_TEST_DEBUG(
            "update_patching_table :: after  updating tidx : ",
            tensor_idx,
            ", tinfo: ",
            *ti);
      }
    } else {
      for (size_t i = 0; i < dtensorinfos->size(); ++i) {
        auto& ti = *(dtensorinfos->at(i));
        auto tensor_id = ti.get_tensor_id();
        HABANA_ASSERT(m_actual_shapes.count(tensor_id));
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

  auto create_empty_tensor{[](const PtTensorInfo& ti) -> at::Tensor {
    auto pt_tensor = at::empty(ti.get_shape(), ti.get_topts(), ti.get_mf());
    auto hb_internal_tensor = habana_lazy::GetHbInternalTensorImpl(pt_tensor);
    PT_BRIDGE_DEBUG(
        "Cache created a BE tensor, HbInternal address: ", hb_internal_tensor);
    TORCH_CHECK(
        hb_internal_tensor != nullptr,
        "Tensor for ",
        ti.get_ir_name(),
        " does not have HbInternalTensor");
    auto internal_lf = hb_internal_tensor->GetTensorLayout();
    auto internal_lf_new = ti.getHbInternalLayoutFormat();
    if (internal_lf != internal_lf_new) {
      PT_BRIDGE_DEBUG(
          "For ",
          ti.get_ir_name(),
          " updating HbInternalTensorImpl layout from ",
          internal_lf,
          " to ",
          internal_lf_new);
      hb_internal_tensor->SetTensorLayout(internal_lf_new);
    }
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
      PT_BRIDGE_DEBUG(
          "Setting synapse permutation as saved in the cache to the output tensor id: ",
          ti.get_tensor_id(),
          " permutation: ",
          VecToString(ti.getHbInternalPermute()));
      hb_internal_tensor->SetMemoryPermutation(ti.getHbInternalPermute());
    }
    return pt_tensor;
  }};

  // Patch the input buffers
  // Running index on dtensorinfos
  size_t ridx = 0;
  std::unordered_map<size_t, IValPtrShared> inputIVpshMap;
  for (auto const& input : input_refs) {
    if (input.isTensor()) {
      auto& tensor = input.toTensor();
      auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
      PT_BRIDGE_DEBUG(
          "Cache input HbInternal address: ",
          impl,
          " permute: ",
          VecToString(impl->GetMemoryPermutation()));
      bool is_shape_tensor = impl && impl->isShapeTensor();
      if (false == is_shape_tensor) {
        dtensorinfos->at(ridx)->patch_exact(input.toTensor());
        IValPtrShared ivpsh = std::make_shared<IVal>(input);
        inputIVpshMap.emplace(ridx, ivpsh);
      } else {
        dtensorinfos->at(ridx)->set_host_ptr(impl->get_host_ptr());
      }
      ridx++;
    } else if (input.isTensorList()) {
      for (const at::Tensor& t : input.toTensorList()) {
        dtensorinfos->at(ridx)->patch_exact(t);
        IValPtrShared ivpsh = std::make_shared<IVal>(t);
        inputIVpshMap.emplace(ridx, ivpsh);
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
    for (; ridx < induplicates_index_end; ridx++) {
      size_t parent_idx = dtensorinfos->at(ridx)->get_parent_index();
      dtensorinfos->at(ridx)->patch(*(dtensorinfos->at(parent_idx)));
      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: Input duplicate : parent idx ",
          parent_idx,
          ", parent buffer ptr ",
          dtensorinfos->at(parent_idx)->get_buffer());
    }
  }

  // Patch the dma inputs if there are any
  if (num_dma_inputs) {
    // For DMA inputs patching works in reverse. The tensor is stored
    // within recipe and the corresponding index is stored in the tinfo.
    // The DMA input tensor needs to be populated.
    size_t dma_inputs_index_end =
        num_inputs + num_induplicates + num_dma_inputs;
    for (; ridx < dma_inputs_index_end; ridx++) {
      auto& ti = *(dtensorinfos->at(ridx));

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
      dma_inputs_ptr->push_back(dma_ivpsh);

      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: DMA input : buffer ptr ",
          dtensorinfos->at(ridx)->get_buffer());
    }
  }

  // Shape tensor patching is already done from name shape map
  ridx = ridx + num_shape_tensors;

  // TODO : Creation of output tensors and associated patching should
  // be part of a member function of RecipeValueSpec

  // Patch persistent intermediates
  // The persistent intermediates are retained in the rv
  size_t intermediates_start =
      num_inputs + num_induplicates + num_dma_inputs + num_shape_tensors;
  size_t intermediates_end = intermediates_start + num_intermediates;
  auto intermediate_idx = 0;
  std::unordered_map<size_t, IValPtrShared> intermediateIVpshMap;
  std::vector<at::Tensor> intermediate_tensors;
  for (; ridx < intermediates_end; ridx++) {
    PtTensorInfo& ti = *(dtensorinfos->at(ridx));
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
      auto pt_intermediate = create_empty_tensor(ti);
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
    intermediateIVpshMap.emplace(ridx, ivpsh);
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache hit :: Intermediate : buffer ptr ",
        rv_intermediate_tensor.data_ptr());
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

  aten_outputs = std::make_shared<std::vector<IValPtrShared>>(
      std::vector<IValPtrShared>(aten_output_num));

  // Patch outputs
  std::unordered_map<size_t, IValPtrShared> outputIVpshMap;
  size_t outputs_end = intermediates_end + num_outputs;
  for (; ridx < outputs_end; ridx++) {
    PtTensorInfo& ti = *(dtensorinfos->at(ridx));
    auto output_idx = ti.get_output_index();
    TORCH_CHECK(
        output_idx < aten_output_num,
        "output index ",
        output_idx,
        " is greater than #outputs ",
        aten_output_num);
    auto tshape{ti.get_shape()};
    auto pt_output = create_empty_tensor(ti);
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache hit :: Creating new output with shape : ",
        pt_output.sizes());
    IValPtrShared ivpsh = std::make_shared<IVal>(pt_output);
    aten_outputs->at(output_idx) = ivpsh;

    outputIVpshMap.emplace(ridx, ivpsh);

    // Patch the buffer for the output
    ti.patch(pt_output);
  }

  // Patch the duplicates of output that are going back to graph
  size_t outduplicates_end = outputs_end + num_outduplicates;
  if (num_outduplicates) {
    for (; ridx < outduplicates_end; ridx++) {
      size_t parent_idx = dtensorinfos->at(ridx)->get_parent_index();
      dtensorinfos->at(ridx)->patch(*(dtensorinfos->at(parent_idx)));
    }
  }

  TORCH_CHECK(
      ridx == outduplicates_end,
      "tensor info idx",
      ridx,
      " mismatch with outduplicates_end",
      outduplicates_end);

  // Patch the input duplicates if there are any
  size_t input_to_outduplicates_end =
      outduplicates_end + num_input_to_outduplicates;
  if (num_input_to_outduplicates) {
    for (; ridx < input_to_outduplicates_end; ridx++) {
      create_outdup(ridx, inputIVpshMap, "inputIVpshMap");
    }
  }

  TORCH_CHECK(
      ridx == input_to_outduplicates_end,
      "tensor info idx ",
      ridx,
      " mismatch with input_to_outduplicates_end ",
      input_to_outduplicates_end);

  // Patch the interim duplicates if there are any
  size_t interim_to_outduplicates_end =
      input_to_outduplicates_end + num_intermediate_to_outduplicates;
  if (num_intermediate_to_outduplicates) {
    for (; ridx < interim_to_outduplicates_end; ridx++) {
      create_outdup(ridx, intermediateIVpshMap, "intermediateIVpshMap");
    }
  }

  TORCH_CHECK(
      ridx == interim_to_outduplicates_end,
      "tensor info idx ",
      ridx,
      " mismatch with interim_to_outduplicates_end ",
      interim_to_outduplicates_end);

  // Patch the output duplicates if there are any
  size_t output_to_outduplicates_end =
      interim_to_outduplicates_end + num_output_to_outduplicates;
  if (num_output_to_outduplicates) {
    for (; ridx < output_to_outduplicates_end; ridx++) {
      create_outdup(ridx, outputIVpshMap, "outputIVpshMap");
    }
  }

  TORCH_CHECK(
      ridx == output_to_outduplicates_end,
      "tensor info idx ",
      ridx,
      " mismatch with output_to_outduplicates_end ",
      output_to_outduplicates_end);

  TORCH_CHECK(
      ridx == num_tinfos,
      "tensor info idx ",
      ridx,
      ", mismatch with num_tinfos",
      num_tinfos);
  PT_BRIDGE_END;
}

void RecipeValueSpec::populate_syn_tensor_ids() {
  if (!recipe) {
    PT_BRIDGE_DEBUG("Empty recipie. No need to retrive tensor ids.");
    return;
  }
  for (size_t i = 0; i < num_tinfos; ++i) {
    num_tensors++;
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_USE_SYN_TENSOR_IDS)) {
    if (nullptr == tensor_names) {
      tensor_ids = new uint64_t[num_tensors];
      tensor_names = new const char*[num_tensors];
    }

    size_t tensor_idx{0};
    for (size_t i = 0; i < num_tinfos; ++i) {
      PtTensorInfo& ti = *(dtensorinfos->at(i));
      tensor_names[tensor_idx++] = ti.get_syn_namec_str();
    }

    synStatus status = synTensorRetrieveIds(
        recipe->syn_recipe_handle_, tensor_names, tensor_ids, num_tensors);
    if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) {
      PT_BRIDGE_FATAL(
          "synTensorRetrieveIds launch failed ", std::to_string(status));
    }
  }
}

void RecipeValueSpec::patch_launch_info(
    std::vector<synLaunchTensorInfo>& syn_launch_info_vec,
    std::vector<size_t>& external_tensor_info_indexes) {
  TORCH_CHECK(
      (num_tensors != 0 && tensor_ids != nullptr && tensor_names != nullptr),
      "syn tensor ids are not populated");

  size_t tensor_idx{0};
  for (size_t i = 0; i < num_tinfos; ++i) {
    PtTensorInfo& ti = *(dtensorinfos->at(i));
    switch (ti.tensor_type()) {
      case SHAPE_TENSOR:
      case INPUT_DESCRIBING_SHAPE_TENSOR: {
        const auto& tsv = ti.syn_shape();
        syn_launch_info_vec.emplace_back(synLaunchTensorInfo{
            ti.get_syn_namec_str(),
            0,
            ti.tensor_type(),
            {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4]},
            tensor_ids[tensor_idx++]});
        break;
      }
      case HOST_TO_DEVICE_TENSOR: {
        const auto& tsv = ti.syn_shape();
        syn_launch_info_vec.emplace_back(synLaunchTensorInfo{
            ti.get_syn_namec_str(),
            ti.get_host_ptr(),
            ti.tensor_type(),
            {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4], tsv[5], tsv[6], tsv[7]},
            tensor_ids[tensor_idx++]});
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
            tensor_ids[tensor_idx++]});
        break;
      }
      case DEVICE_SHAPE_TENSOR: {
        const auto& tsv = ti.syn_shape();
        syn_launch_info_vec.emplace_back(synLaunchTensorInfo{
            ti.get_syn_namec_str(),
            ti.get_buffer_syn(),
            ti.tensor_type(),
            {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4]},
            tensor_ids[tensor_idx++]});
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

void RecipeValueSpec::PrintDebugInfo(
    at::ArrayRef<torch::jit::IValue>& input_refs,
    std::shared_ptr<std::vector<IValPtrShared>>& intermediate_tensors_ptr) {
  PT_BRIDGE_DEBUG(
      "Details of recipe",
      ", #inputs=",
      input_refs.size(),
      ", #intermediates=",
      intermediate_tensors_ptr->size(),
      ", #outputs=",
      aten_outputs->size());

  for (size_t idx{0}; idx < input_refs.size(); idx++) {
    ValPtr vp = (jit_graph_ ? jit_graph_->inputs().at(idx) : nullptr);
    PT_BRIDGE_DEBUG(
        "Input[",
        idx,
        "]",
        (vp ? (": %" + vp->debugName()) : std::string()),
        " -> ",
        habana_helpers::DebugString(input_refs[idx]));
  }
  if (intermediate_tensors_ptr) {
    size_t idx{0};
    for (auto& a : *intermediate_tensors_ptr) {
      PT_BRIDGE_DEBUG(
          "Intermediate[", idx, "] -> ", habana_helpers::DebugString(a));
      idx += 1;
    }
  }
  if (aten_outputs) {
    size_t idx{0};
    for (auto& a : *aten_outputs) {
      ValPtr vp = (jit_graph_ ? jit_graph_->outputs().at(idx) : nullptr);
      PT_BRIDGE_DEBUG(
          "Output[",
          idx,
          "]",
          (vp ? (": %" + vp->debugName()) : std::string()),
          " -> ",
          habana_helpers::DebugString(a));
      idx += 1;
    }
  }
  PT_BRIDGE_DEBUG(*this);
}

void RecipeValueSpec::launch(
    synapse_helpers::hpuStream_t hpu_stream,
    synEventHandle event_handle,
    synapse_helpers::hpuStream_t event_stream,
    bool event_flag,
    at::ArrayRef<torch::jit::IValue>& input_refs,
    std::shared_ptr<std::vector<IValPtrShared>>& intermediate_tensors_ptr,
    std::shared_ptr<std::vector<IValPtrShared>> dma_inputs_ptr) {
  PT_BRIDGE_BEGIN;
  SelfCheck();

  if (IS_BRIDGE_DEBUG_ENABLED) {
    PrintDebugInfo(input_refs, intermediate_tensors_ptr);
  }

  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto& stream_handle = device.get_compute_stream(hpu_stream);

  std::vector<at::Tensor> ptRefs;
  std::vector<at::Tensor> outPtRefs;
  std::vector<synapse_helpers::device_ptr> outDevPtr;

  std::vector<synLaunchTensorInfo> syn_launch_info;
  std::vector<size_t> external_tensor_info_indexes;
  if (recipe) {
    patch_launch_info(syn_launch_info, external_tensor_info_indexes);
  } else {
    PT_BRIDGE_DEBUG("Skipping patch_launch_info for empty recipe");
  }

  if (device.IsStreamASyncEnabled()) {
    // Get the reference to the tensor it is operating on to prevent
    // it from being deallocated while the operation is still in flight.
    std::vector<synapse_helpers::device_ptr> inDevPtr;
    inDevPtr.reserve(num_inputs);
    for (auto& input : input_refs) {
      if (input.isTensor()) {
        at::Tensor tensor = input.toTensor();
        ptRefs.push_back(std::move(tensor));
        inDevPtr.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
            input.toTensor().storage().data_ptr().get()));
      }
    }
    if (dma_inputs_ptr != nullptr && dma_inputs_ptr->size() > 0) {
      for (auto& dma_input : *dma_inputs_ptr) {
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
        num_inputs + num_outputs + num_input_to_outduplicates +
        num_intermediate_to_outduplicates);
    for (auto& output : *aten_outputs) {
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
          stream_handle, recipe->syn_recipe_handle_, &ti, []() {}));

      // Remove collective kenrel inputs from outDevPtr since they will be
      // signaled from the graph (if they are external)
      PT_BRIDGE_DEBUG(
          "Remove tensor ",
          ti.tensorName,
          " address ",
          std::hex,
          ti.pTensorAddress,
          std::dec,
          " from outDevPtr since it is an external tensor");
      outDevPtr.erase(
          std::remove(outDevPtr.begin(), outDevPtr.end(), ti.pTensorAddress),
          outDevPtr.end());
    }

    auto& recipe_counter = device.get_active_recipe_counter();
    recipe_counter.increase();
    std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
    {
      synapse_helpers::TimeScope ts(std::move(time_slot_));
      if (recipe) {
        auto&& error_optional{synapse_helpers::graph::launch(
            device,
            *recipe,
            workspace_size,
            syn_launch_info,
            address_lock,
            ext_events,
            stream_handle)};
        if (ABSL_PREDICT_FALSE(error_optional.has_value())) {
          recipe_counter.decrease_and_notify();
          auto& error = error_optional.value();
          PT_BRIDGE_FATAL(
              "syn launch encountered : ", error.error, " ", error.status);
          TORCH_CHECK(
              false,
              std::string("syn launch failed ") + std::string(error.error) +
                  std::string(" ") + std::to_string(error.status));
        }
      } else {
        PT_BRIDGE_DEBUG("Skipping recipe launch. empty recipe");
      }
    }

    // register events for external tensors on compute
    for (size_t i = 0; i < ext_events.size(); ++i) {
      device.register_producer_on_stream(stream_handle, ext_events.at(i));
    }

    // Use wrapper for resources that must survive async part of the compute.
    struct ResourceHolder {
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_id_;
      std::vector<at::Tensor> output_tensors_;
      std::vector<at::Tensor> input_tensors_;
      std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
      synapse_helpers::active_recipe_counter* recipe_counter_ptr;
    };
    auto resource_holder = std::shared_ptr<ResourceHolder>(
        new ResourceHolder(), [](ResourceHolder* resource_holder) {
          resource_holder->recipe_counter_ptr->decrease_and_notify();
          PT_LAZY_DEBUG("call decrease and notify of recipe_counter");
          delete resource_holder;
        });
    // recipe_id_ needs to be passed to done_cb to ensure its lifetime until
    // corresponding recipe is finished on stream
    const auto& recipe_ptr = recipe;
    resource_holder->recipe_id_ = recipe_ptr;
    resource_holder->input_tensors_ = ptRefs;
    resource_holder->output_tensors_ = outPtRefs;
    resource_holder->address_lock = std::move(address_lock);
    resource_holder->recipe_counter_ptr = &recipe_counter;
    // ResourceHolder could be used directly as callback, if we would only
    // implement operator(), but copying of ResourceHolder would result in
    // copying of all shared_ptr stored inside (including std::vector). To make
    // sharing more lightweight we hide ResourceHolder behind one shared_ptr.
    // This indirection allows us to maintain only one shared reference.
    auto cleanup_callback = [resource_holder]() mutable {
      resource_holder.reset();
    };

    // if event is a timer event, use this handle to record the event
    // dont use it for launch
    if (event_flag) {
      // regsiter an event on the compute
      device.register_producer_on_stream(
          std::move(outDevPtr), stream_handle, cleanup_callback, nullptr);
      // now record the timer_event
      if (event_handle) {
        auto& ev_stream_handle = device.get_compute_stream(event_stream);
        auto status = synEventRecord(event_handle, ev_stream_handle);
        if (synStatus::synSuccess != status) {
          PT_LAZY_FATAL("synEventRecord failed ", status);
        }
      }
    } else {
      // this case will happen only if there is direct launch or via set stream
      // or event record with record stream and current stream are same
      device.register_producer_on_stream(
          std::move(outDevPtr), stream_handle, cleanup_callback, event_handle);
    }
    // Launch collective ops
    HABANA_ASSERT(
        collective_kernels_info.empty() ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES))
    for (auto kernel_info : collective_kernels_info) {
      CollectiveOperator* collective =
          dynamic_cast<CollectiveOperator*>(kernel_info->kernel.get());
      HABANA_ASSERT(collective);
      PT_BRIDGE_DEBUG("Running collective op ", collective->GetGuid());
      collective->RunCollective(
          kernel_info->input_tensor_infos, true, cleanup_callback);
    }

  } else {
    std::vector<synapse_helpers::shared_event> ext_events;
    std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
    synapse_helpers::TimeScope ts(std::move(time_slot_));
    if (recipe) {
      auto&& error_optional{synapse_helpers::graph::launch(
          device,
          *recipe,
          workspace_size,
          syn_launch_info,
          address_lock,
          ext_events,
          stream_handle)};
      if (ABSL_PREDICT_FALSE(error_optional.has_value())) {
        auto& error = error_optional.value();
        PT_BRIDGE_FATAL(
            "syn launch encountered : ", error.error, " ", error.status);
        TORCH_CHECK(
            false,
            std::string("syn launch failed ") + std::string(error.error) +
                std::string(" ") + std::to_string(error.status));
      }
    }
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");

    // Launch collective ops
    HABANA_ASSERT(
        collective_kernels_info.empty() ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES))
    for (auto kernel_info : collective_kernels_info) {
      CollectiveOperator* collective =
          dynamic_cast<CollectiveOperator*>(kernel_info->kernel.get());
      HABANA_ASSERT(collective);
      PT_BRIDGE_DEBUG("Running collective op ", collective->GetGuid());
      collective->RunCollective(kernel_info->input_tensor_infos, false, [] {});
    }
  }

  num_launches++;
  increment_launch_count();
  PT_BRIDGE_END;
}

void RecipeCacheLRU::add(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<RecipeValueSpec>& val) {
  std::lock_guard<std::mutex> lg(mutex_);

  insert(key, val);

  if (disk_cache_) {
    disk_cache_->Add(*val, *key);
  }
}

void RecipeCacheLRU::insert(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<RecipeValueSpec>& val) {
  TORCH_CHECK(
      map_.size() == list_.size(),
      "lru cache corruption, map size ",
      map_.size(),
      " not equal to list_size ",
      list_.size());

  size_t rcnt{0};
  bool dropped{true};
  const bool is_ds_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!is_ds_enabled) {
    while (
        !map_.empty() && dropped &&
        (map_.size() >= max_size_ || habana::IsHostMemoryThresholdReached())) {
      dropped = drop_lru_impl(rcnt);
      if (!dropped) {
        PT_BRIDGE_DEBUG(
            "all recipes are in use, could not drop any, current recipe count ",
            rcnt);
      }
    }
  }

  val->increment_recipe_count();

  auto mit = map_.find(key);
  if (mit != map_.end()) {
    PT_BRIDGE_DEBUG(
        "problematic key ",
        key->hashCode(),
        " another recipe already exists in cache");
  } else {
    list_.push_front(std::pair<
                     std::shared_ptr<RecipeArgumentSpec>,
                     std::shared_ptr<RecipeValueSpec>>(key, val));
    map_.emplace(key, list_.begin());

    RecipeValueSpec::total_recipe_ntbytes += val->ntensorbytes;
  }
}

std::shared_ptr<RecipeValueSpec> RecipeCacheLRU::get(
    std::shared_ptr<RecipeArgumentSpec>& key) {
  std::lock_guard<std::mutex> lg(mutex_);
  if (exists(key)) {
    TORCH_CHECK(
        map_.size() == list_.size(),
        "lru cache corruption, map size ",
        map_.size(),
        " not equal to list_size ",
        list_.size());

    TORCH_CHECK(exists(key), "Recipe does not exist in map");

    auto mit = map_.find(key);
    list_.splice(list_.begin(), list_, mit->second);

    // wait till the execution complete
    bool use_flag{false};
    do {
      use_flag = list_.front().second->get_use_flag();
      if (use_flag) {
        PT_BRIDGE_DEBUG(
            "waiting for the completion of recipe, key ", key->hashCode());
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    } while (use_flag);

    // set the use flag true so that the recipe is not removed from cache
    // it is the responsibility of the caller of get function to
    // set the use flag to false after the execution is completed
    list_.front().second->set_use_flag(true);

    return list_.front().second;
  } else if (disk_cache_) {
    auto val = disk_cache_->Find(*key);
    if (val) {
      PT_BRIDGE_DEBUG(
          "recipe was not found in LRU cache, but was found on disk, key:",
          key->hashCode());
      insert(key, val);
      return val;
    }
  }
  return {nullptr};
}

bool RecipeCacheLRU::drop_lru(size_t& num_recipes) {
  std::lock_guard<std::mutex> lg(mutex_);
  bool dropped = drop_lru_impl(num_recipes, true);
  return dropped;
}

bool RecipeCacheLRU::drop_lru_impl(size_t& num_recipes, bool mem_exhausted) {
  bool dropped{false};
  int use_count = 0;
  // remove a recipe from the last that is not being used
  if (!map_.empty()) {
    auto lit = list_.end();
    lit--;

    while (lit->second->get_use_flag() == true && lit != list_.begin()) {
      PT_BRIDGE_DEBUG(
          "recipe is in use, key ",
          lit->first->hashCode(),
          ", size ",
          synapse_helpers::get_mem_str(lit->second->ntensorbytes));
      lit--;
    }

    // delete the recipe only if it is not in use
    // otherwise the caller need to wait
    if (lit->second->get_use_flag() == false) {
      if (mem_exhausted) {
        PT_BRIDGE_DEBUG(
            "memory exhausted : removing recipe, key ",
            lit->first->hashCode(),
            ", size ",
            synapse_helpers::get_mem_str(lit->second->ntensorbytes));
      } else {
        PT_BRIDGE_DEBUG(
            "lru max size ",
            max_size_,
            " reached : removing recipe, key ",
            lit->first->hashCode(),
            ", size ",
            synapse_helpers::get_mem_str(lit->second->ntensorbytes));
      }

      lit->second->decrement_recipe_count();
      RecipeValueSpec::total_recipe_ntbytes -= lit->second->ntensorbytes;

      // Drop the entry from map_ and list_
      dropped_recipe.first = lit->first;
      dropped_recipe.second = lit->second;
      map_.erase(lit->first);
      list_.erase(lit);
      dropped = true;

      PT_BRIDGE_DEBUG(
          "after dropping lru recipe, #recipes ",
          RecipeValueSpec::get_recipe_count(),
          ", total size of graph recipes ",
          synapse_helpers::get_mem_str(RecipeValueSpec::total_recipe_ntbytes));
    } else {
      use_count++;
      PT_BRIDGE_DEBUG(
          "all recipes are in use used_recipe_count=",
          use_count,
          " can not drop any recipe");
    }
  }

  num_recipes = map_.size() - use_count;
  return dropped;
}

RecipeCacheLRU::RecipeCacheLRU() {
  InitDiskCache();
}

void RecipeCacheLRU::InitDiskCache() {
  // Set disk_cache_ if PT_RECIPE_CACHE_PATH is defined
  const char* recipe_cache_path = GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH);
  if (!IS_ENV_FLAG_DEFINED_NEW(PT_RECIPE_CACHE_PATH))
    return;

  disk_cache_ = absl::make_unique<DiskCache>(recipe_cache_path);
}

void RecipeCacheLRU::ResetDiskCache() {
  // Reset disk_cache_ if PT_RECIPE_CACHE_PATH is defined
  const char* recipe_cache_path = GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH);
  if (!IS_ENV_FLAG_DEFINED_NEW(PT_RECIPE_CACHE_PATH))
    return;

  if (disk_cache_) {
    disk_cache_.reset();
  }
  disk_cache_ = absl::make_unique<DiskCache>(recipe_cache_path);
}

void RecipeCacheLRU::SetHostMemoryThreshold(uint32_t host_memory_threshold) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT)) {
    SET_ENV_FLAG_NEW(
        PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT, host_memory_threshold, 1);
  }
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
  map_.clear();
}

size_t RecipeCacheLRU::Size() const {
  size_t size = 0;
  for (auto const& [recipeArgumentSpec, recipeValueSpec] : list_) {
    size += recipeArgumentSpec->Size();
    size += recipeValueSpec->Size();
  }
  return size;
}

size_t RecipeCacheLRU::SynapseRecipeSize() const {
  size_t size = 0;
  for (auto const& p : list_) {
    size += p.second->recipe->get_recipe_host_mem_size();
  }
  return size;
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
      synapse_helpers::get_mem_str(RecipeCacheLRU::get_cache().Size()));
}

void RecipeCacheLRU::DumpSynapseRecipeMemoryStat() {
  PT_HOSTSTAT_DEBUG(
      "Size of Synapse Recipe: ",
      synapse_helpers::get_mem_str(
          RecipeCacheLRU::get_cache().SynapseRecipeSize()));
}

void RecipeCacheLRU::DumpDynamicShapeMemoryStat() {
  PT_HOSTSTAT_DEBUG(
      "DS MemoryStats:- Bucket::",
      synapse_helpers::get_mem_str(DynamicBucketInfoMap::get_instance().Size()),
      ", History::",
      synapse_helpers::get_mem_str(
          DynamicBucketInfoMap::get_instance().HistSize()),
      ", Recipe::",
      synapse_helpers::get_mem_str(RecipeCacheLRU::get_cache().Size()),
      ", SynapseRecipe::",
      synapse_helpers::get_mem_str(
          RecipeCacheLRU::get_cache().SynapseRecipeSize()));
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

DiskCache::DiskCache(std::string cache_path)
    : recipe_cache_(std::move(cache_path)),
      // TODO: add pytorch version, TICKET SW-62210
      cache_id_suffix_(absl::StrCat(
          "_",
          CacheVersion::libs_env_hash(),
          "_syn",
          synGetVersion())) {
  if (GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_IGNORE_VERSION)) {
    cache_id_suffix_ = "";
  }
}

void DiskCache::Add(
    const RecipeValueSpec& valSpec,
    const RecipeArgumentSpec& argSpec) // NOLINT
{
  std::stringstream ss;
  valSpec.Serialize(ss);
  auto hashCode = std::to_string(argSpec.hashCode());
  recipe_cache_.store(
      hashCode + cache_id_suffix_, valSpec.recipe, std::move(ss));
  if (valSpec.recipe && !valSpec.recipe->recipe_name_.empty()) {
    PT_BRIDGE_DEBUG(
        "Storing in disc cache: recipe:key: ",
        valSpec.recipe->recipe_name_,
        ":",
        hashCode);
  } else {
    PT_BRIDGE_DEBUG("Storing only metadata in disc cache, key: ", hashCode);
  }

  static const auto dump_debug_info =
      GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_DUMP_DEBUG);
  if (dump_debug_info) {
    static int debug_id = 0;
    std::string recipe_name = valSpec.recipe
        ? valSpec.recipe->recipe_name_
        : "recipe " + std::to_string(debug_id++);
    std::string hash_content_filepath = recipe_cache_.get_cache_path() + "/" +
        hashCode + cache_id_suffix_ + "_" + recipe_name + ".hash_content";
    std::ofstream hash_content_file(hash_content_filepath.c_str());
    if (!hash_content_file.is_open()) {
      LOG(FATAL) << "Failed to open hash content file for writing...";
    }
    hash_content_file << argSpec;
    hash_content_file.close();
  }
}

std::shared_ptr<RecipeValueSpec> DiskCache::Find(
    const RecipeArgumentSpec& spec) {
  std::stringstream ss;
  auto res = recipe_cache_.lookup(
      std::to_string(spec.hashCode()) + cache_id_suffix_, ss);
  if (res) {
    auto recipeValueSpec = std::make_shared<RecipeValueSpec>(ss);
    if (*res != nullptr) {
      if (!recipeValueSpec->recipe) {
        PT_BRIDGE_WARN(
            "Unexpected nullptr recipe came from cache entry for hash ",
            std::to_string(spec.hashCode()));
        return nullptr;
      }
      recipeValueSpec->recipe->syn_recipe_handle_ = *res;
      recipeValueSpec->recipe->in_execution_phase_ = true;
    }
    return recipeValueSpec;
  }
  return nullptr;
}

void ClearDynamicBucketRecipeInfo() {
  RecipeCacheLRU::get_cache().clear();
  DynamicBucketInfoMap::get_instance().clear();
}
} // namespace habana
