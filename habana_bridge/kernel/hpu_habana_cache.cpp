/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_device/HPUAllocator.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_info.h"
#include "habana_helpers/tensor_utils.h"
#include "synapse_helpers/env_flags.h"

namespace habana {

std::mutex RecipeCacheLRU::mutex_;
RecipeCacheLRU* RecipeCacheLRU::instance_ = nullptr;
size_t RecipeCacheLRU::max_size_ = PGM_LRU_MAX_LAZY_NRECIPES;
size_t RecipeValueSpec::recipe_count = 0;
size_t RecipeValueSpec::total_recipe_ntbytes = 0;

std::mutex DynamicBucketInfoMap::mutex_;
DynamicBucketInfoMap* DynamicBucketInfoMap::instance_ = nullptr;

std::ostream& operator<<(std::ostream& O, PGMCachingPolicy P) {
  switch (P) {
    case PGMCachingPolicy::simple:
      O << "simple";
      break;
    case PGMCachingPolicy::single:
      O << "single";
      break;
    case PGMCachingPolicy::lru:
      O << "lru";
      break;
    default:
      O << "unknown";
  }
  return O;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::string id)
    : cas(false, input_refs) {
  ComputeGraphHashCode(irgraph, id);
  hash_code = graph_hash_code;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    at::ArrayRef<torch::jit::IValue> input_refs,
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const uint64_t token,
    const std::string id)
    : cas(false, input_refs) {
  ComputeGraphHashCode(irgraph, id);
  hash_code = at::hash_combine(hash_code, graph_hash_code);
  hash_code = at::hash_combine(hash_code, token);

  ComputeOffsetHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, offset_hash_code);
  dynamic_hash_code = hash_code;
}

RecipeArgumentSpec::RecipeArgumentSpec(
    bool with_grad,
    at::ArrayRef<torch::jit::IValue> input_refs,
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id)
    : cas(with_grad, input_refs),
      opstrs(std::string()),
      hash_code(cas.hashCode()) {
  cargspec_hash_code = cas.hashCode();
  ComputeGraphHashCode(irgraph, id);
  hash_code = at::hash_combine(hash_code, graph_hash_code);
  hash_code = at::hash_combine(hash_code, irgraph->outputs().size());
  hash_code = habana_helpers::hash_combine_scalars(hash_code, input_refs);

  ComputeOffsetHashCode(input_refs);
  hash_code = at::hash_combine(hash_code, offset_hash_code);
}

void RecipeArgumentSpec::ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id) {
  std::hash<std::string> str_hash;
  opstrs.append((id.empty() ? std::string("UNNAMED") : id) + "::\n");
  std::unordered_map<torch::jit::Node*, size_t> node_idx_map;
  size_t idx{0};
  for (auto node : irgraph->nodes()) {
    if (node->kind() != torch::jit::prim::Constant) {
      std::string s(node->kind().toQualString());
      s.append("(");
      bool is_start{true};
      for (auto value_in : node->inputs()) {
        if (!is_start) {
          s.append(",");
        }
        is_start = false;
        s.append(value_in->node()->kind().toQualString());
      }
      s.append(")");
      // Adding delemeters for better readability
      opstrs.append(s + "\n");
    } else {
      std::ostringstream oss;
      oss << *node;
      opstrs.append(oss.str());
    }
    node_idx_map.emplace(node, idx);
    idx++;
  }
  graph_hash_code = str_hash(opstrs);

  size_t connection_hash{0};
  // Adding input hash
  for (size_t i = 0; i < irgraph->inputs().size(); ++i) {
    auto value_in = irgraph->inputs().at(i);
    size_t input_connection_hash = i;
    for (auto& use : value_in->uses()) {
      auto node = use.user;
      HABANA_ASSERT(node);
      input_connection_hash =
          at::hash_combine(input_connection_hash, node_idx_map[node]);
    }
    connection_hash = at::hash_combine(connection_hash, input_connection_hash);
  }
  // Adding output hash
  for (size_t i = 0; i < irgraph->outputs().size(); ++i) {
    auto value_out = irgraph->outputs().at(i);
    size_t output_connection_hash = i;
    auto node = value_out->node();
    HABANA_ASSERT(node);
    output_connection_hash =
        at::hash_combine(output_connection_hash, node_idx_map[node]);
    connection_hash = at::hash_combine(connection_hash, output_connection_hash);
  }
  graph_hash_code = at::hash_combine(graph_hash_code, connection_hash);
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
    delete[] tensor_ids;
    delete[] tensor_names;
  }
}

std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v) {
  O << "---- recipe details :: begin" << '\n';
  O << " <id : " << v.id << "> "
    << " <iteration : " << v.iter_idx << "> "
    << " <addr : " << v.recipe.get() << "> "
    << " <use_count : " << v.recipe.use_count() << "> " << '\n';
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

  if (v.aten_dma_inputs.size()) {
    O << "aten_dma_inputs #" << v.aten_dma_inputs.size() << " ::";
    O << '\n';
    size_t idx{0};
    for (auto& a : v.aten_dma_inputs) {
      O << idx++ << " : ";
      PrintATenTensor(a);
    }
  }

  if (v.aten_intermediates.size()) {
    O << "aten_intermediates #" << v.aten_intermediates.size() << " ::";
    O << '\n';
    size_t idx{0};
    for (auto& a : v.aten_intermediates) {
      O << idx++ << " : ";
      PrintATenTensor(a);
    }
  }

  if (v.aten_outputs) {
    O << "aten_outputs #" << v.aten_outputs->size() << " ::";
    O << '\n';
    size_t idx{0};
    for (auto& a : *v.aten_outputs) {
      O << idx++ << " : ";
      PrintATenTensor(a);
    }
  }

  if (v.dtensorinfos) {
    O << "dtensorinfos #" << v.dtensorinfos->size() << "::";
    O << '\n';
    size_t idx{0};
    for (auto& a : *v.dtensorinfos) {
      O << idx++ << " : ";
      O << a << '\n';
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
  unsigned buf_size = dtensorinfos->at(buf_idx).get_size();

  out << "iteration " << iteration_count << " : <"
      << ((buf_idx >= num_inputs) ? "output" : "input") << "> :: < "
      << dtensorinfos->at(buf_idx).get_ir_name() << " : "
      << "shape [" << dtensorinfos->at(buf_idx).get_shape() << "] : "
      << "numel " << dtensorinfos->at(buf_idx).get_numel() << " : "
      << "size (" << buf_size << " b) >";
  out << "<buffer" << '[' << buf_idx << ']' << "@"
      << dtensorinfos->at(buf_idx).get_buffer() << ">";

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

  unsigned buf_size = dtensorinfos->at(buf_idx).get_size();
  if (buf_size > htensor_wbuff_size) {
    buf_size = htensor_wbuff_size;
  }
  PT_BRIDGE_DEBUG("tensor dump will write ", htensor_wbuff_size, " bytes");

  auto& device = synapse_helpers::HPURegistrar::get_device();
  std::atomic<bool> copyDone{false};
  auto syn_error = device.copy_data_to_host(
      (uint64_t)dtensorinfos->at(buf_idx).get_buffer(),
      (void*)htensor_wbuff,
      dtensorinfos->at(buf_idx).get_buffer_start_syn(),
      buf_size,
      [&copyDone]() { copyDone = true; });
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }
}

std::string RecipeValueSpec::get_header_str() {
  if (header_str.empty()) {
    std::ostringstream o;
    o << "\n key " << key << "\n num_inputs " << num_inputs
      << "\n num_induplicates " << num_induplicates << "\n num_dma_inputs "
      << num_dma_inputs << "\n num_intermediates " << num_intermediates
      << "\n num_outputs " << num_outputs << "\n num_outduplicates "
      << num_outduplicates << "\n num_input_to_outduplicates "
      << num_input_to_outduplicates << "\n num_intermediate_to_outduplicates "
      << num_intermediate_to_outduplicates << "\n size "
      << synapse_helpers::get_mem_str(ntensorbytes);

    header_str = o.str();
  }

  return header_str;
}

int RecipeValueSpec::update_hit_count() {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  device.get_recipe_handle_cache().increaseHitCount(key);
  auto rv_hit_count = device.get_recipe_handle_cache().getHitCount(key);

  auto max_hit_count = GET_ENV_FLAG(PT_HABANA_MAX_RECIPE_HIT_COUNT);
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

void RecipeValueSpec::update_patching_table(
    at::ArrayRef<torch::jit::IValue>& input_refs,
    std::shared_ptr<std::vector<IValPtrShared>>& dma_inputs,
    const habana::NameShapeMap& m_actual_shapes,
    bool enable_tensor_release) {
  // Patch the input buffers
  // Running index on dtensorinfos
  size_t ridx = 0;

  if (dynamic_graph) {
    for (size_t i = 0; i < dtensorinfos->size(); ++i) {
      auto& ti = dtensorinfos->at(i);
      if (ti.is_tensor()) {
        auto syn_name = ti.get_syn_name();
        HABANA_ASSERT(m_actual_shapes.count(syn_name));
        auto dims = m_actual_shapes.at(syn_name).get_dims();
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

  std::unordered_map<size_t, IValPtrShared> inputIVpshMap;
  for (auto const& input : input_refs) {
    if (input.isTensor()) {
      dtensorinfos->at(ridx).patch_exact(input.toTensor());
      IValPtrShared ivpsh = std::make_shared<IVal>(input);
      inputIVpshMap.emplace(ridx, ivpsh);
      ridx++;
    } else if (input.isTensorList()) {
      for (const at::Tensor& t : input.toTensorList()) {
        dtensorinfos->at(ridx).patch_exact(t);
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
      size_t parent_idx = dtensorinfos->at(ridx).get_parent_index();
      dtensorinfos->at(ridx).patch(dtensorinfos->at(parent_idx));
      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: Input duplicate : parent idx ",
          parent_idx,
          ", parent buffer ptr ",
          dtensorinfos->at(parent_idx).get_buffer());
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
      auto& ti = dtensorinfos->at(ridx);

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
      dma_inputs->push_back(dma_ivpsh);

      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit :: DMA input : buffer ptr ",
          dtensorinfos->at(ridx).get_buffer());
    }
  }

  // Patch the shape tensor inputs if there are any
  if (num_shape_tensors) {
    size_t shape_start = num_inputs + num_induplicates + num_dma_inputs;
    size_t shape_end = shape_start + num_shape_tensors;
    for (; ridx < shape_end; ridx++) {
      auto& ti = dtensorinfos->at(ridx);
      auto tshape{ti.get_shape()};
      at::TensorOptions topts(ti.get_topts());
      // TODO: Create storageless tensor
      auto pt_shape = at::empty(tshape, topts, ti.get_mf());
      ti.patch_exact(pt_shape);
    }
  }

  if (enable_tensor_release) {
    // TODO : Creation of output tensors and associated patching should
    // be part of a member function of RecipeValueSpec

    // Patch persistent intermediates
    // The persistent intermediates are retained in the rv
    size_t intermediates_start =
        num_inputs + num_induplicates + num_dma_inputs + num_shape_tensors;
    size_t intermediates_end = intermediates_start + num_intermediates;
    auto intermediate_idx = 0;
    std::unordered_map<size_t, IValPtrShared> intermediateIVpshMap;
    for (; ridx < intermediates_end; ridx++) {
      PtTensorInfo& ti = dtensorinfos->at(ridx);
      TORCH_CHECK(
          ti.is_tensor(),
          "non tensor tinfo found for persistent intermediates");
      auto tshape{ti.get_shape()};

      if (GET_ENV_FLAG(PT_HPU_ENABLE_INTERMEDIATE_TENSOR_RELEASE)) {
        if (ti.is_duplicate()) {
          auto ti_parent_index = ti.get_parent_index();
          auto pt_parent_index = ti_parent_index - intermediates_start;
          TORCH_CHECK(
              pt_parent_index < aten_intermediates.size(),
              "out of range duplicate intermediate tensor index ",
              pt_parent_index,
              " #aten_intermediates ",
              aten_intermediates.size());

          auto pt_parent = aten_intermediates[pt_parent_index];

          auto pt_sizes{ti.get_shape()};
          auto pt_strides{ti.get_strides()};
          long pt_offset = (long)ti.get_offset() / pt_parent.itemsize();
          auto pt_opt_offset = c10::make_optional(pt_offset);

          at::Tensor pt_intermediate =
              at::as_strided(pt_parent, pt_sizes, pt_strides, pt_opt_offset);

          PT_BRIDGE_DEBUG(
              "HabanaOp recipe cache hit :: Intermediate : Duplicate with shape : ",
              tshape);

          aten_intermediates.push_back(pt_intermediate);
        } else {
          auto pt_intermediate = at::empty(tshape, ti.get_topts(), ti.get_mf());
          PT_BRIDGE_DEBUG(
              "HabanaOp recipe cache hit :: Intermediate : Creating new with shape : ",
              tshape);

          aten_intermediates.push_back(pt_intermediate);
        }
      }
      auto& rv_intermediate_tensor = aten_intermediates.at(intermediate_idx++);

      // Theoretically the data and storage pts of an interim tinfo
      // should not change over iterations. That possibility will only
      // arise if we support freeing of aten_intermediates after the
      // recipe execution.
      ti.patch(rv_intermediate_tensor);

      IValPtrShared ivpsh = std::make_shared<IVal>(rv_intermediate_tensor);
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
      PtTensorInfo& ti = dtensorinfos->at(ridx);
      auto output_idx = ti.get_output_index();
      TORCH_CHECK(
          output_idx < aten_output_num,
          "output index ",
          output_idx,
          " is greater than #outputs ",
          aten_output_num);
      if (ti.is_tensor()) {
        auto tshape{ti.get_shape()};
        auto pt_output = at::empty(tshape, ti.get_topts(), ti.get_mf());
        PT_BRIDGE_DEBUG(
            "HabanaOp recipe cache hit :: Creating new output with shape : ",
            pt_output.sizes());
        IValPtrShared ivpsh = std::make_shared<IVal>(pt_output);
        aten_outputs->at(output_idx) = ivpsh;

        outputIVpshMap.emplace(ridx, ivpsh);

        // Patch the buffer for the output
        ti.patch(pt_output);
      } else {
        IValPtrShared ivpsh = std::make_shared<IVal>(ti.get_ivalue());
        aten_outputs->at(output_idx) = ivpsh;
      }
    }

    // Patch the duplicates if there are any
    // Dead code : currently num_outduplicates should always be 0
    // TODO : Clean up this
    size_t outduplicates_end = outputs_end + num_outduplicates;
    if (num_outduplicates) {
      for (; ridx < outduplicates_end; ridx++) {
        size_t parent_idx = dtensorinfos->at(ridx).get_parent_index();
        dtensorinfos->at(ridx).patch(dtensorinfos->at(parent_idx));
      }
    }

    TORCH_CHECK(
        num_outduplicates == 0,
        "Encountering non zero value ",
        num_outduplicates,
        " for num_outduplicates");

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
  }
}

void RecipeValueSpec::populate_syn_tensor_ids() {
  for (size_t i = 0; i < num_tinfos; ++i) {
    PtTensorInfo& ti = dtensorinfos->at(i);
    if (ti.is_tensor()) {
      num_tensors++;
    }
  }

  if (GET_ENV_FLAG(PT_HPU_USE_SYN_TENSOR_IDS)) {
    if (nullptr == tensor_names) {
      tensor_ids = new uint64_t[num_tensors];
      tensor_names = new const char*[num_tensors];
    }

    size_t tensor_idx{0};
    for (size_t i = 0; i < num_tinfos; ++i) {
      PtTensorInfo& ti = dtensorinfos->at(i);
      if (ti.is_tensor()) {
        tensor_names[tensor_idx++] = ti.get_syn_namec_str();
      }
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
    std::vector<synLaunchTensorInfoExt>& syn_launch_info_vec) {
  TORCH_CHECK(
      (num_tensors != 0 && tensor_ids != nullptr && tensor_names != nullptr),
      "syn tensor ids are not populated");

  size_t tensor_idx{0};
  for (size_t i = 0; i < num_tinfos; ++i) {
    PtTensorInfo& ti = dtensorinfos->at(i);
    if (ti.is_tensor()) {
      switch (ti.tensor_type()) {
        case DATA_TENSOR: {
          syn_launch_info_vec.emplace_back(synLaunchTensorInfoExt{
              ti.get_syn_namec_str(),
              ti.get_buffer_syn(),
              ti.tensor_type(),
              {0},
              tensor_ids[tensor_idx++]});
          break;
        }
        case SHAPE_TENSOR:
        case INPUT_DESCRIBING_SHAPE_TENSOR: {
          const auto& tsv = ti.syn_shape();
          syn_launch_info_vec.emplace_back(synLaunchTensorInfoExt{
              ti.get_syn_namec_str(),
              0,
              ti.tensor_type(),
              {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4]},
              tensor_ids[tensor_idx++]});
          break;
        }
        case DATA_TENSOR_DYNAMIC: {
          const auto& tsv = ti.syn_shape();
          syn_launch_info_vec.emplace_back(synLaunchTensorInfoExt{
              ti.get_syn_namec_str(),
              ti.get_buffer_syn(),
              ti.tensor_type(),
              {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4], tsv[5], tsv[6], tsv[7]},
              tensor_ids[tensor_idx++]});
          break;
        }
        case DEVICE_SHAPE_TENSOR: {
          const auto& tsv = ti.syn_shape();
          syn_launch_info_vec.emplace_back(synLaunchTensorInfoExt{
              ti.get_syn_namec_str(),
              ti.get_buffer_syn(),
              ti.tensor_type(),
              {tsv[0], tsv[1], tsv[2], tsv[3], tsv[4]},
              tensor_ids[tensor_idx++]});
          break;
        }
        case TENSOR_TYPE_MAX:
          TORCH_CHECK(
              false,
              "Patching of ",
              ti.tensor_type(),
              " is not supported yet.");
          break;
        default:
          TORCH_CHECK(false, "Unreachable condition.");
      }
    }
  }
}

void RecipeValueSpec::launch(
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::shared_ptr<std::vector<IValPtrShared>> dma_inputs) {
  SelfCheck();

  PT_BRIDGE_DEBUG("RecipeValueSpec::launch\n", *this);

  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto& stream_handle = device.get_compute_stream();
  std::vector<at::Tensor> ptRefs;
  std::vector<at::Tensor> outPtRefs;
  std::vector<synapse_helpers::device_ptr> outDevPtr;

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
    if (dma_inputs != nullptr && dma_inputs->size() > 0) {
      for (auto& dma_input : *dma_inputs) {
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

    if (GET_ENV_FLAG(PT_HPU_ENABLE_INTERMEDIATE_TENSOR_RELEASE)) {
      // Hold on to the pytorch tensors for the intermediates untill the recipe
      // execution completes
      for (auto& tensor : aten_intermediates) {
        ptRefs.push_back(std::move(tensor));
      }
      aten_intermediates.clear();
    }

    outDevPtr.reserve(
        num_outputs + num_input_to_outduplicates +
        num_intermediate_to_outduplicates);
    for (auto& output : *aten_outputs) {
      if (output && output->isTensor()) {
        at::Tensor tensor = output->toTensor();
        outDevPtr.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
            tensor.storage().data_ptr().get()));
        outPtRefs.push_back(std::move(tensor));
      }
    }
  }

  std::vector<synLaunchTensorInfoExt> syn_launch_info;
  patch_launch_info(syn_launch_info);
  if (device.IsStreamASyncEnabled()) {
    auto& recipe_counter = device.get_active_recipe_counter();
    recipe_counter.increase();
    auto&& error_optional{synapse_helpers::graph::launch(
        device, *recipe, workspace_size, syn_launch_info)};
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
    const auto& recipe_ptr = recipe;
    // regsiter an event on the compute
    device.register_producer_on_stream(
        std::move(outDevPtr),
        stream_handle,
        [ptRefs, outPtRefs, recipe_ptr, &recipe_counter]() {
          recipe_counter.decrease_and_notify();
          return;
        });
  } else {
    auto&& error_optional{synapse_helpers::graph::launch(
        device, *recipe, workspace_size, syn_launch_info)};
    if (ABSL_PREDICT_FALSE(error_optional.has_value())) {
      auto& error = error_optional.value();
      PT_BRIDGE_FATAL(
          "syn launch encountered : ", error.error, " ", error.status);
      TORCH_CHECK(
          false,
          std::string("syn launch failed ") + std::string(error.error) +
              std::string(" ") + std::to_string(error.status));
    }
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

void RecipeCacheSimple::add(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<RecipeValueSpec>& val) {
  RecipeValueSpec::recipe_count++;
  map_.emplace(key, val);
  RecipeValueSpec::total_recipe_ntbytes += val->ntensorbytes;
}

std::ostream& operator<<(std::ostream& O, const RecipeCacheSimple& v) {
  O << "number of recipes : " << v.map_.size() << '\n';
  for (auto& i : v.map_) {
    O << "-------------------" << '\n';
    O << "key :: " << *i.first;
    O << "-------------------" << '\n';
    O << "val :: " << *i.second;
    O << "-------------------" << '\n';
  }
  return O;
}

void RecipeCacheSingle::add(
    std::shared_ptr<RecipeArgumentSpec>& rargpsh,
    std::shared_ptr<RecipeValueSpec>& rvalpsh) {
  if (!is_valid) {
    RecipeValueSpec::recipe_count++;
    is_valid = true;
  } else {
    TORCH_CHECK(
        RecipeValueSpec::total_recipe_ntbytes >= last_rvalpsh->ntensorbytes,
        "error in total tensor byte accounting, total_recipe_ntbytes ",
        RecipeValueSpec::total_recipe_ntbytes,
        " should be greater than last_recipe.ntensorbytes ",
        last_rvalpsh->ntensorbytes);

    RecipeValueSpec::total_recipe_ntbytes -= last_rvalpsh->ntensorbytes;
  }
  last_rargpsh = rargpsh;
  last_rvalpsh = rvalpsh;
  RecipeValueSpec::total_recipe_ntbytes += last_rvalpsh->ntensorbytes;
}

std::ostream& operator<<(std::ostream& O, const RecipeCacheSingle& v) {
  O << "-------------------" << '\n';
  O << "key :: " << *v.last_rargpsh;
  O << "-------------------" << '\n';
  O << "val :: " << *v.last_rvalpsh;
  O << "-------------------" << '\n';
  return O;
}

void RecipeCacheLRU::add(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<RecipeValueSpec>& val) {
  std::lock_guard<std::mutex> lg(mutex_);

  TORCH_CHECK(
      map_.size() == list_.size(),
      "lru cache corruption, map size ",
      map_.size(),
      " not equal to list_size ",
      list_.size());

  size_t rcnt{0};
  bool dropped{true};
  while (!map_.empty() && dropped && map_.size() >= max_size_) {
    dropped = drop_lru_impl(rcnt);
    if (!dropped) {
      PT_BRIDGE_DEBUG(
          "all recipes are in use, could not drop any, current recipe count ",
          rcnt);
    }
  }

  RecipeValueSpec::recipe_count++;

  auto mit = map_.find(key);
  TORCH_CHECK(
      mit == map_.end(),
      "problematic key ",
      key,
      " another recipe already exists in cache");

  list_.push_front(std::pair<
                   std::shared_ptr<RecipeArgumentSpec>,
                   std::shared_ptr<RecipeValueSpec>>(key, val));
  map_.emplace(key, list_.begin());

  RecipeValueSpec::total_recipe_ntbytes += val->ntensorbytes;
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
  }

  return {nullptr};
}

bool RecipeCacheLRU::drop_lru(size_t& recipe_count) {
  std::lock_guard<std::mutex> lg(mutex_);
  bool dropped = drop_lru_impl(recipe_count, true);
  return dropped;
}

bool RecipeCacheLRU::drop_lru_impl(size_t& recipe_count, bool mem_exhausted) {
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

      RecipeValueSpec::recipe_count--;
      RecipeValueSpec::total_recipe_ntbytes -= lit->second->ntensorbytes;

      // Drop the entry from map_ and list_
      map_.erase(lit->first);
      list_.erase(lit);
      dropped = true;

      PT_BRIDGE_DEBUG(
          "after dropping lru recipe, #recipes ",
          RecipeValueSpec::recipe_count,
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

  recipe_count = map_.size() - use_count;
  return dropped;
}

std::shared_ptr<habana_helpers::DynamicBucketInfo> DynamicBucketInfoMap::get(
    std::shared_ptr<RecipeArgumentSpec>& key) {
  std::lock_guard<std::mutex> lg(mutex_);
  if (exists(key)) {
    return map_[key];
  }
  return {nullptr};
}

void DynamicBucketInfoMap::add(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<habana_helpers::DynamicBucketInfo>& val) {
  std::lock_guard<std::mutex> lg(mutex_);
  map_.emplace(key, val);
}

} // namespace habana
