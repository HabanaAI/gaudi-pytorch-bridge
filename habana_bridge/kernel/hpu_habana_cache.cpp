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

size_t RecipeValueSpec::recipe_count = 0;
size_t RecipeValueSpec::total_recipe_ntbytes = 0;

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
    bool with_grad,
    at::ArrayRef<torch::jit::IValue> input_refs,
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id)
    : cas(with_grad, input_refs),
      hash_code(cas.hashCode()),
      opstrs(std::string()) {
  std::hash<std::string> str_hash;
  opstrs.append(id + "::\n");
  for (auto* node : irgraph->nodes()) {
    std::string s(node->kind().toQualString());
    // Adding delemeters for better readability
    opstrs.append("<" + s + ">");
    if (node->kind() == torch::jit::prim::Constant) {
      std::ostringstream oss;
      oss << *node;
      opstrs.append(":" + oss.str());
    }
  }
  hash_code = at::hash_combine(hash_code, str_hash(opstrs));
  hash_code = at::hash_combine(hash_code, irgraph->outputs().size());
  hash_code = habana_helpers::hash_combine_scalars(hash_code, input_refs);
}

std::ostream& operator<<(std::ostream& O, const RecipeArgumentSpec& v) {
  O << v.hash_code << '\n';
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
}

std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v) {
  O << "---- recipe details :: begin" << '\n';
  O << " <id : " << v.id << "> "
    << " <iteration : " << v.iter_idx << "> "
    << " <addr : " << v.recipe.get() << "> "
    << " <use_count : " << v.recipe.use_count() << "> " << '\n';
  O << " ntensorbytes : " << synapse_helpers::get_mem_str(v.ntensorbytes)
    << '\n';
  O << " workspace    : "
    << synapse_helpers::get_mem_str(v.launch_info->workspace_buffer_size_)
    << '\n';

  O << " num_inputs       : " << v.num_inputs << '\n'
    << " num_induplicates : " << v.num_induplicates << '\n'
    << " num_dma_inputs   : " << v.num_dma_inputs << '\n'
    << " num_interims     : " << v.num_interims << '\n'
    << " num_outputs      : " << v.num_outputs << '\n';

  if (v.aten_intermediates.size()) {
    O << "aten_intermediates #" << v.aten_intermediates.size() << " ::";
    O << '\n';
    for (auto& a : v.aten_intermediates) {
      PrintATenTensor(a);
    }
  }

  if (v.aten_outputs) {
    O << "aten_outputs #" << v.aten_outputs->size() << " ::";
    O << '\n';
    for (auto& a : *v.aten_outputs) {
      PrintATenTensor(a);
    }
  }

  if (v.dtensorinfos) {
    O << "dtensorinfos #" << v.dtensorinfos->size() << "::";
    O << '\n';
    for (auto& a : *v.dtensorinfos) {
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
      << "shape " << dtensorinfos->at(buf_idx).get_shape_str() << " : "
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
  TORCH_CHECK(num_tensors > buf_idx, "buf_idx is out of range");

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
      dtensorinfos->at(buf_idx).get_storage_data_ptr(),
      buf_size,
      [&copyDone]() { copyDone = true; });
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }
}

void RecipeValueSpec::create_launch_info() {
  if (!launch_info) {
    launch_info.emplace(recipe->device_);
    synapse_helpers::graph::create_launch_info(*launch_info, *recipe);
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
    outDevPtr.reserve(num_outputs + num_in_to_outduplicates);
    for (auto& output : *aten_outputs) {
      if (output && output->isTensor()) {
        outDevPtr.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
            output->toTensor().storage().data_ptr().get()));
      }
    }
  }

  std::vector<synLaunchTensorInfo> syn_launch_info;

  // Populate the <name,buffer> pairs from PtTensorInfo for synLaunch
  for (size_t i = 0; i < num_tensors; ++i) {
    if (dtensorinfos->at(i).is_tensor()) {
      syn_launch_info.emplace_back(synLaunchTensorInfo{
          dtensorinfos->at(i).get_syn_namec_str(),
          reinterpret_cast<uint64_t>(dtensorinfos->at(i).get_buffer()),
          DATA_TENSOR,
          {0}});
    }
  }

  if (device.IsStreamASyncEnabled()) {
    auto& recipe_counter = device.get_active_recipe_counter();
    recipe_counter.increase();
    auto&& error_optional{
        synapse_helpers::graph::launch(*launch_info, *recipe, syn_launch_info)};
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
        [ptRefs, recipe_ptr, &recipe_counter]() {
          recipe_counter.decrease_and_notify();
          return;
        });
  } else {
    auto&& error_optional{
        synapse_helpers::graph::launch(*launch_info, *recipe, syn_launch_info)};
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

  PT_BRIDGE_DEBUG(
      "  adding new recipe, key ",
      key->hashCode(),
      ", ntensorbytes ",
      val->ntensorbytes);

  PT_BRIDGE_DEBUG(
      "  after adding new recipe, nrecipes ",
      RecipeValueSpec::recipe_count,
      " total_recipe_ntbytes ",
      RecipeValueSpec::total_recipe_ntbytes);
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
          ", ntensorbytes ",
          lit->second->ntensorbytes);
      lit--;
    }

    // delete the recipe only if it is not in use
    // otherwise the caller need to wait
    if (lit->second->get_use_flag() == false) {
      if (mem_exhausted) {
        PT_BRIDGE_DEBUG(
            "memory exhausted : removing recipe, key ",
            lit->first->hashCode(),
            ", ntensorbytes ",
            lit->second->ntensorbytes);
      } else {
        PT_BRIDGE_DEBUG(
            "lru max size ",
            max_size_,
            " reached : removing recipe, key ",
            lit->first->hashCode(),
            ", ntensorbytes ",
            lit->second->ntensorbytes);
      }

      RecipeValueSpec::recipe_count--;
      RecipeValueSpec::total_recipe_ntbytes -= lit->second->ntensorbytes;

      // Drop the entry from list_ and map_
      map_.erase(lit->first);
      list_.pop_back();
      dropped = true;

      PT_BRIDGE_DEBUG(
          "after dropping lru recipe, nrecipes ",
          RecipeValueSpec::recipe_count,
          " total_recipe_ntbytes ",
          RecipeValueSpec::total_recipe_ntbytes);
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
