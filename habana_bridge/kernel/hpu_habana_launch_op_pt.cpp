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
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUAllocator.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "habana_bridge/kernel/hpu_habana_meta_op_list.h"
#include "habana_device/tensor_builder.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch::jit;

// static initializations
bool   TensorInfo::watch_tensor_flag = false;
size_t RecipeValueSpec::count = 0;

std::mutex RecipeCacheLRU::mutex_;
RecipeCacheLRU* RecipeCacheLRU::instance_ = nullptr;
size_t RecipeCacheLRU::max_size_ = PGM_LRU_MAX_NRECIPES;

size_t HabanaLaunchOpPT::instance_count_ = 0;
size_t HabanaLaunchOpPT::recipe_count = 0;
size_t HabanaLaunchOpPT::total_recipe_ntbytes = 0;
std::unordered_set<std::string> HabanaLaunchOpPT::watchlist_ = {};
//--------------------------------------

TensorInfo::TensorInfo(
    const IValPtrShared& ivpsh,
    const std::string& sn,
    const ValPtr& vp) {
  TORCH_CHECK(ivpsh->isTensor(), "aten tensor is expected");
  ir_name = "%" +  vp->debugName();

  syn_name = sn;

  auto pt_tensor = ivpsh->toTensor();
  {
    std::ostringstream oss;
    oss << pt_tensor.sizes();
    shape_str = oss.str();
  }

  buffer = pt_tensor.data_ptr();
  numel = pt_tensor.numel();
  size = pt_tensor.nbytes();
  watch = watch_tensor_flag;
}

TensorInfo::TensorInfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn) {
  ir_name = irn;
  syn_name = sn;

  std::ostringstream oss;
  oss << pt_tensor.sizes();
  shape_str = oss.str();

  buffer = pt_tensor.data_ptr();
  numel = pt_tensor.numel();
  size = pt_tensor.nbytes();
  watch = watch_tensor_flag;
}

std::ostream& operator<<(std::ostream& O, const TensorInfo& t) {
  O << '<' << t.ir_name << ':' << t.shape_str << ':' << t.numel << ':' << '('
    << t.size << " b)"
    << " :: " << t.syn_name << ':' << t.buffer << '>';

  if (t.is_duplicate) {
    O << " duplicate";
  }

  return O;
}

void PrintATenTensor(const at::Tensor& a) {
  std::ostream& O = std::cout;
  O << " Tensor -> ";
  if (a.has_storage()) {
    O << " @ " << a.data_ptr() << " : "
      << " dim " << a.dim() << " : " << a.sizes();
  } else {
    O << " does not have storage";
  }
  O << ',' << " use_count " << a.use_count()
    << '\n';
}

void PrintATenTensor(const IValPtrShared &a) {
  if (a->isTensor()) {
    PrintATenTensor(a->toTensor());
  }
}

std::ostream & operator<<(std::ostream & O, PGMCachingPolicy P) {
  switch (P) {
    case PGMCachingPolicy::simple :
      O << "simple";
      break;
    case PGMCachingPolicy::single :
      O << "single";
      break;
    case PGMCachingPolicy::lru :
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
    const std::string &id)
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
  hash_code = torch::hash_combine(hash_code, str_hash(opstrs));
}

void adjustSizesforPT(at::Tensor* tensor, bool is_output) {
  auto sizes = tensor->sizes().vec();
  auto strides = tensor->strides().vec();
  at::IntArrayRef out_pos = {0, 3, 1, 2};
  at::IntArrayRef in_pos = {0, 2, 3, 1};

  at::IntArrayRef new_pos_arr = is_output ? out_pos : in_pos;
  auto new_pos = new_pos_arr.vec();
  std::vector<long int> swapped_sizes = {sizes[new_pos[0]],
                                         sizes[new_pos[1]],
                                         sizes[new_pos[2]],
                                         sizes[new_pos[3]]};
  std::vector<long int> swapped_strides = {strides[new_pos[0]],
                                           strides[new_pos[1]],
                                           strides[new_pos[2]],
                                           strides[new_pos[3]]};

  //*tensor_new = at::alias(*tensor);
  tensor->unsafeGetTensorImpl()->set_sizes_and_strides(
      swapped_sizes, swapped_strides);
}

std::ostream& operator<<(std::ostream& O, const RecipeArgumentSpec& v) {
  O << v.hash_code << '\n';
  return O;
}

RecipeValueSpec::~RecipeValueSpec() {
  PT_BRIDGE_DEBUG("Destroying recipe with key : ", key);

  if (htensor_wbuff ) {
    synStatus status;
    auto& device = synapse_helpers::HPURegistrar::get_device();
    auto  device_id = device.id();
    status = synHostFree(device_id, (void*)(htensor_wbuff), 0);
    if (status != synSuccess)
      PT_BRIDGE_DEBUG("host-free failed");
  }
}

std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v) {
  O << "---- recipe details ::"
    << " <id : " << v.id << "> "
    << " <iteration : " << v.iter_idx << "> "
    << " <addr : " << v.recipe.get() << "> "
    << " <use_count : " << v.recipe.use_count() << "> " << '\n';

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
  unsigned buf_size = dtensorinfos->at(buf_idx).size;

  out << "iteration " << iteration_count << " : <"
      << ((buf_idx >= num_inputs) ? "output" : "input") << "> :: < "
      << dtensorinfos->at(buf_idx).ir_name << " : "
      << "shape " << dtensorinfos->at(buf_idx).shape_str << " : "
      << "numel " << dtensorinfos->at(buf_idx).numel << " : "
      << "size (" << buf_size << " b) >";
  out << "<buffer" << '[' << buf_idx << ']' << "@"
      << dtensorinfos->at(buf_idx).buffer << ">";

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


  unsigned buf_size = dtensorinfos->at(buf_idx).size;
  if (buf_size > htensor_wbuff_size) {
    buf_size = htensor_wbuff_size;
  }
  PT_BRIDGE_DEBUG("tensor dump will write ", htensor_wbuff_size, " bytes");

  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.IsStreamASyncEnabled()) {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_host(
        (uint64_t)dtensorinfos->at(buf_idx).buffer,
        (void*)htensor_wbuff,
        buf_size,
        [&copyDone]() { copyDone = true; });
    TORCH_CHECK(syn_error.status == 0, syn_error.error);

    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  } else {
    synStatus status;
    synDeviceId device_id = device.id();
    synEventHandle upldEvntDone;
    synStreamHandle upStrmHdl = device.get_device_to_host_stream();
    status = synEventCreate(&upldEvntDone, device_id, 0);
    TORCH_CHECK(status == synSuccess, "create upldEvntDone failed");

    status = synMemCopyAsync(
        upStrmHdl,
        (uint64_t)dtensorinfos->at(buf_idx).buffer,
        buf_size,
        htensor_wbuff,
        DRAM_TO_HOST);
    TORCH_CHECK(status == synSuccess, "synMemCopyAsync failed");

    status = synEventRecord(upldEvntDone, upStrmHdl);
    TORCH_CHECK(status == synSuccess, "register to signal on d2h copy done");

    status = synStreamSynchronize(upStrmHdl);
    TORCH_CHECK(status == synSuccess, "wait on completion of d2h copy");

    status = synEventDestroy(upldEvntDone);
    TORCH_CHECK(status == synSuccess, "destroy upldEvntDone failed");
  }
}

void RecipeCacheSimple::add(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<RecipeValueSpec>& val) {
  HabanaLaunchOpPT::recipe_count++;
  map_.emplace(key, val);
  HabanaLaunchOpPT::total_recipe_ntbytes += val->ntensorbytes;
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
    std::shared_ptr<RecipeArgumentSpec> &rargpsh,
    std::shared_ptr<RecipeValueSpec> &rvalpsh) {
  if (!is_valid) {
    HabanaLaunchOpPT::recipe_count++;
    is_valid = true;
  } else {
    TORCH_CHECK(HabanaLaunchOpPT::total_recipe_ntbytes >= last_rvalpsh->ntensorbytes,
        "error in total tensor byte accounting, total_recipe_ntbytes ",
        HabanaLaunchOpPT::total_recipe_ntbytes,
        " should be greater than last_recipe.ntensorbytes ",
        last_rvalpsh->ntensorbytes);

    HabanaLaunchOpPT::total_recipe_ntbytes -= last_rvalpsh->ntensorbytes;
  }
  last_rargpsh = rargpsh;
  last_rvalpsh = rvalpsh;
  HabanaLaunchOpPT::total_recipe_ntbytes += last_rvalpsh->ntensorbytes;
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

  TORCH_CHECK(map_.size() == list_.size(),
      "lru cache corruption, map size ", map_.size(),
      " not equal to list_size ", list_.size());

  size_t rcnt{0};
  bool dropped{true};
  while (!map_.empty() && dropped && map_.size() >= max_size_) {
    dropped = drop_lru_impl(rcnt);
    if (!dropped) {
      PT_BRIDGE_DEBUG("all recipes are in use, could not drop any, current recipe count ", rcnt);
    }
  }

  HabanaLaunchOpPT::recipe_count++;

  auto mit = map_.find(key);
  TORCH_CHECK(mit == map_.end(), "problematic key ", key, " another recipe already exists in cache");

  list_.push_front(
      std::pair<std::shared_ptr<RecipeArgumentSpec>, std::shared_ptr<RecipeValueSpec>>(key, val));
  map_.emplace(key, list_.begin());

  HabanaLaunchOpPT::total_recipe_ntbytes += val->ntensorbytes;

  PT_BRIDGE_DEBUG("  adding new recipe, key ",
      key->hashCode(),
      ", ntensorbytes ",
      val->ntensorbytes);

  PT_BRIDGE_DEBUG("  after adding new recipe, nrecipes ",
      HabanaLaunchOpPT::recipe_count,
      " total_recipe_ntbytes ",
      HabanaLaunchOpPT::total_recipe_ntbytes);
}

std::shared_ptr<RecipeValueSpec> RecipeCacheLRU::get(std::shared_ptr<RecipeArgumentSpec>& key) {
  std::lock_guard<std::mutex> lg(mutex_);
  if (exists(key)) {
    TORCH_CHECK(map_.size() == list_.size(),
        "lru cache corruption, map size ", map_.size(),
        " not equal to list_size ", list_.size());

    TORCH_CHECK(exists(key), "Recipe does not exist in map");

    auto mit = map_.find(key);
    list_.splice(list_.begin(), list_, mit->second);

    // wait till the execution complete
    bool use_flag {false};
    do {
      use_flag = list_.front().second->get_use_flag();
      if (use_flag) {
        PT_BRIDGE_DEBUG("waiting for the completion of recipe, key ", key->hashCode());
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

bool RecipeCacheLRU::drop_lru(size_t &recipe_count) {
  std::lock_guard<std::mutex> lg(mutex_);
  bool dropped = drop_lru_impl(recipe_count, true);
  return dropped;
}

bool RecipeCacheLRU::drop_lru_impl(size_t &recipe_count, bool mem_exhausted) {
  bool dropped {false};

  // remove a recipe from the last that is not being used
  if (!map_.empty()) {
    auto lit = list_.end();
    lit--;

    while(lit->second->get_use_flag() == true && lit != list_.begin()) {
      PT_BRIDGE_DEBUG("recipe is in use, key ", lit->first->hashCode(),
          ", ntensorbytes ", lit->second->ntensorbytes);
      lit--;
    }

    // delete the recipe only if it is not in use
    // otherwise the caller need to wait
    if (lit->second->get_use_flag() == false) {
      if (mem_exhausted) {
        PT_BRIDGE_DEBUG("memory exhausted : removing recipe, key ",
            lit->first->hashCode(),
            ", ntensorbytes ",
            lit->second->ntensorbytes);
      } else {
        PT_BRIDGE_DEBUG("lru max size ", max_size_,
            " reached : removing recipe, key ", lit->first->hashCode(),
            ", ntensorbytes ", lit->second->ntensorbytes);
      }

      HabanaLaunchOpPT::recipe_count--;
      HabanaLaunchOpPT::total_recipe_ntbytes -= lit->second->ntensorbytes;

      // Drop the entry from list_ and map_
      map_.erase(lit->first);
      list_.pop_back();
      dropped = true;

      PT_BRIDGE_DEBUG("after dropping lru recipe, nrecipes ",
          HabanaLaunchOpPT::recipe_count,
          " total_recipe_ntbytes ",
          HabanaLaunchOpPT::total_recipe_ntbytes);
    } else {
      PT_BRIDGE_DEBUG("all recipes are in use, can not drop any recipe");
    }
  }

  recipe_count = map_.size();
  return dropped;
}

bool dropCachedRecipe_LRU (size_t &recipe_count) {
  bool dropped {false};
  dropped = RecipeCacheLRU::get_cache().drop_lru(recipe_count);
  return dropped;
}

HabanaLaunchOpPT::HabanaLaunchOpPT(const torch::jit::Node* node, bool debug) {
  subgraph_ = node->g(attr::Subgraph);
  opname_ = node->kind().toQualString();
  std::replace(opname_.begin(), opname_.end(), ':', '_');
  debug_ = debug;
  std::ostringstream oss;
  oss << opname_ << '_' << instance_count_;
  instance_count_++;
  id_str = oss.str();

  PT_BRIDGE_DEBUG("Creating : ", id_str);

  char* caching_policy_str = getenv("HABANA_PGM_CACHING_POLICY");
  if (caching_policy_str != nullptr) {
    if (std::string("simple") == std::string(caching_policy_str)) {
      caching_policy = PGMCachingPolicy::simple;
    } else if (std::string("single") == std::string(caching_policy_str)) {
      caching_policy = PGMCachingPolicy::single;
    } else if (std::string("lru") == std::string(caching_policy_str)) {
      caching_policy = PGMCachingPolicy::lru;
      if (!at::habana::HPUDeviceAllocator::drop_cached_recipe_cb) {
        at::habana::HPUDeviceAllocator::drop_cached_recipe_cb = dropCachedRecipe_LRU;
      }
    }
  }

  value_to_persistent_flag = {};

  tensor_dump_numel_ = -2;

  char* snumel = getenv("HABANA_PGM_DUMP_TENSOR_NUMEL");
  if (snumel != nullptr) {
    tensor_dump_numel_ = atoi(snumel);
    char* wfile_name = getenv("HABANA_PGM_WATCHLIST_FILE");
    if (watchlist_.empty() && wfile_name) {
      std::ifstream wfile(wfile_name);
      TORCH_CHECK(wfile.is_open(), "Unable to open watchlist file ", wfile_name);

      std::string opname;
      while (wfile) {
        getline(wfile, opname);
        watchlist_.insert(opname);
      }
      wfile.close();
    }
  }

  enable_tensor_dump_ = (tensor_dump_numel_ >= -1) ? true : false;
  enable_caching_ = true;
  if (const auto envp = getenv("HABANA_PGM_ENABLE_CACHE")) {
    enable_caching_ = atoi(envp) == 1;
  }

  use_persistent_tensors = false;
  if (const auto envp = getenv("HABANA_USE_PERSISTENT_TENSOR")) {
    use_persistent_tensors = atoi(envp) == 1;
  }

  if (enable_tensor_dump_) {
    struct stat st = {0};
    std::string dir_name{"./tensor_dumps"};
    mode_t dir_mode{0755};

    if (stat(dir_name.c_str(), &st) == -1) {
      auto ret = mkdir(dir_name.c_str(), dir_mode);
      TORCH_CHECK(0 == ret, std::string("failed to create " + dir_name));
    }

    dir_name += std::string("/") + id_str;

    if (stat(dir_name.c_str(), &st) == -1) {
      auto ret = mkdir(dir_name.c_str(), dir_mode);
      TORCH_CHECK(0 == ret, std::string("failed to create " + dir_name));
    }

    tdmp_dir_name_ = dir_name;

    {
      std::ostringstream oss;
      oss << tdmp_dir_name_ << "/"
          << (enable_caching_ ? "tensors_chon" : "tensors_choff")
          << "_pre.tdmp";
      tdmp_file_name_pre_ = oss.str();

      std::ofstream tensor_file;
      tensor_file.open(tdmp_file_name_pre_.c_str());
      tensor_file << "---- id_str : " << id_str << '\n'
                  << "---- tensor dump of the following graph" << '\n';
      tensor_file << subgraph_->toString() << "----" << '\n' << '\n';
      tensor_file.close();
    }

    {
      std::ostringstream oss;
      oss << tdmp_dir_name_ << "/"
          << (enable_caching_ ? "tensors_chon" : "tensors_choff") << ".tdmp";
      tdmp_file_name_ = oss.str();

      std::ofstream tensor_file;
      tensor_file.open(tdmp_file_name_.c_str());
      tensor_file << "---- id_str : " << id_str << '\n'
                  << "---- tensor dump of the following graph" << '\n';
      tensor_file << subgraph_->toString() << "----" << '\n' << '\n';
      tensor_file.close();
    }

    if (tensor_dump_numel_ > 0) {
      htensor_wbuff_size = sizeof(float) * tensor_dump_numel_;
    }
  }
}

HabanaLaunchOpPT::~HabanaLaunchOpPT() {
  PT_BRIDGE_DEBUG("Destroying : ", id_str);
}

habana::LayoutFormat getPTTensorLayout(at::Tensor& tensor) {
  auto mem_format = tensor.suggest_memory_format();
  if (mem_format == at::MemoryFormat::ChannelsLast ||
      mem_format == at::MemoryFormat::ChannelsLast3d)
    return habana::LayoutFormat::NHWC;
  else
    return habana::LayoutFormat::NCHW;
}

habana::LayoutFormat HabanaLaunchOpPT::getTensorChannelOrder(
    torch::jit::Value* val) {
  // The value of the node keeps the tensor physical layout memorized
  // We can update this later if we see any changes to the way layouts are
  // handled
  TORCH_CHECK(
      value_to_tensor_layout.find(val) != std::end(value_to_tensor_layout),
      "HabanaFusion : Channel order not updated");
  return value_to_tensor_layout[val].layout;
}

// See if we are in any leagally accepted channel orders
bool HabanaLaunchOpPT::isChannelOrderSupported(
    torch::jit::Value* val,
    const habana::LayoutFormat& supported_channel_order) {
  return (supported_channel_order == habana::LayoutFormat::ANY) ||
      (supported_channel_order == getTensorChannelOrder(val));
}

bool HabanaLaunchOpPT::isInGraphOutputs(torch::jit::Value* value) {
  auto graph_outs = subgraph_->outputs();
  for (auto value_out : graph_outs) {
    if (value->unique() == value_out->unique()) {
      return true;
    }
  }
  return false;
}

bool HabanaLaunchOpPT::isInGraphOutputs(torch::jit::Node* node, size_t index) {
  auto node_outs = node->outputs();
  TORCH_CHECK(index <= node_outs.size());

  return isInGraphOutputs(node_outs[index]);
}

std::vector<bool> HabanaLaunchOpPT::nodeOutputPersistence(
    torch::jit::Node* node) {
  auto node_outs = node->outputs();
  std::vector<bool> is_persistent{};
  for (auto value_out : node_outs) {
    if (use_persistent_tensors) {
      // Highest priority is given to the env variable
      is_persistent.emplace_back(true);
    } else if (
        value_to_persistent_flag.find(value_out) !=
        value_to_persistent_flag.end()) {
      // If we use per tensor persistence flag, it takes next higher priority
      is_persistent.emplace_back(value_to_persistent_flag[value_out]);
    } else {
      // If no specific flag is set, a tensor is persistent if it goes to graph
      // output
      is_persistent.emplace_back(isInGraphOutputs(value_out));
    }
  }
  return is_persistent;
}

void HabanaLaunchOpPT::GetSynapseInputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node) {
  auto node_ins = node->inputs();
  int input_idx = 0;
  for (const auto value_in : node_ins) {
    if (value_to_ivalue[value_in] && (value_to_ivalue[value_in]->isTensor() ||
                                      value_to_ivalue[value_in]->isTensorList())) {
      // special case for avg pool backward, we only need to set 1 input, since
      // TPC kernel expects only 1 input
      if (!strcmp("aten::avg_pool2d_backward", node->kind().toQualString()) &&
          (input_idx > 0)) {
        continue;
      }

      // Find if an input tensor is already mapped
      // NB: It seems Habana doesn't support shared input to
      // different nodes in graph
      auto is_already_mapped =
          pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
          std::end(pt_to_synapse_tensors);

      SharedSynTensorOrRefListPtr tensorList = std::make_shared<SynTensorOrRefList>();
      if (is_already_mapped) {
        auto syn_tensor_input =
            pt_to_synapse_tensors.find(value_to_ivalue[value_in]);
        for (synapse_helpers::tensor& tensor : *(syn_tensor_input->second)) {
          synapse_helpers::tensor& syn_tensor =
              habana_op->SetSynapseInput(std::move(tensor));
          tensorList->emplace_back(tensor_or_ref(syn_tensor));
        }

        pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
      } else {
        std::vector<at::Tensor> pyTensorList;
        if (value_to_ivalue[value_in]->isTensor()) {
          pyTensorList.emplace_back(value_to_ivalue[value_in]->toTensor());
        }
        else {
          c10::List<at::Tensor> pytList = value_to_ivalue[value_in]->toTensorList();
          for (at::Tensor pyTensor : pytList) {
            pyTensorList.emplace_back(pyTensor);
          }
        }

        std::vector<TensorInfo> tiv;
        for (auto& pt_tensor : pyTensorList) {
          if (!pt_tensor.defined()) {
            continue;
          }
          auto& syn_tensor =
              habana_op->AllocateSynapseInput(*syn_graph_ptr, pt_tensor, true);

          tensorList->emplace_back(tensor_or_ref(syn_tensor));

          std::string irn = "%"+value_in->debugName();
          TensorInfo ti (pt_tensor, syn_tensor.tensor_name_, irn);
          tiv.push_back(ti);
        }

        if (tensorList->empty()) {
          continue;
        }

        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);

        if (enable_caching_) {
          input_tiv_map.emplace(value_to_ivalue[value_in], tiv);
        } else {
          input_tivs.emplace_back(tiv);
        }
      }
      input_idx++;
    }
  }
}

void HabanaLaunchOpPT::GetSynapseOutputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node) {
  auto output_tensors_pt = habana_op->GetOutputs();
  auto& output_tensors_syn = habana_op->GetSynOutputs();
  auto& excluded_out_indices = habana_op->GetSynOutputIndicesExcludedInNode();

  auto output_nodes = node->outputs();
  auto habana_kernel_meta_data = habana_op->GetKernelMetaData();
  habana::LayoutFormat out_layout;

  /* Note the input layout information for the node to pass on to output edge */
  auto node_ins = node->inputs();
  habana::LayoutFormat assigned_input_layout = habana::LayoutFormat::NCHW;
  habana::LayoutFormat origin_input_layout = habana::LayoutFormat::NCHW;

  int node_idx = 0;
  for (auto value_in : node_ins) {
    if (value_to_ivalue[value_in] &&
        value_in->type()->kind() == c10::TypeKind::TensorType) {
      /* Get the input tensor layout information */
      assigned_input_layout = node_idx == 0 ? getTensorChannelOrder(value_in) : assigned_input_layout;
      //Get the origin layout too, to pass it along..we see if any of the inputs in NHWC origin
      //then we mark the origin layout as NHWC
      //We need to make this more robust by having a tensor level memory of layout
      //We need to mark weight tensors by meta data so that we can recognize them
      //and not permute to NHWC at exit.
      origin_input_layout = value_to_tensor_layout[value_in].layout_at_graph_entry ==
                            habana::LayoutFormat::NHWC ? habana::LayoutFormat::NHWC : origin_input_layout;
      node_idx++;
    }
  }

  size_t output_nodes_idx = 0, output_tensor_idx = 0;
  TORCH_CHECK(
      output_nodes.size() ==
          output_tensors_pt.size() - excluded_out_indices.size(),
      "HabanaFusionOp Lowering: Number of output nodes generated doesnt match the graph");

  size_t meta_size = habana_kernel_meta_data.output_layout.size();
  for (synapse_helpers::tensor& out_tensor_syn : output_tensors_syn) {
    out_layout = output_tensor_idx >= meta_size
        ? habana::LayoutFormat::ANY
        : habana_kernel_meta_data.output_layout.at(output_tensor_idx);

    /* Pass down the layout information from input to output for layout agnostic
       output (only for single input ans single output op nodes) */
    value_to_tensor_layout[output_nodes[output_nodes_idx]].layout =
        out_layout == habana::LayoutFormat::ANY ? assigned_input_layout
                                                : out_layout;
    value_to_tensor_layout[output_nodes[output_nodes_idx]].layout_at_graph_entry = origin_input_layout;

    if (excluded_out_indices.find(output_tensor_idx) ==
        excluded_out_indices.end()) {
      IValPtrShared ivpsh =
          std::make_shared<IVal>(output_tensors_pt[output_tensor_idx]);
      value_to_ivalue[output_nodes[output_nodes_idx]] = ivpsh;

      if (use_persistent_tensors &&
          false == isInGraphOutputs(output_nodes[output_nodes_idx])) {
        aten_intermediates.push_back(ivpsh->toTensor());
      }
      SharedSynTensorOrRefListPtr tensorList = std::make_shared<SynTensorOrRefList>();
      tensorList->emplace_back(tensor_or_ref(out_tensor_syn));
      pt_to_synapse_tensors.emplace(
          value_to_ivalue[output_nodes[output_nodes_idx]], tensorList);

      if (use_persistent_tensors ? true
                                 : isInGraphOutputs(node, output_nodes_idx)) {
        output_tensorinfos.emplace_back(TensorInfo(
            ivpsh,
            out_tensor_syn.tensor_name_,
            output_nodes[output_nodes_idx]));
      }

      output_nodes_idx++;
    }
    output_tensor_idx++;
  }
}

at::IntArrayRef getDimsForLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order) {
  at::IntArrayRef dims;

  if (current_order == habana::LayoutFormat::NCHW) {
    if (channel_order == habana::LayoutFormat::NHWC) {
      dims = {0, 2, 3, 1};
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      dims = {2, 3, 1, 0};
    } else {
      TORCH_CHECK(
          0, " Habana Fusion op permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::NHWC) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      dims = {0, 3, 1, 2};
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      dims = {1, 2, 3, 0};
    } else {
      TORCH_CHECK(
          0, " Habana Fusion op permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::HWCK) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      dims = {3, 2, 0, 1};
    } else if (channel_order == habana::LayoutFormat::NHWC) {
      dims = {3, 0, 1, 2};
    } else {
      TORCH_CHECK(
          0, " Habana Fusion op permute called for unsupported channel order");
    }
  } else {
    TORCH_CHECK(
        0, " Habana Fusion op permute called for unsupported channel order");
  }

  return dims;
}

// For now, we permute tensors at graph leaves once
// THis function permutes a given tensor to desired layout and modifies
// input_tensor list to have the new tensor

at::Tensor HabanaLaunchOpPT::permuteTensor(
    torch::jit::Value* value_in,
    const at::Tensor& input,
    habana::LayoutFormat permute_order) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  HabanaOperatorPtr permute_kernel = habana::KernelRegistry().get(
      device_id, "aten::permute", input.scalar_type());
  TORCH_CHECK(
      permute_kernel != nullptr,
      " \n Permute kernel isnt supported in graph mode ");

  TORCH_CHECK(value_to_ivalue[value_in]->isTensor(),
      "non tensor input for permute");

  habana_kernels.push_back(permute_kernel);
  // set input synapse tensors
  auto is_already_mapped =
      pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
      std::end(pt_to_synapse_tensors);

  SharedSynTensorOrRefListPtr tensorList = std::make_shared<SynTensorOrRefList>();
  if (is_already_mapped) {
    auto syn_tensor_input =
        pt_to_synapse_tensors.find(value_to_ivalue[value_in]);

    for (synapse_helpers::tensor& tensor : *(syn_tensor_input->second)) {
      synapse_helpers::tensor& syn_tensor =
        permute_kernel->SetSynapseInput(std::move(tensor));
      tensorList->emplace_back(tensor_or_ref(syn_tensor));
    }
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
  } else {
    auto pt_tensor = value_to_ivalue[value_in]->toTensor();
    auto& syn_tensor =
        permute_kernel->AllocateSynapseInput(*syn_graph_ptr, pt_tensor, true);
    tensorList->emplace_back(tensor_or_ref(syn_tensor));
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);

    TensorInfo ti(value_to_ivalue[value_in], syn_tensor.tensor_name_, value_in);
    if (enable_caching_) {
      input_tiv_map.emplace(value_to_ivalue[value_in], ti);
    } else {
      input_tivs.emplace_back(ti);
    }
  }

  auto dims = getDimsForLayout(permute_order, value_to_tensor_layout[value_in].layout);

  torch::jit::Stack input_stack = {IValue(input), IValue(dims)};
  // setup the config params for the kernels
  bool persistent = isInGraphOutputs(value_in);
  permute_kernel->AllocateAndAddSynapseNode(
      *syn_graph_ptr, input_stack, persistent);
  auto outputs_permute = permute_kernel->GetOutputs();

  // set output synapse tensor
  auto& output_tensors_syn = permute_kernel->GetSynOutputs();
  for (synapse_helpers::tensor& out_tensor_syn : output_tensors_syn) {
    // make the output of permute the input for next synapse kernel
    // permute has a single output
    SharedSynTensorOrRefListPtr tensorList = std::make_shared<SynTensorOrRefList>();
    tensorList->emplace_back(tensor_or_ref(out_tensor_syn));
    if (persistent) {
      aten_intermediates.push_back(value_to_ivalue[value_in]->toTensor());
    }
    value_to_ivalue[value_in] = std::make_shared<IVal>(outputs_permute[0]);
    value_to_tensor_layout[value_in].layout = permute_order;
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);

    if (persistent) {
      output_tensorinfos.emplace_back(TensorInfo(
          value_to_ivalue[value_in], out_tensor_syn.tensor_name_, value_in));
    }
  }
  return outputs_permute[0];
}

bool HabanaLaunchOpPT::isInGraphInputs(torch::jit::Value* value) {
  auto graph_ins = subgraph_->inputs();
  for (auto value_in : graph_ins) {
    if (value->unique() == value_in->unique()) {
      return true;
    }
  }
  return false;
}

void HabanaLaunchOpPT::create_duplicate_syn_tensor(
    at::Tensor* tensor,
    torch::jit::Value* value_in,
    bool persistence) {
  auto syn_tensorlist_input = pt_to_synapse_tensors.find(value_to_ivalue[value_in]);
  TORCH_CHECK(syn_tensorlist_input->second->size() == 1,
      "not implemented the handling of syn_tensorlist_input size ", syn_tensorlist_input->second->size());
  synapse_helpers::tensor& syn_tensor_input = syn_tensorlist_input->second->back();

  auto dtype = tensor->scalar_type();
  // if both are persistent, use same memeory section
  if (syn_tensor_input.is_persistent() && persistence) {
    // create a tensor variant on the same memory section as the input
    auto variant =
        synapse_helpers::tensor_builder(
            tensor->sizes(), habana_helpers::pytorch_to_synapse_type(dtype))
            .mark_persistence(true)
            .with_memory_section(syn_tensor_input.memorysection())
            .build(
                synapse_helpers::HPURegistrar::get_device(
                    tensor->device().index()),
                syn_tensor_input.graph());

    meta_syn_tensors.push_back(
        absl::get<synapse_helpers::tensor>(std::move(variant)));

    TensorInfo ti(value_to_ivalue[value_in], meta_syn_tensors.back().tensor_name_, value_in);
    if (!isInGraphOutputs(value_in)) {
      duplicate_tivs.emplace_back(ti);
    }
    else {
      output_tensorinfos.emplace_back(ti);
    }
  } else {
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    auto variant = habana_helpers::create_tensor(
        *tensor, syn_tensor_input.graph(), persistence);
    meta_syn_tensors.push_back((std::move(variant)));
  }

  auto& syn_tensor = meta_syn_tensors.back();
  pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);

  SharedSynTensorOrRefListPtr tensorList = std::make_shared<SynTensorOrRefList>();
  tensorList->emplace_back(tensor_or_ref(syn_tensor));
  pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
}
void adjustInputWeight(at::Tensor* tensor, bool is_input) {
  if (tensor->dim() != 4)
    return;

  auto sizes = tensor->sizes().vec();
  auto strides = tensor->strides().vec();
  at::IntArrayRef in = {2, 3, 1, 0};
  at::IntArrayRef out = {3, 2, 0, 1};
  // TODO : Remove these hardcoded dims, maybe take it from config file?
  at::IntArrayRef new_pos_arr = is_input ? in : out;
  auto new_pos = new_pos_arr.vec();
  std::vector<long int> swapped_sizes = {sizes[new_pos[0]],
                                         sizes[new_pos[1]],
                                         sizes[new_pos[2]],
                                         sizes[new_pos[3]]};
  std::vector<long int> swapped_strides = {strides[new_pos[0]],
                                           strides[new_pos[1]],
                                           strides[new_pos[2]],
                                           strides[new_pos[3]]};
  tensor->unsafeGetTensorImpl()->set_sizes_and_strides(
      swapped_sizes, swapped_strides);
}

void HabanaLaunchOpPT::processInputs(
    torch::jit::Node* node,
    const HabanaOperatorPtr& habana_kernel) {
  // Get the metadata for all inputs, used for preprocessing inputs
  auto& habana_kernel_meta_data = habana_kernel->GetKernelMetaData();
  // Check if its ok to change the input tensor in the graph attached to value
  auto node_ins = node->inputs();
  size_t tensor_idx = 0;
  habana::LayoutFormat in_layout, prev_layout = habana::LayoutFormat::ANY;
  size_t meta_size = habana_kernel_meta_data.input_layout.size();
  for (const auto value_in : node_ins) {
    if (value_to_ivalue[value_in] &&
        value_in->type()->kind() == c10::TypeKind::TensorType) {
      in_layout = tensor_idx >= meta_size
          ? habana::LayoutFormat::ANY
          : habana_kernel_meta_data.input_layout.at(tensor_idx);

      if (in_layout == habana::LayoutFormat::ANY && tensor_idx > 0) {
        // ATTENTION : We will support only homogeneous layouts for kernels
        // which dont pass meta data requirements for inputs
        // We make inputs homogeneous layouts in case kernel doesnt specify any
        // layout
        // TODO : Add a debug log heres
        in_layout = prev_layout;
      }

      auto tensor = value_to_ivalue[value_in]->toTensor();

      if (in_layout == habana::LayoutFormat::HWCK) {
        in_layout = habana::LayoutFormat::ANY;
        value_to_tensor_layout[value_in].layout = habana::LayoutFormat::HWCK;
      }

      if (!(isChannelOrderSupported(value_in, in_layout))) {
        // We only support 4D tensors
        TORCH_CHECK(
            tensor.dim() <= 4,
            "WARNING: Kernel wants permute on non 4D tensor, not supproted");
        // permute
        if (tensor.dim() == 4) {
          permuteTensor(value_in, tensor, in_layout);
        }
      }
      prev_layout =
          tensor_idx == 0 ? getTensorChannelOrder(value_in) : prev_layout;
      tensor_idx++;
    }
  }
  // TODO : add checks for doing flattening/slicing anything that is
  // required.
}

void HabanaLaunchOpPT::postProcessOutputs() {
  // Do we need a optimization pass here? What should we look for?
  for (auto node : subgraph_->nodes()) {
    auto node_outs = node->outputs();
    for (const auto value_out : node_outs) {
      IValPtrShared ival = value_to_ivalue[value_out];
      if (!ival)
        continue;
      if (!(ival->isTensor()))
        continue;

      if (ival && value_out->type()->kind() == c10::TypeKind::TensorType &&
          isInGraphOutputs(value_out)) {
        auto tensor = ival->toTensor();
        // Add permutes only for 4D non weight tensors
        if (tensor.dim() == 4) {
          auto pre_layout = value_to_tensor_layout[value_out].layout_at_graph_entry;
          if (getTensorChannelOrder(value_out) == habana::LayoutFormat::HWCK) {
            //Do Nothing
          } else if (getTensorChannelOrder(value_out) != pre_layout) {
            permuteTensor(value_out, tensor, pre_layout);
            if (pre_layout == habana::LayoutFormat::NHWC) {
              // Make the shape according to NCHW again as PT maintains that
              // even for NHWC tensors Whereas we process internally as NHWC
              // shape only
              adjustSizesforPT(&tensor, true);
              value_to_ivalue.erase(value_out);
              value_to_ivalue[value_out] = std::make_shared<IVal>(tensor);
            }
          } else {
            if (getTensorChannelOrder(value_out) ==
                habana::LayoutFormat::NHWC) {
              // Make the shape according to NCHW again as PT maintains that
              // even for NHWC tensors Whereas we process internally as NHWC
              // shape only
              adjustSizesforPT(&tensor, true);
              value_to_ivalue.erase(value_out);
              value_to_ivalue[value_out] = std::make_shared<IVal>(tensor);
            }
          }
        }
      }
      // TODO : add checks for doing flattening/slicing anything that is
      // required.
    }
  }
}

void HabanaLaunchOpPT::handlePrimNodes(torch::jit::Node* node) {
  TORCH_CHECK(
      node->kind() == torch::jit::prim::Constant,
      "Habana Fusion only supports constant type prim nodes");
  auto node_vals = node->outputs();
  for (const auto value : node_vals) {
    IValPtrShared ivptrsh = std::make_shared<IVal>(toIValue(value).value());
    value_to_ivalue[value] = ivptrsh;
  }
}

torch::jit::Stack HabanaLaunchOpPT::getStackForNode(torch::jit::Node* node) {
  torch::jit::Stack stack_in;
  auto node_inputs = node->inputs();
  for (auto input : node_inputs) {
    if (value_to_ivalue[input])
      stack_in.insert(stack_in.end(), *value_to_ivalue[input]);
    else
      stack_in.insert(stack_in.end(), IValue());
  }
  return stack_in;
}

c10::ScalarType HabanaLaunchOpPT::getNodeScalarType(torch::jit::Node* node) {
  // return the data type of first input tensor
  for (auto input : node->inputs()) {
    if (value_to_ivalue[input] && value_to_ivalue[input]->isTensor()) {
      return value_to_ivalue[input]->toTensor().scalar_type();
    }
  }
  // Default return float for now if no tensor found
  return c10::ScalarType::Float;
}

void HabanaLaunchOpPT::handleMetaOps(torch::jit::Node* node) {
  // Call the meta op via CPU impl
  // Some ops dont support c10 op.callBoxed so we need to call via JIT
  torch::jit::Stack stack;
  void *in_data, *out_data;
  auto node_ins = node->inputs();
  habana::LayoutFormat out_layout, out_origin_layout;
  IValPtrShared input_ptr{nullptr};


  for (const auto value_in : node_ins) {
    stack.insert(stack.end(), *value_to_ivalue[value_in]);
    if (value_to_ivalue[value_in]->isTensor()) {
      auto tensor = value_to_ivalue[value_in]->toTensor();
      out_layout = value_to_tensor_layout[value_in].layout;
      out_origin_layout = value_to_tensor_layout[value_in].layout_at_graph_entry;
      if (pt_to_synapse_tensors.find(value_to_ivalue[value_in]) ==
          std::end(pt_to_synapse_tensors)) {
        in_data = tensor.data_ptr();
        input_ptr = value_to_ivalue[value_in];
        auto dtype = tensor.scalar_type();
        meta_syn_tensors.push_back(habana_helpers::create_tensor(
            tensor, syn_graph_ptr->get_graph_handle(), true, dtype));
        SharedSynTensorOrRefListPtr tensorList = std::make_shared<SynTensorOrRefList>();
        tensorList->emplace_back(tensor_or_ref(meta_syn_tensors.back()));
        pt_to_synapse_tensors.emplace(
            value_to_ivalue[value_in], tensorList);

        if (enable_caching_) {
          input_tiv_map.emplace(
              value_to_ivalue[value_in],
              TensorInfo(
                  value_to_ivalue[value_in],
                  meta_syn_tensors.back().tensor_name_,
                  value_in));
        } else {
          input_tivs.emplace_back(TensorInfo(
              value_to_ivalue[value_in],
              meta_syn_tensors.back().tensor_name_,
              value_in));
        }
      }
    }
  }
  torch::jit::Operator jit_op = node->getOperator();
  auto offset = jit_op.getOperation()(stack);

  TORCH_CHECK(offset == 0);

  auto node_outs = node->outputs();
  auto outputs = last(stack, node_outs.size());
  int i = 0;
  for (const auto val_out : node_outs) {
    IValPtrShared ival = std::make_shared<IVal>(outputs[i]);
    value_to_ivalue[val_out] = ival;
    if (ival->isTensor()) {
      auto tensor = ival->toTensor();
      value_to_tensor_layout[val_out].layout = out_layout;
      value_to_tensor_layout[val_out].layout_at_graph_entry = out_origin_layout;
      create_duplicate_syn_tensor(&tensor, val_out, true);
      out_data = tensor.data_ptr();
    }
    i++;
  }
  TORCH_CHECK(
      in_data == out_data, "HabanaFusion : Data pointer changed in Meta op");
}

void HabanaLaunchOpPT::OrderInputs(RecipeValueSpec& rv) {
  if (enable_caching_) {
    // Order the input_tivs according to the order of suggraph inputs
    size_t i = pt_stack_sh.size() - num_inputs;
    for (; i < pt_stack_sh.size(); i++) {
      IValPtrShared ivpsh = pt_stack_sh.at(i);
      if (ivpsh->isTensor() || ivpsh->isTensorList()) {
        auto it = input_tiv_map.find(ivpsh);
        if (it != input_tiv_map.end()) {
          input_tivs.push_back(it->second);
        } else {
          TORCH_CHECK(false, "synapse tensor not found");
        }
      }
    }
    TORCH_CHECK(
        input_tivs.size() == num_tensor_inputs,
        "number of input tensors ", num_tensor_inputs, " mismatch with #input_tivs ", input_tivs.size());
  }
}

void HabanaLaunchOpPT::FlattenAndLinkInputTIVs(RecipeValueSpec& rv) {
  // dtensorinfos maintain the flattened tinfo list
  rv.dtensorinfos =
      std::make_shared<std::vector<TensorInfo>>(std::vector<TensorInfo>());

  std::unordered_map<void *, size_t> buff_to_inputtividx_map;
  for (auto & tiv : input_tivs) {
    if (absl::holds_alternative<TensorInfo>(tiv)) {
      const auto ti = absl::get<TensorInfo>(tiv);
      rv.dtensorinfos->push_back(ti);
      if (enable_caching_) {
        buff_to_inputtividx_map.emplace(ti.buffer, rv.dtensorinfos->size()-1);
      }
    }
    else if (absl::holds_alternative<std::vector<TensorInfo>>(tiv)) {
      for (const auto & ti : absl::get<std::vector<TensorInfo>>(tiv)) {
        rv.dtensorinfos->push_back(ti);
        if (enable_caching_) {
          buff_to_inputtividx_map.emplace(ti.buffer, rv.dtensorinfos->size()-1);
        }
      }
    }
    else {
      TORCH_CHECK(false, "Error condition for input tiv");
    }
  }
  // At this point inputs tinfos are populated
  rv.num_inputs = rv.dtensorinfos->size();

  // Link the input tivs with the duplicate
  size_t nduplicates {0};
  for (auto & tiv : duplicate_tivs) {
    if (absl::holds_alternative<TensorInfo>(tiv)) {
      auto ti = absl::get<TensorInfo>(tiv);
      if (enable_caching_) {
        auto it_parent = buff_to_inputtividx_map.find(ti.buffer);
        TORCH_CHECK(buff_to_inputtividx_map.end() != it_parent,
            "parent tinfo is missing for input duplicate");
        ti.is_duplicate = true;
        size_t parent_idx = it_parent->second;
        TORCH_CHECK(parent_idx < num_inputs,
            "out of bound parent index : ", parent_idx, " for ", ti.syn_name);
        ti.parent_index = parent_idx;
      }
      rv.dtensorinfos->push_back(ti);
      nduplicates++;
    }
    else {
      TORCH_CHECK(false, "duplicate tiv must be a tensor");
    }
  }
  TORCH_CHECK(nduplicates == duplicate_tivs.size(),
      "#duplicate_tivs ", duplicate_tivs.size(),
      " is not matching with num_duplicates ", nduplicates);

  rv.num_duplicates = nduplicates;

  // At this point inputs and duplicate tinfos are populated
  TORCH_CHECK((rv.num_inputs + rv.num_duplicates == rv.dtensorinfos->size()),
      "num_inputs ", rv.num_inputs,
      "num_duplicates ", rv.num_duplicates,
      " are not adding up to #dtensorinfos ", rv.dtensorinfos->size());
}

void HabanaLaunchOpPT::CompileAndExecuteHabanaFusedOpKernel() {
  // figure out the right device id
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  std::ostringstream oss;
  oss << opname_ << "_" << instance_count_;
  synapse_helpers::graph syn_graph =
      habana_helpers::create_graph(device_id, oss.str());
  syn_graph_ptr = &syn_graph;

  // for each node in IR graph, at this point the graph is a list with nodes
  // topoloically sorted
  // TODO : check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = subgraph_->nodes();
  for (auto* node : graph_nodes) {
    TensorInfo::watch_tensor_flag = false;
    std::string opname(node->kind().toQualString());
    if (watchlist_.empty() ||
        watchlist_.find(opname) != watchlist_.end()) {
      TensorInfo::watch_tensor_flag = true;
    }

    // Prim nodes require special handling and are a special case
    if (node->kind().is_prim()) {
      handlePrimNodes(node);
      continue;
    }

    // If its a meta op we need to call the CPU impl and capture changes
    // Only valid for single tensor ops
    // Can we avoid the string match here?
    if (HabanaMetaOpList::isHabanaMetaOp(node->kind().toQualString())) {
      handleMetaOps(node);
      continue;
    }
    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::KernelRegistry().get(
        device_id, node->kind().toQualString(), getNodeScalarType(node));

    TORCH_CHECK(
        HabanaKernel != nullptr,
        std::string(" \n  kernel ") + std::string(node->kind().toQualString()) +
            std::string(" isnt supported in graph mode "));

    // See if we need to modify/permute tesnors
    processInputs(node, HabanaKernel);

    // Create/attach the synapse inputs from aten tensors
    GetSynapseInputs(HabanaKernel, node);

    torch::jit::Stack input_stack = getStackForNode(node);

    // setup the config params for the kernels
    auto outputPersistent = nodeOutputPersistence(node);
    if (outputPersistent.size() == 1) {
      HabanaKernel->AllocateAndAddSynapseNode(
          syn_graph, input_stack, outputPersistent[0]);
    } else {
      HabanaKernel->AllocateAndAddSynapseNode(
          syn_graph, input_stack, outputPersistent);
    }

    // Get the output tensors created back from the kernel
    // We set type so that the created tensor is propagated throughout graph
    GetSynapseOutputs(HabanaKernel, node);

    auto patch_info = HabanaKernel->getAppendedTensorInfos();
    if (!patch_info.empty()) {
      for (const auto& p : patch_info) {
        std::string irn{"%interim"};
        interim_tensorinfos.emplace_back(TensorInfo(p.second, p.first, irn));
        aten_intermediates.push_back(p.second);
      }
    }

    // Adding to a vector as we share context through shared pointers and we
    // dont want to call delete untill we are done with whole graph
    habana_kernels.push_back(HabanaKernel);
  }

  postProcessOutputs();

  TORCH_CHECK(false == syn_graph.is_empty(), "empty graph encountered");

  auto&& error_variant{syn_graph.compile()};
  if (ABSL_PREDICT_FALSE(
          absl::holds_alternative<synapse_helpers::synapse_error>(
              error_variant))) {
    auto& error = absl::get<synapse_helpers::synapse_error>(error_variant);
    PT_BRIDGE_FATAL(
        "syn compile encountered : ", error.error, " ", error.status);
    TORCH_CHECK(false, "syn compile failed");
  }

  auto cur_recipe = get_value(std::move(error_variant));
  //RecipeValueSpec rv (cur_recipe);

  std::shared_ptr<RecipeValueSpec> rvalpsh =
      std::make_shared<RecipeValueSpec>(cur_recipe);

  RecipeValueSpec &rv = *rvalpsh;

  // output_tensorinfos is populated during compile and does not need any post
  // processing, whereas input_tivs need to be reordered

  OrderInputs(rv);

  // rv.num_inputs and rv.num_duplicates will be set by FlattenAndLinkInputTIVs
  FlattenAndLinkInputTIVs(rv);

  if (!interim_tensorinfos.empty()) {
    rv.num_interims = interim_tensorinfos.size();
    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        interim_tensorinfos.begin(),
        interim_tensorinfos.end());
  }

  // At this point inputs, duplicate and interim tinfos are populated
  TORCH_CHECK((rv.num_inputs + rv.num_duplicates + rv.num_interims == rv.dtensorinfos->size()),
      "num_inputs ", rv.num_inputs,
      "num_duplicates ", rv.num_duplicates,
      "num_interims ", rv.num_interims,
      " are not adding up to #dtensorinfos ", rv.dtensorinfos->size());

  for (auto & ti : *rv.dtensorinfos) {
    if (!ti.is_duplicate) {
      rv.ntensorbytes += ti.size;
    }
  }

  rv.dtensorinfos->insert(
      rv.dtensorinfos->end(),
      output_tensorinfos.begin(),
      output_tensorinfos.end());

  rv.num_outputs = output_tensorinfos.size();
  rv.num_tensors = rv.dtensorinfos->size();

  // At this point the inputs, duplicate, interim and output tinfos are populated
  TORCH_CHECK(
      (rv.num_inputs + rv.num_duplicates + rv.num_interims + rv.num_outputs) ==
      rv.dtensorinfos->size(),
      "num_inputs ", rv.num_inputs,
      "num_duplicates ", rv.num_duplicates,
      "num_interims ", rv.num_interims,
      " are not adding up to #dtensorinfos ", rv.dtensorinfos->size());

  if (enable_tensor_dump_) {
    if (0 == htensor_wbuff_size) {
      for (size_t i = 0; i < rv.num_tensors; ++i) {
        htensor_wbuff_size = std::max(htensor_wbuff_size, rv.dtensorinfos->at(i).size);
      }
    }

    if (!htensor_wbuff) {
      synStatus status;
      status = synHostMalloc(
          device_id, htensor_wbuff_size, 0, (void**)&(htensor_wbuff));
      TORCH_CHECK(status == synSuccess, "host-malloc failed");
    }
    rv.htensor_wbuff = htensor_wbuff;
    rv.htensor_wbuff_size = htensor_wbuff_size;
  }

  rv.aten_outputs = std::make_shared<std::vector<IValPtrShared>>(
      std::vector<IValPtrShared>());
  for (auto output : subgraph_->outputs()) {
    auto oit = value_to_ivalue.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue.end(),
        "value_to_ivalue does not have an entry for %",
        output->debugName());
    if (oit != value_to_ivalue.end()) {
      IValPtrShared ivpsh = oit->second;
      rv.aten_outputs->push_back(ivpsh);
    }
  }

  rv.aten_intermediates = std::move(aten_intermediates);

  if (enable_tensor_dump_) {
    DumpTensors_pre(rv);
  }

  LaunchRecipe(rv, input_refs);

  if (enable_tensor_dump_) {
    DumpTensors(rv);
  }

  if (enable_caching_) {
    // Add the <key,value> pair to the map
    std::shared_ptr<RecipeArgumentSpec> rargpsh =
        std::make_shared<RecipeArgumentSpec>(false, input_refs, subgraph_, id_str);
    rv.key = rargpsh->hashCode();

    switch (caching_policy) {
      case PGMCachingPolicy::simple :
        recipe_cache_simple.add(rargpsh, rvalpsh);
        break;
      case PGMCachingPolicy::single :
        recipe_cache_single.add(rargpsh, rvalpsh);
        break;
      case PGMCachingPolicy::lru :
        RecipeCacheLRU::get_cache().add(rargpsh, rvalpsh);
        break;
      default :
        TORCH_CHECK(false, "should not be reachable");
    }
  }

  UpdateOutputs(rv);
}

void HabanaLaunchOpPT::DumpTensors_pre(RecipeValueSpec& rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(
        tdmp_file_name_pre_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tensors; ++i) {
      if (rv.dtensorinfos->at(i).watch) {
        rv.d2h_dbuff(i);
        rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
      }
    }
    tensor_file.close();
  }
}

void HabanaLaunchOpPT::DumpTensors(RecipeValueSpec& rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(tdmp_file_name_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tensors; ++i) {
      if (rv.dtensorinfos->at(i).watch) {
        rv.d2h_dbuff(i);
        rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
      }
    }
    tensor_file.close();
  }
}

bool HabanaLaunchOpPT::CompileSynapseGraph(
    std::shared_ptr<synapse_helpers::graph::recipe_handle>& synh_recipe) {
  auto compile_result = syn_graph_ptr->compile();
  synh_recipe = get_value(std::move(compile_result));
  return (synh_recipe != nullptr);
}

void HabanaLaunchOpPT::PrintATenTensors(RecipeValueSpec& rv) {
  std::ostream& O = std::cout;

  O << "aten_inputs #" << num_inputs << "::" << '\n';
  for (size_t i = pt_stack_sh.size() - num_inputs; i < pt_stack_sh.size();
       i++) {
    PrintATenTensor(pt_stack_sh.at(i));
  }

  if (rv.aten_intermediates.size()) {
    O << "aten_intermediates #" << rv.aten_intermediates.size() << "::" << '\n';
    for (auto& a : rv.aten_intermediates) {
      PrintATenTensor(a);
    }
  }

  if (rv.aten_outputs) {
    O << "aten_outputs #" << rv.aten_outputs->size() << "::" << '\n';
    for (auto& a : *rv.aten_outputs) {
      PrintATenTensor(a);
    }
  }
}

void HabanaLaunchOpPT::LaunchRecipe(
    RecipeValueSpec& rv,
    at::ArrayRef<torch::jit::IValue> input_refs) {
  rv.SelfCheck();

  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto& stream_handle = device.get_compute_stream();
  std::vector<at::Tensor> ptRefs;
  std::vector<synapse_helpers::device_ptr> outDevPtr;

  if (device.IsStreamASyncEnabled()) {
    // Get the reference to the tensor it is operating on to prevent
    // it from being deallocated while the operation is still in flight.
    std::vector<synapse_helpers::device_ptr> inDevPtr;
    inDevPtr.reserve(rv.num_inputs);
    for (auto& input : input_refs) {
      if (input.isTensor()) {
        at::Tensor tensor = input.toTensor();
        ptRefs.push_back(std::move(tensor));
        inDevPtr.push_back(
            reinterpret_cast<uint64_t>(input.toTensor().data_ptr()));
      }
    }
    // wait for input DMA to complete before launching the compute.
    device.add_wait_events_on_stream(inDevPtr, stream_handle);
    outDevPtr.reserve(rv.num_outputs);
    for (auto& output : *rv.aten_outputs) {
      if (output && output->isTensor()) {
        outDevPtr.push_back(
            reinterpret_cast<uint64_t>(output->toTensor().data_ptr()));
      }
    }
  }

  std::vector<synLaunchTensorInfo> syn_launch_info;

  // Populate the <name,buffer> pairs from TensorInfo for synLaunch
  for (size_t i = 0; i < rv.num_tensors; ++i) {
    syn_launch_info.emplace_back(synLaunchTensorInfo{
        rv.dtensorinfos->at(i).syn_name.c_str(),
        reinterpret_cast<uint64_t>(rv.dtensorinfos->at(i).buffer)});
  }
  synapse_helpers::graph::launch_info ln_info(rv.recipe->device_);
  synapse_helpers::graph::create_launch_info(ln_info, *rv.recipe);

  auto& recipe_counter = device.get_active_recipe_counter();
  recipe_counter.increase();
  auto&& error_optional{
      synapse_helpers::graph::launch(ln_info, *rv.recipe, syn_launch_info)};
  if (ABSL_PREDICT_FALSE(error_optional.has_value())) {
    recipe_counter.decrease_and_notify();
    auto& error = error_optional.value();
    PT_BRIDGE_FATAL(
        "syn launch encountered : ", error.error, " ", error.status);
    TORCH_CHECK(false, "syn launch failed");
  }

  if (device.IsStreamASyncEnabled()) {
    // regsiter an event on the compute
    device.register_producer_on_stream(
        std::move(outDevPtr), stream_handle, [ptRefs, &rv, &recipe_counter]() {
          recipe_counter.decrease_and_notify();
          rv.nop();
          return;
        });
  } else {
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

void HabanaLaunchOpPT::UpdateOutputs() {
  // Update the stack
  drop(*pt_stack, num_inputs);
  for (auto output : subgraph_->outputs()) {
    // pt_stack->insert(pt_stack->end(), *value_to_ivalue[output]);
    if (value_to_ivalue[output])
      pt_stack->insert(pt_stack->end(), *value_to_ivalue[output]);
    else
      pt_stack->insert(pt_stack->end(), IValue());
    // TORCH_CHECK(false, "missing output aten tensor");
  }
}

void HabanaLaunchOpPT::UpdateOutputs(RecipeValueSpec& rv) {
  // Update the stack from the recipe itself
  drop(*pt_stack, num_inputs);
  for (const auto& ivpsh : *(rv.aten_outputs)) {
    pt_stack->insert(pt_stack->end(), *ivpsh);
  }
}

template <typename T>
void HabanaLaunchOpPT::clearMember(T& m_container) {
  T empty;
  using std::swap;
  swap(m_container, empty);
}

void HabanaLaunchOpPT::clear() {
  pt_stack = nullptr;
  pt_stack_sh.clear();
  syn_graph_ptr = nullptr;

  habana_kernels.clear();

  input_tivs.clear();
  duplicate_tivs.clear();
  input_tiv_map.clear();

  interim_tensorinfos.clear();
  output_tensorinfos.clear();
  value_to_tensor_layout.clear();
  value_to_persistent_flag.clear();

  value_to_ivalue.clear();
  pt_to_synapse_tensors.clear();
  meta_syn_tensors.clear();

  aten_intermediates.clear();

  num_tensor_inputs = 0;
}

std::shared_ptr<RecipeValueSpec> HabanaLaunchOpPT::GetCachedRecipe(
    std::shared_ptr<RecipeArgumentSpec>& spec_key) {
  switch (caching_policy) {
    case PGMCachingPolicy::simple :
      return recipe_cache_simple.get(spec_key);
    case PGMCachingPolicy::single :
      return recipe_cache_single.get(spec_key);
    case PGMCachingPolicy::lru :
      return RecipeCacheLRU::get_cache().get(spec_key);
    default :
      TORCH_CHECK(false, "should not be reachable");
  }

  return {nullptr};
}


void HabanaLaunchOpPT::ReturnCachedRecipe(RecipeValueSpec &rv) {
  switch (caching_policy) {
    case PGMCachingPolicy::simple :
      break;
    case PGMCachingPolicy::single :
      break;
    case PGMCachingPolicy::lru :
      rv.set_use_flag(false);
      break;
    default :
      TORCH_CHECK(false, "should not be reachable");
  }
}

void HabanaLaunchOpPT::run(torch::jit::Stack& stack) {
  PT_BRIDGE_BEGIN;
  num_inputs = subgraph_->inputs().size();
  num_tensor_inputs = 0;
  auto subgraph_inputs = subgraph_->inputs();
  input_refs = last(stack, num_inputs);
  ref_count_++;
  iteration_count_++;

  // Keep a handle to the stack for future use
  pt_stack = &stack;

  size_t j = stack.size() - num_inputs;
  for (; j < stack.size(); j++) {
    IValPtrShared ivpsh = std::make_shared<IVal>(stack[j]);
    pt_stack_sh.push_back(ivpsh);
    if (ivpsh->isTensor() || ivpsh->isTensorList()) {
      num_tensor_inputs++;
    }
  }

  // Fusion pass should ensure all nodes are on Habana, if all nodes not on
  // habana device, we should assert
  bool is_all_hpu = true;
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      is_all_hpu = input.toTensor().device().type() != c10::DeviceType::HABANA
          ? false
          : is_all_hpu;
    }
  }

  // We dont support running some ops on CPU while running fused op on Habana
  // All tensors should be alocated to habana before entering this phase
  TORCH_CHECK(
      is_all_hpu == true, " Habana Fusion needs all tensors to be in HPU ");

  // caching :: begin
  if (enable_caching_) {
    std::shared_ptr<RecipeArgumentSpec> spec_key =
        std::make_shared<RecipeArgumentSpec>(false, input_refs, subgraph_, id_str);

    std::shared_ptr<RecipeValueSpec> rvpsh = GetCachedRecipe(spec_key);
    if (ABSL_PREDICT_TRUE(rvpsh)) {
      RecipeValueSpec &rv = *rvpsh;

      PT_BRIDGE_DEBUG("PGM cache hit, key:", spec_key->hashCode(),
          ", ntensorbytes ", rv.ntensorbytes,
          ", total_recipe_ntbytes ", total_recipe_ntbytes);

      // Patch the input buffers
      // Running index on rv.dtensorinfos
      size_t ridx = 0;
      for (auto const& input : input_refs) {
        if (input.isTensor()) {
          rv.dtensorinfos->at(ridx).buffer = input.toTensor().data_ptr();
          ridx++;
        } else if (input.isTensorList()) {
          for (at::Tensor t : input.toTensorList()) {
            rv.dtensorinfos->at(ridx).buffer = t.data_ptr();
            ridx++;
          }
        }
      }

      TORCH_CHECK(ridx ==  rv.num_inputs,
          "running index ", ridx, " mismatch with num_inputs ", rv.num_inputs);

      // Patch the duplicates if there are any
      if (rv.num_duplicates) {
        size_t duplicates_index_end = rv.num_inputs+rv.num_duplicates;
        for (; ridx < duplicates_index_end; ridx++) {
          size_t parent_idx = rv.dtensorinfos->at(ridx).parent_index;
          rv.dtensorinfos->at(ridx).buffer = rv.dtensorinfos->at(parent_idx).buffer;
        }
      }

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }

      LaunchRecipe(rv, input_refs);

      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);

      clear();
      PT_BRIDGE_END;
      return;
    }
    else {
      PT_BRIDGE_DEBUG("PGM cache miss, key : ", spec_key->hashCode());
    }
  }
  // caching :: end
  {
    for (size_t j = 0; j < pt_stack_sh.size(); j++) {
      auto value_input = subgraph_inputs[j];
      value_to_tensor_layout[value_input].layout = habana::LayoutFormat::NCHW;
      value_to_tensor_layout[value_input].layout_at_graph_entry = habana::LayoutFormat::NCHW;

      if (pt_stack_sh[j]->isTensor()) {
        // Taking alias as that allows us to detach it from PT and do metadata
        // changes It gives us more control over tensor changes, but caution is
        // needed. Its might be a bit dangerous, but only way to communicate
        // layour changes PT doesnt allow any stride changes we want, we can
        // review it with PT folks

        auto tensor = at::alias(pt_stack_sh[j]->toTensor());

        // Get  the logical layout from PT tensor
        // We dont touch this, even while doing permutes, the PT logical tensor
        // is retained For us all tensors are contiguous PT doesnt let us mark
        // logical layout directly so we dont change them

        value_to_tensor_layout[value_input].layout = getPTTensorLayout(tensor);
        value_to_tensor_layout[value_input].layout_at_graph_entry = getPTTensorLayout(tensor);

        if (getPTTensorLayout(tensor) == habana::LayoutFormat::NHWC) {
          // Make the sizes according to NCHW as PT maintains
          // NCHW shapes even for NHWC tensors(It doesnt change shape)
          adjustSizesforPT(&tensor, false);
          IValPtrShared ivptrsh = std::make_shared<IVal>(tensor);
          value_to_ivalue[value_input] = ivptrsh;
          pt_stack_sh[j] = ivptrsh;
        } else {
          IValPtrShared ivptrsh = std::make_shared<IVal>(tensor);
          value_to_ivalue[value_input] = ivptrsh;
          pt_stack_sh[j] = ivptrsh;
        }
      } else {
        value_to_ivalue[value_input] = pt_stack_sh[j];
      }
    }
  }

  //<Decription> This is the main function that
  //  a. creates the HabanaLaunchOp
  //  b. compiles and executes the same
  CompileAndExecuteHabanaFusedOpKernel();

  // clear the context
  // TODO : See if we need to add a contect to this object pointer or clearing
  // like this is good?
  clear();

  PT_BRIDGE_END;
}
