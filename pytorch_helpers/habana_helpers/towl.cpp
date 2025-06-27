/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "towl.h"
#include <unistd.h>
#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/memory_defragmentation.h"

// Defined here, should not be used directly inside bridge.

#define PT_TOWL_DEBUG(...) HLLOG_DEBUG(PT_TOWL, FORMAT_AND_MSG(__VA_ARGS__))
#define PT_TOWL_WARN(...) HLLOG_WARN(PT_TOWL, FORMAT_AND_MSG(__VA_ARGS__))

namespace towl::impl {

namespace {

int GetRankFromEnv() {
  auto rank = getenv("RANK");
  if (rank == nullptr) {
    return 0;
  }
  try {
    return std::stoi(rank);
  } catch (...) {
    return -1;
  }
}

/* Configuration of towl-bridge interagration:
 * The environment variable PT_TOWL_LOG_CONFIG contains string with
 * parameters to the logger. Parameters have format 'key=value' and are
 * separated by semicolon.
 *
 * Parameters:
 *  log_devmem=[0|1]       - enables devmem logging category
 *  log_recipe=[0|1]       - enables recipe logging category
 *  log_python=[0|1]       - enables python logging category
 *  log_collective=[0|1]   - enables collective logging category
 *  log_defrag=[0|1]       - enables defragmenter logging category
 *  log_copy=[0|1]         - enables copy logging category
 *  log_metrics=[0|1]      - enables metrics logging category
 *  rank=int               - logs only under given rank (determined by env RANK)
 *  any_rank=[0|1]         - ignore `rank` option and always log events
 */

struct Config {
  bool log_devmem_buf = true;
  bool log_devmem_summary = true;
  bool log_recipe = true;
  bool log_recipe_compile = true;
  bool log_python = true;
  bool log_collective = true;
  bool log_defrag = true;
  bool log_copy = true;
  bool log_metrics = false;
  int rank = -1;
  bool any_rank = false;

  static Config parse(const std::string& config_str) {
    std::vector<std::pair<std::string, std::string>> keyvals;
    Config config;

    auto addField = [&](const std::string& field) {
      std::string key;
      std::string value;
      auto pos = field.find('=');
      if (pos == std::string::npos) {
        key = field;
      } else {
        key = field.substr(0, pos);
        value = field.substr(pos + 1, field.size() - pos - 1);
      }

      PT_TOWL_WARN("Field ", key, " ", value);
      keyvals.emplace_back(key, value);
    };

    auto findFields = [&] {
      std::size_t begin = 0;
      std::size_t end = 0;
      for (; end < config_str.size(); ++end) {
        if (config_str[end] == ':') {
          addField(config_str.substr(begin, end - begin));
          begin = end + 1;
        }
      }
      addField(config_str.substr(begin, end - begin));
    };

    auto interpretFields = [&] {
      for (auto& kv : keyvals) {
        auto& key = kv.first;
        auto& value = kv.second;
        if (key == "log_all") {
          bool flag = value == "1";
          PT_TOWL_WARN("log_all ", flag, value);
          config.log_devmem_buf = flag;
          config.log_devmem_summary = flag;
          config.log_recipe = flag;
          config.log_python = flag;
          config.log_collective = flag;
          config.log_defrag = flag;
          config.log_copy = flag;
          config.log_metrics = flag;
        } else if (key == "log_devmem_buf") {
          config.log_devmem_buf = value == "1";
        } else if (key == "log_devmem_summary") {
          config.log_devmem_summary = value == "1";
        } else if (key == "log_python") {
          config.log_python = value == "1";
        } else if (key == "log_recipe") {
          config.log_recipe = value == "1";
        } else if (key == "log_collective") {
          config.log_collective = value == "1";
        } else if (key == "log_defrag") {
          config.log_defrag = value == "1";
        } else if (key == "log_copy") {
          config.log_copy = value == "1";
        } else if (key == "log_metrics") {
          config.log_metrics = value == "1";
        } else if (key == "rank") {
          if (value == "any") {
            config.any_rank = true;
          } else {
            try {
              config.rank = std::stoi(value);
              config.any_rank = false;
            } catch (...) {
              PT_TOWL_WARN("Invalid value for rank config: ", value);
            }
          }
        } else {
          PT_TOWL_WARN("Unknown config entry: ", key);
        }
      }
    };

    findFields();
    interpretFields();

    return config;
  }

  static Config parseAndApply(const std::string& config_str) {
    // Do not even parse config if towl is disabled
    if (not GET_ENV_FLAG_NEW(PT_TOWL_LOG_ENABLE)) {
      return {};
    }
    auto my_rank = GetRankFromEnv();
    auto config = parse(config_str);
    TowlEnabled::flag = config.any_rank or config.rank == my_rank;

    PT_TOWL_WARN("Enable ", TowlEnabled::flag);
    PT_TOWL_WARN("Config string: ", config_str);
    PT_TOWL_WARN("Config log_devmem_buf=", config.log_devmem_buf);
    PT_TOWL_WARN("Config log_devmem_summary=", config.log_devmem_summary);
    PT_TOWL_WARN("Config log_recipe=", config.log_recipe);
    PT_TOWL_WARN("Config log_python=", config.log_python);
    PT_TOWL_WARN("Config log_collective=", config.log_collective);
    PT_TOWL_WARN("Config log_defrag=", config.log_defrag);
    PT_TOWL_WARN("Config log_copy=", config.log_copy);
    PT_TOWL_WARN("Config log_metrics=", config.log_metrics);
    PT_TOWL_WARN(
        "Config rank=",
        config.rank,
        " (getenv(RANK, default=0)=",
        my_rank,
        ")");
    PT_TOWL_WARN("Config any_rank=", config.any_rank);
    return config;
  }
};

Config config = Config::parseAndApply(GET_ENV_FLAG_NEW(PT_TOWL_LOG_CONFIG));

} // namespace

bool TowlEnabled::flag;
bool g_is_physical = GET_ENV_FLAG_NEW(PT_TOWL_LOG_PHYSICAL);

void emitDeviceMemoryAllocated(
    void* ptr,
    std::size_t size,
    std::uint64_t stream,
    bool is_physical) {
  if (not config.log_devmem_buf)
    return;
  if (is_physical != g_is_physical)
    return;
  PT_TOWL_DEBUG("devmem.malloc ", ptr, " size ", size, " stream ", stream);
}

void emitDeviceMemoryDeallocated(void* ptr, bool is_physical) {
  if (not config.log_devmem_buf)
    return;
  if (is_physical != g_is_physical)
    return;
  PT_TOWL_DEBUG("devmem.free ", ptr);
}

void emitDeviceMemoryAllocSuccess(
    void* ptr,
    std::size_t size,
    bool is_workspace) {
  if (not config.log_devmem_buf)
    return;
  PT_TOWL_DEBUG(
      "devmem.alloc.success ptr ",
      ptr,
      " size ",
      size,
      " workspace ",
      is_workspace);
}

void emitDeviceMemoryAllocFailed(std::size_t size, bool is_workspace) {
  if (not config.log_devmem_buf)
    return;
  PT_TOWL_DEBUG("devmem.alloc.failed size ", size, " workspace ", is_workspace);
}

const char* getTensorTypeName(synTensorType tp) {
#define _N(n) \
  case n:     \
    return #n

  switch (tp) {
    _N(DATA_TENSOR);
    _N(SHAPE_TENSOR);
    _N(DATA_TENSOR_DYNAMIC);
    _N(DEVICE_SHAPE_TENSOR);
    _N(HOST_SHAPE_TENSOR);
    _N(HOST_TO_DEVICE_TENSOR);
    default:
      return "other";
  }
}

void emitRecipeFinished(
    const synapse_helpers::graph::recipe_handle* recipe_handle) {
  if (not config.log_recipe)
    return;

  void* ptr = nullptr;
  if (recipe_handle) {
    ptr = recipe_handle->syn_recipe_handle_;
  }

  PT_TOWL_DEBUG("recipe.finished ", ptr);
}
void emitRecipeLaunch(
    [[maybe_unused]] const synapse_helpers::graph::recipe_handle& recipe_handle,
    [[maybe_unused]] uint64_t workspace_size,
    [[maybe_unused]] const std::vector<std::uint64_t>& addresses,
    [[maybe_unused]] const std::vector<synLaunchTensorInfo>& tensors,
    [[maybe_unused]] bool is_physical) {
  if (not config.log_recipe)
    return;

  if (is_physical != g_is_physical)
    return;

  PT_TOWL_DEBUG(
      "recipe.launch ws ",
      workspace_size,
      " handle ",
      recipe_handle.syn_recipe_handle_,
      " bufs ",
      tensors.size(),
      " name ",
      recipe_handle.recipe_name_);

  for (std::size_t i = 0; i < tensors.size(); ++i) {
    auto& tensor = tensors[i];
    auto addr = addresses[i];
    PT_TOWL_DEBUG(
        "recipe.launch.buf ",
        i,
        " id ",
        tensor.tensorId,
        " type ",
        getTensorTypeName(tensor.tensorType),
        " addr ",
        (void*)tensor.pTensorAddress,
        " orig_addr ",
        (void*)addr,
        " name ",
        (tensor.tensorName ? tensor.tensorName : ""));
  }
}

void emitCollectiveLaunch(const std::string& info) {
  if (not config.log_collective) {
    return;
  }
  PT_TOWL_DEBUG("collective.launch ", info);
}

void emitCollectiveFinished(const std::string& info) {
  if (not config.log_collective) {
    return;
  }
  PT_TOWL_DEBUG("collective.finished ", info);
}

void emitDefragLaunch(const std::string& info) {
  if (not config.log_defrag) {
    return;
  }
  PT_TOWL_DEBUG("defrag.launch ", info);
}

void emitDefragFinished(
    const std::string& info,
    const std::unique_ptr<synapse_helpers::defragment_helpers::Region>&
        region) {
  if (not config.log_defrag) {
    return;
  }

  std::string addr_string;
  if (region && (region->begin_ != region->end_)) {
    addr_string.reserve(64);
    addr_string += " @";
    for (auto it = region->begin_; it != region->end_; ++it) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), " %zx", it->handle_);
      addr_string += buffer;
    }
  }

  PT_TOWL_DEBUG("defrag.finished ", info + addr_string);
}

void emitPythonString(const std::string& s) {
  if (not config.log_python)
    return;
  PT_TOWL_DEBUG("python ", s);
}

void emitDeviceMemorySummary(const char* tag) {
  if (not config.log_devmem_summary)
    return;

  auto& device = habana::HPUDeviceContext::get_device();
  auto& device_memory = device.get_device_memory();
  synapse_helpers::MemoryStats stats;
  device_memory.get_memory_stats(&stats);

  PT_TOWL_DEBUG(
      "devmem.summary used ",
      stats.bytes_in_use,
      " workspace ",
      stats.scratch_mem_in_use,
      " persistent ",
      stats.bytes_in_use - stats.scratch_mem_in_use,
      " tag ",
      tag);
}

void emitCopyLaunch(const char* tag, void* src, void* dst, size_t size) {
  if (not config.log_copy) {
    return;
  }
  PT_TOWL_DEBUG(
      "copy.launch ", tag, " src ", src, " dst ", dst, " size ", size);
}

void emitCopyFinished(const char* tag, void* src, void* dst) {
  if (not config.log_copy) {
    return;
  }
  PT_TOWL_DEBUG("copy.finished ", tag, " src ", src, " dst ", dst);
}

void emitCopyMultipleLaunch(
    const char* tag,
    const uint64_t* srcs,
    const uint64_t* dsts,
    const uint64_t* sizes,
    size_t num_copies) {
  if (not config.log_copy) {
    return;
  }
  PT_TOWL_DEBUG("copy.multiple.launch ", tag, " num_copies ", num_copies);
  for (size_t i = 0; i < num_copies; ++i) {
    PT_TOWL_DEBUG(
        "copy.multiple.launch ",
        tag,
        " src ",
        reinterpret_cast<void*>(srcs[i]),
        " dst ",
        reinterpret_cast<void*>(dsts[i]),
        " size ",
        sizes[i]);
  }
}

void emitCopyMultipleFinished(
    const char* tag,
    std::shared_ptr<synapse_helpers::device_ptr_lock>& locked) {
  if (not config.log_copy) {
    return;
  }
  auto num_copies = std::size_t(std::distance(locked->begin(), locked->end()));
  PT_TOWL_DEBUG("copy.multiple.finished ", tag, " num_copies ", num_copies);
  for (size_t i = 0; i < num_copies; ++i) {
    PT_TOWL_DEBUG(
        "copy.multiple.finished ",
        tag,
        " dst ",
        reinterpret_cast<void*>(locked->at(i)));
  }
}

void emitRecipeCompileSuccess(
    const synapse_helpers::graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    const std::string& name,
    double compile_duration) {
  if (not config.log_recipe_compile)
    return;
  PT_TOWL_DEBUG(
      "recipe.compile.success name ",
      name,
      " handle ",
      fmt::ptr(recipe_handle.syn_recipe_handle_),
      " workspace ",
      workspace_size,
      " compile_time_ms ",
      compile_duration);
}

void emitRecipeCompileFailed(
    const std::string& error_info,
    double compile_duration) {
  if (not config.log_recipe_compile)
    return;
  PT_TOWL_DEBUG(
      "recipe.compile.failed ",
      error_info,
      " compile_time_ms ",
      compile_duration);
}

void emitMetrics(const std::string& name, float value) {
  if (not config.log_metrics)
    return;
  auto msg = name + std::to_string(value);
  PT_TOWL_DEBUG(msg);
}

void emitTimeDurationJit(const std::string& name, float value) {
  if (not config.log_metrics)
    return;
  PT_TOWL_DEBUG("time.duration.jit pass ", name, " took(ms): ", value);
}

void emitTimeDurationFX(const std::string& name, float value) {
  if (not config.log_metrics)
    return;
  PT_TOWL_DEBUG("time.duration.fx pass ", name, " took(ms): ", value);
}

void emitRecipeName(const std::string& param_data) {
  if (not config.log_recipe)
    return;
  PT_TOWL_DEBUG("recipe.name ", param_data);
}

void emitRecipeHandle(synRecipeHandle recipe_handle) {
  if (not config.log_recipe)
    return;
  PT_TOWL_DEBUG("recipe.handle ", recipe_handle);
}

void emitRecipeRequireWorkspace(const std::string& workspace) {
  if (not config.log_recipe)
    return;
  PT_TOWL_DEBUG("recipe.require.workspace ", workspace);
}

void emitRecipeTensorToUse(const std::string& dtensorinfo_dump) {
  if (not config.log_recipe)
    return;
  PT_TOWL_DEBUG("recipe.tensor.touse ", dtensorinfo_dump);
}
} // namespace towl::impl

namespace towl {

void configure(bool enable, std::string config_str) {
  impl::TowlEnabled::flag = enable;
  if (config_str.empty()) {
    config_str = GET_ENV_FLAG_NEW(PT_TOWL_LOG_CONFIG);
  }
  impl::config = impl::Config::parseAndApply(config_str);
}

} // namespace towl
