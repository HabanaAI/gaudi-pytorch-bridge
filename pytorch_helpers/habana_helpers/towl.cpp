/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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
#include "towl.h"
#include <unistd.h>
#include "backend/habana_device/hpu_cached_devices.h"

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
 *  log_devmem=[0|1]    - enables devmem logging category
 *  log_recipe=[0|1]    - enables recipe logging category
 *  log_python=[0|1]    - enables python logging category
 *  rank=int            - logs only under given rank (determined by env RANK)
 *  any_rank=[0|1]      - ignore `rank` option and always log events
 */

struct Config {
  bool log_devmem_buf = true;
  bool log_devmem_summary = true;
  bool log_recipe = true;
  bool log_python = true;
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
        } else if (key == "log_devmem_buf") {
          config.log_devmem_buf = value == "1";
        } else if (key == "log_devmem_summary") {
          config.log_devmem_summary = value == "1";
        } else if (key == "log_python") {
          config.log_python = value == "1";
        } else if (key == "log_recipe") {
          config.log_recipe = value == "1";
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

void emitDeviceMemoryAllocated(
    void* ptr,
    std::size_t size,
    std::uint64_t stream) {
  if (not config.log_devmem_buf)
    return;
  PT_TOWL_DEBUG("devmem.malloc ", ptr, " size ", size, " stream ", stream);
}

void emitDeviceMemoryDeallocated(void* ptr) {
  if (not config.log_devmem_buf)
    return;
  PT_TOWL_DEBUG("devmem.free ", ptr);
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

  void* ptr = 0x0;
  if (recipe_handle) {
    ptr = recipe_handle->syn_recipe_handle_;
  }

  PT_TOWL_DEBUG("recipe.finished ", ptr);
}
void emitRecipeLaunch(
    [[maybe_unused]] const synapse_helpers::graph::recipe_handle& recipe_handle,
    [[maybe_unused]] uint64_t workspace_size,
    [[maybe_unused]] const std::vector<std::uint64_t>& addresses,
    [[maybe_unused]] const std::vector<synLaunchTensorInfo>& tensors) {
  if (not config.log_recipe)
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

void emitPythonString(const std::string& s) {
  if (not config.log_python)
    return;
  PT_TOWL_DEBUG("python ", s);
}

void emitDeviceMemorySummary(const char* tag) {
  if (not config.log_devmem_summary)
    return;

  auto& device = habana::HPURegistrar::get_device();
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