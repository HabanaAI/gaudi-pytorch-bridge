/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include "cache_version.h"

#include <ATen/ATen.h>
#include <absl/types/span.h>
#include <dlfcn.h>
#include "pytorch_helpers/habana_helpers/logging.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_logger/synapse_logger.h"

#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

extern char** environ;

// quick trick function to retrieve full path to habana_device library
// (ourselves)
std::string habana_device_path(void) {
  Dl_info dl_info;
  dladdr((void*)habana_device_path, &dl_info);
  std::string lib_path = dl_info.dli_fname;
  HABANA_ASSERT(
      lib_path.find("libhabana_pytorch_plugin.so") != std::string::npos);
  return lib_path;
}

size_t hash64_file_content(const std::string& path_to_file) {
  int fh = open(path_to_file.c_str(), O_RDONLY);
  if (fh == -1) {
    PT_HABHELPER_WARN("Failed to open file: ", path_to_file);
    return 0;
  }
  struct stat sb;
  if (fstat(fh, &sb) == -1) {
    PT_HABHELPER_WARN("Failed to get stat of file: ", path_to_file);
    return 0;
  }

  char* fileAddr = (char*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fh, 0);
  if (fileAddr == MAP_FAILED) {
    PT_HABHELPER_WARN("Failed in mapping file: ", path_to_file);
    return 0;
  }
  size_t hashRes{0};
  auto range = absl::MakeSpan(fileAddr, fileAddr + sb.st_size);
  for (char c : range) {
    hashRes = c10::hash_combine(hashRes, c10::_hash_detail::simple_get_hash(c));
  }
  PT_HABHELPER_DEBUG(
      "Calculated hash for file ", path_to_file, ", hash: ", std::hex, hashRes);
  return hashRes;
}

bool check_env_fo_hashing(const std::string& env_var) {
  static std::vector<std::string> hashed_env_vars{
      // TODO: Come up with a list of envs impacting graph building and
      // compilation: SW-62228
      // Below list was compiled with:
      // cat synapse/src/graph_compiler/habana_global_conf.cpp |grep MakePublic
      // -b5 | grep GlobalConf
      // and only entries for Gaudi were considered, not ones for graph/stats
      // dumping
      "ENABLE_RAGGED_SOFTMAX_OPT             ",
      "RAGGED_SOFTMAX_OPT_AMP_VAL            ",
      "TPC_ENGINES_ENABLED_MASK              ",
      "DISABLE_SYNAPSE_QUANTIZATION          ",
      "SYNAPSE_DATA_TYPE_SELECTION           ",
      "PROFILE_PRECISION                     ",
      "PRECISION_TO_RAISE                    ",
      "NUM_OF_LAYERS_TO_RAISE                ",
      "DISABLE_REMOVE_CLIPS                  ",
      "ENABLE_SPARSITY_WEIGHTS               ",
      "ENABLE_STAGED_SUBMISSION              ",
      "INT16_LIMITED_BITS                    ",
      "MME_STRATEGY_ALIGNED_ADDRESSES_ENABLED",
      "ELIMINATE_FIRST_TRANSPOSE             ",
      "ELIMINATE_LAST_TRANSPOSE              ",
      "DISABLE_TENSORS_PINNING               "};
  for (auto const& v : hashed_env_vars) {
    if (env_var.find(v) != std::string::npos)
      return true;
  }
  return false;
}

std::string CacheVersion::libs_env_hash() {
  auto path_to_syn_helpers = habana_device_path();
  size_t hash{0};

  hash = at::hash_combine(hash, hash64_file_content(path_to_syn_helpers));
  hash = at::hash_combine(
      hash, hash64_file_content(synapse_logger::getSynapseLibPath()));

  if (IS_ENV_FLAG_DEFINED(GC_KERNEL_PATH)) {
    std::string gc_kernel_path = GET_ENV_FLAG(GC_KERNEL_PATH);
    auto foundComma = gc_kernel_path.find(":");
    if (foundComma != std::string::npos) {
      // GC_KERNEL_PATH can be a list of paths to libs, comma separated, need to
      // hash them all
      do {
        std::string path(
            gc_kernel_path.begin(), gc_kernel_path.begin() + foundComma);
        hash = at::hash_combine(hash, hash64_file_content(path));
        gc_kernel_path = std::string(
            gc_kernel_path.begin() + foundComma + 1, gc_kernel_path.end());
        foundComma = gc_kernel_path.find(",");
      } while (foundComma != std::string::npos);
    }
    // if it's a list do/while gets all the paths but the last one, else it's a
    // single path
    hash = at::hash_combine(hash, hash64_file_content(gc_kernel_path));
  }
  PT_HABHELPER_TRACE("Combined hash for all important libs: ", std::hex, hash);

  char** s = environ;
  for (; *s; s++) {
    std::string env_var(*s);
    if (check_env_fo_hashing(env_var)) {
      PT_HABHELPER_TRACE("Combining hash for: ", env_var);
      hash = at::hash_combine(hash, c10::hash<std::string>()(env_var));
    }
  }
  PT_HABHELPER_TRACE(
      "Combined hash for all important libs and envs: ", std::hex, hash);
  std::stringstream stream;
  stream << std::hex << hash;
  return stream.str();
}
