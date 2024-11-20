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
#include "cache_version.h"

#include <ATen/ATen.h>
#include <absl/types/span.h>
#include <dlfcn.h>
#include <link.h>
#include "habana_helpers/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

extern char** environ;

std::string get_synapse_lib_path(void) {
  std::string map_file_name{"/proc/" + std::to_string(getpid()) + "/maps"};
  std::ifstream proc_map_stream{map_file_name};
  std::string line;
  while (std::getline(proc_map_stream, line)) {
    size_t name_start{line.rfind('/')};
    size_t path_start{line.find('/')};
    if (name_start == std::string::npos) {
      continue;
    }

    absl::string_view soname{line.c_str() + name_start + 1};
    absl::string_view sopath{line.c_str() + path_start};
    // in docker the name is libSynapse.so.1
    if (soname == "libSynapse.so" || soname == "libSynapse.so.1") {
      return std::string{sopath};
    }
  }
  HABANA_ASSERT(false, "Synapse lib not loaded");
  return "";
}

// quick trick function to retrieve full path to habana_device library
// (ourselves)
std::string habana_device_path(void) {
  Dl_info dl_info;
  dladdr((void*)habana_device_path, &dl_info);
  std::string lib_path = dl_info.dli_fname;
  HABANA_ASSERT(
      lib_path.find("libhabana_pytorch_backend.so") != std::string::npos);
  return lib_path;
}

size_t hash64_file_content(const std::string& path_to_file) {
  int fh = open(path_to_file.c_str(), O_RDONLY);
  if (fh == -1) {
    PT_HABHELPER_WARN("Failed to open file: ", path_to_file);
    return 0;
  }

  size_t hashRes{0};
  if (fh) {
    struct stat sb;
    if (fstat(fh, &sb) == -1) {
      PT_HABHELPER_WARN("Failed to get stat of file: ", path_to_file);
    } else {
      char* fileAddr = (char*)mmap(
          NULL, static_cast<size_t>(sb.st_size), PROT_READ, MAP_PRIVATE, fh, 0);
      if (fileAddr == MAP_FAILED) {
        PT_HABHELPER_WARN("Failed in mapping file: ", path_to_file);
      } else {
        size_t hashRes{0};
        auto range = absl::MakeSpan(fileAddr, fileAddr + sb.st_size);
        for (char c : range) {
          hashRes =
              c10::hash_combine(hashRes, c10::_hash_detail::simple_get_hash(c));
        }
        PT_HABHELPER_DEBUG(
            "Calculated hash for file ",
            path_to_file,
            ", hash: ",
            reinterpret_cast<void*>(hashRes));
      }
      close(fh);
    }
  }
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
  hash = at::hash_combine(hash, hash64_file_content(get_synapse_lib_path()));

  if (IS_ENV_FLAG_DEFINED_NEW(GC_KERNEL_PATH)) {
    std::string gc_kernel_path = GET_ENV_FLAG_NEW(GC_KERNEL_PATH);
    auto foundComma = gc_kernel_path.find(":");
    if (foundComma != std::string::npos) {
      // GC_KERNEL_PATH can be a list of paths to libs, comma separated, need to
      // hash them all
      do {
        using difference_type = decltype(gc_kernel_path)::difference_type;
        std::string path(
            gc_kernel_path.begin(),
            gc_kernel_path.begin() + static_cast<difference_type>(foundComma));
        hash = at::hash_combine(hash, hash64_file_content(path));
        gc_kernel_path = std::string(
            gc_kernel_path.begin() + static_cast<difference_type>(foundComma) +
                1,
            gc_kernel_path.end());
        foundComma = gc_kernel_path.find(",");
      } while (foundComma != std::string::npos);
    }
    // if it's a list do/while gets all the paths but the last one, else it's a
    // single path
    hash = at::hash_combine(hash, hash64_file_content(gc_kernel_path));
  }
  PT_HABHELPER_DEBUG(
      "Combined hash for all important libs: ", reinterpret_cast<void*>(hash));

  char** s = environ;
  for (; *s; s++) {
    std::string env_var(*s);
    if (check_env_fo_hashing(env_var)) {
      PT_HABHELPER_DEBUG("Combining hash for: ", env_var);
      hash = at::hash_combine(hash, c10::hash<std::string>()(env_var));
    }
  }
  PT_HABHELPER_DEBUG(
      "Combined hash for all important libs and envs: ",
      reinterpret_cast<void*>(hash));
  std::stringstream stream;
  stream << std::hex << hash;
  return stream.str();
}
