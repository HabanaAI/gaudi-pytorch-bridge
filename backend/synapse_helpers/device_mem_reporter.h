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
#include <absl/strings/str_format.h>
#include <list>
#include "backend/synapse_helpers/device_mem_stats.h"
#include "backend/synapse_helpers/device_types.h"

#define TO_GB(arg) ((arg) / (static_cast<double>(1024 * 1024) * 1024.))
#define TO_REPORT_EVENT(key, value)                                         \
  std::string(" \"") + key + std::string("\":\"") + std::to_string(value) + \
      std::string("\"")
#define TO_REPORT_EVENT_GB(key, value)                                      \
  std::string(" \"") + key + std::string("\":\"") + std::to_string(value) + \
      std::string(" (") +                                                   \
      std::to_string(static_cast<double>(value) / (1024 * 1024 * 1024.)) +  \
      std::string(" GB)\"")
#define TO_GB_STR(value)                                                   \
  std::to_string(value) + std::string(" (") +                              \
      std::to_string(static_cast<double>(value) / (1024 * 1024 * 1024.)) + \
      std::string(" GB)")

namespace synapse_helpers {
struct MemoryConsumption {
  uint64_t total_allocs_bytes; /* Total of bytes allocated. */
  uint64_t max_alloc_bytes; /* The maximum single byte allocated. */
  uint64_t
      pre_allocated_bytes; /* Preallocate bytes allocated for HCL or other. */
  uint64_t workspace_allocated; /* Scratch memory allocated. */
  uint64_t persistent_tensor_size; /* Persistent memory allocated */
  // uint64_t live_tensors_allocs_bytes; /* Live tensors bytes in use. */
  // uint64_t ghost_tensors_allocs_bytes; /* Ghost tensors bytes in use. */

  MemoryConsumption()
      : total_allocs_bytes(0),
        max_alloc_bytes(0),
        pre_allocated_bytes(0),
        workspace_allocated(0),
        persistent_tensor_size(0) {}

  void update(synapse_helpers::MemoryStats& mem_stats) {
    this->total_allocs_bytes = mem_stats.bytes_in_use;
    this->max_alloc_bytes = mem_stats.largest_alloc_size;
    this->pre_allocated_bytes = mem_stats.pre_allocate_size;
    this->workspace_allocated = mem_stats.scratch_mem_in_use;
    this->persistent_tensor_size =
        (mem_stats.bytes_in_use - mem_stats.scratch_mem_in_use -
         this->pre_allocated_bytes);
  }

  std::string DebugString() const {
    return absl::StrFormat(
        "Total Allocate Size:               %20lld (%.6f GB)\n"
        "Maximum Allocate Size:             %20lld (%.6f GB)\n"
        "Pre Allocate Size:                 %20lld (%.6f GB)\n"
        "Scratch Memory Allocated:          %20lld (%.6f GB)\n"
        "Persistent Memory Allocated:       %20lld (%.6f GB)\n",
        this->total_allocs_bytes,
        TO_GB(static_cast<double>(this->total_allocs_bytes)),
        this->max_alloc_bytes,
        TO_GB(static_cast<double>(this->max_alloc_bytes)),
        this->pre_allocated_bytes,
        TO_GB(static_cast<double>(this->pre_allocated_bytes)),
        this->workspace_allocated,
        TO_GB(static_cast<double>(this->workspace_allocated)),
        this->persistent_tensor_size,
        TO_GB(static_cast<double>(this->persistent_tensor_size)));
  };

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") + "MemoryConsumption\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"MemoryConsumption\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT_GB("TotalAllocatedSize", this->total_allocs_bytes) +
        std::string(",") +
        TO_REPORT_EVENT_GB("PreAllocatedSize", this->pre_allocated_bytes) +
        std::string(",") +
        TO_REPORT_EVENT_GB("MaxAllocateSize", this->max_alloc_bytes) +
        std::string(",") +
        TO_REPORT_EVENT_GB("ScratchMemoryAllocateSize",
                           this->workspace_allocated) +
        std::string(",") +
        TO_REPORT_EVENT_GB("PersistentMemoryAllocateSize",
                           this->persistent_tensor_size) +
        std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "MemoryConsumption\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"MemoryConsumption\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};

struct MemoryAllocatorStats {
  uint64_t total_num_allocs; /* Total number of allocations. */
  uint64_t new_num_allocs; /* New number of allocations. */
  uint64_t total_num_frees; /* Total number of frees. */
  uint64_t new_num_frees; /* New number of frees. */

  MemoryAllocatorStats()
      : total_num_allocs(0),
        new_num_allocs(0),
        total_num_frees(0),
        new_num_frees(0) {}

  void update(synapse_helpers::MemoryStats& mem_stats) {
    this->total_num_allocs = mem_stats.total_allocs;
    this->new_num_allocs = mem_stats.num_allocs;
    this->total_num_frees = mem_stats.total_frees;
    this->new_num_frees = mem_stats.num_frees;
  }

  std::string DebugString() const {
    return absl::StrFormat(
        "Total Number of Allocs:            %20lld\n"
        "New Number of Allocs:              %20lld\n"
        "Total Number of Frees:             %20lld\n"
        "New Number of Frees:               %20lld\n",
        this->total_num_allocs,
        this->new_num_allocs,
        this->total_num_frees,
        this->new_num_frees);
  };

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") +
        "MemoryAllocatorStats\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"MemoryAllocatorStats\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT("TotalNumAllocs", this->total_num_allocs) +
        std::string(",") +
        TO_REPORT_EVENT("NewNumAllocs", this->new_num_allocs) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumFrees", this->total_num_frees) +
        std::string(",") + TO_REPORT_EVENT("NewNumFrees", this->new_num_frees) +
        std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "MemoryAllocatorStats\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"MemoryAllocatorStats\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};

struct FragmentationStats {
  uint64_t fragmentation_percent; /* Fragmentation percentage. */
  uint64_t total_num_chunks; /* Total number of chunks. */
  uint64_t total_num_alloc_chunks; /* Total number of alloc chunks. */
  uint64_t total_num_free_chunks; /* Total number of free chunks. */
  uint64_t total_alloc_size; /* Total alloc size. */
  uint64_t total_free_size; /* Total free size. */
  uint64_t max_cntg_chunk_free_size; /* Maximum contiguous chunk free size
                                        available. */
  uint64_t min_chunk_size; /* Minimum chunk size. */
  uint64_t max_chunk_size; /* Maximum Chunk size. */
  std::string fragmentation_histogram; /* Fragmentation histogram. */

  FragmentationStats()
      : fragmentation_percent(0),
        total_num_chunks(0),
        total_num_alloc_chunks(0),
        total_num_free_chunks(0),
        total_alloc_size(0),
        total_free_size(0),
        max_cntg_chunk_free_size(0),
        min_chunk_size(0),
        max_chunk_size(0),
        fragmentation_histogram("") {}

  void update(synapse_helpers::MemoryStats& mem_stats) {
    this->fragmentation_percent = mem_stats.fragmentation_percent;
    this->total_num_chunks = mem_stats.total_chunks;
    this->total_num_alloc_chunks = mem_stats.occupied_chunks;
    this->total_num_free_chunks = mem_stats.free_chunks;
    this->total_alloc_size = mem_stats.occupied_size;
    this->total_free_size = mem_stats.free_chunks_size;
    this->max_cntg_chunk_free_size = mem_stats.max_cntgs_free_chunks_size;
    this->min_chunk_size = mem_stats.min_chunk_size;
    this->max_chunk_size = mem_stats.max_chunk_size;
    this->fragmentation_histogram = mem_stats.fragmentation_mask;
  }

  std::string DebugString() const {
    return absl::StrFormat(
        "Fragmentation Percentage:            %20lld\n"
        "Total Number of Chunks:              %20lld\n"
        "Total Number of Alloc Chunks:        %20lld\n"
        "Total Number of Free Chunks:         %20lld\n"
        "Total Alloc Size:                    %20lld\n"
        "Total Free Size:                     %20lld\n"
        "Max Contiguous Chunk Free Size:      %20lld\n"
        "Minimum Chunk size:                  %20lld\n"
        "Maximum Chunk size:                  %20lld\n"
        "Fragmentation Histogram:             %20s\n",
        this->fragmentation_percent,
        this->total_num_chunks,
        this->total_num_alloc_chunks,
        this->total_num_free_chunks,
        this->total_alloc_size,
        this->total_free_size,
        this->max_cntg_chunk_free_size,
        this->min_chunk_size,
        this->max_chunk_size,
        this->fragmentation_histogram);
  };

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") + "FragmentationStats\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"FragmentationStats\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT("FragmentationPercentage",
                        this->fragmentation_percent) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumChunks", this->total_num_chunks) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumAllocChunks", this->total_num_alloc_chunks) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumFreeChunks", this->total_num_free_chunks) +
        std::string(",") +
        TO_REPORT_EVENT_GB("TotalAllocSize", this->total_alloc_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("TotalFreeSize", this->total_free_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("MaxContiguousFreeSize",
                           this->max_cntg_chunk_free_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("MinChunkSize", this->min_chunk_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("MaxChunkSize", this->max_chunk_size) +
        std::string(",") + std::string(" \"FragmentationHistogram\":\"") +
        this->fragmentation_histogram + std::string("\"") +
        std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "FragmentationStats\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"FragmentationStats\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};

struct GraphStats {
  struct LiveGraphInfo {
    size_t id{0};
    size_t num_inputs{0};
    size_t num_outputs{0};
    size_t persistent_tensor_size{0};
    uint64_t workspace_req_size{0};
    uint64_t workspace_alloc_size{0};

    std::string toJsonEvent(std::string& header_begin, std::string& header_end)
        const {
      std::string event_chunk_begin = header_begin;
      event_chunk_begin += std::string(":\"ID:") + std::to_string(this->id) +
          std::string("\\nInputTensors:") + std::to_string(this->num_inputs) +
          std::string("\\nOutputTensors:") + std::to_string(this->num_outputs) +
          std::string("\\nPersistentTensorSize:") +
          TO_GB_STR(this->persistent_tensor_size) +
          std::string("\\nWorkspaceRequireSize:") +
          TO_GB_STR(this->workspace_req_size) +
          std::string("\\nWorkspaceAllocateSize:") +
          TO_GB_STR(this->workspace_alloc_size) + std::string("\"");
      std::string event_chunk_end = header_end;
      return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
    }
  };
  std::unordered_map<size_t, LiveGraphInfo> live_graph_map;

  GraphStats() {}

  void addGraph(
      size_t graph_key,
      size_t id,
      size_t num_inputs,
      size_t num_outputs,
      size_t persistent_tensor_size,
      uint64_t workspace_req_size,
      uint64_t workspace_alloc_size) {
    if (this->live_graph_map.find(graph_key) == this->live_graph_map.end()) {
      LiveGraphInfo live_graph;
      live_graph.id = id;
      live_graph.num_inputs = num_inputs;
      live_graph.num_outputs = num_outputs;
      live_graph.persistent_tensor_size = persistent_tensor_size;
      live_graph.workspace_req_size = workspace_req_size;
      live_graph.workspace_alloc_size = workspace_alloc_size;
      this->live_graph_map.emplace(graph_key, live_graph);
    }
  }

  void removeLiveGraph(size_t graph_key) {
    if (this->live_graph_map.find(graph_key) != this->live_graph_map.end()) {
      this->live_graph_map.erase(graph_key);
    }
  }

  void updateGraph(
      size_t graph_key,
      size_t persistent_tensor_size,
      uint64_t workspace_alloc_size) {
    if (this->live_graph_map.count(graph_key) > 0) {
      auto live_graph = this->live_graph_map[graph_key];
      live_graph.persistent_tensor_size = persistent_tensor_size;
      live_graph.workspace_alloc_size = workspace_alloc_size;
      this->live_graph_map[graph_key] = live_graph;
    }
  }

  std::string DebugString() const {
    return absl::StrFormat(
        "Total Live Graph Size:                    %20lld\n",
        this->live_graph_map.size());
  };

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") + "GraphStats\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"GraphStats\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT("NumberOfLiveGraph", this->live_graph_map.size());
    int i = 0;
    for (auto graph_item : this->live_graph_map) {
      event_chunk_begin += std::string(",");
      std::string header_begin_ =
          std::string("\"Graph-") + std::to_string(i++) + std::string("\"");
      std::string header_end_ = std::string("");
      event_chunk_begin +=
          graph_item.second.toJsonEvent(header_begin_, header_end_);
    }
    event_chunk_begin += std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "GraphStats\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"GraphStats\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};

struct TensorStats {
  struct TensorInfo {
    int64_t uid{0};
    size_t req_size{0};
    size_t alloc_size{0};
    std::string name{""};

    std::string toJsonEvent(std::string& header_begin, std::string& header_end)
        const {
      std::string event_chunk_begin = header_begin;
      event_chunk_begin += std::string(":\"UniqueId:") +
          std::to_string(this->uid) + std::string("\\nRequestSize:") +
          TO_GB_STR(this->req_size) + std::string("\\nAllocateSize:") +
          TO_GB_STR(this->alloc_size) + std::string("\\nName:") + this->name +
          std::string("\"");
      std::string event_chunk_end = header_end;
      return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
    }
  };

  /* tensor create history map via tensor unique_id */
  std::unordered_map<int64_t, void*> tensor_uid_device_ptr_map;
  /* tensor info history map via tensor address lock */
  std::unordered_map<void*, TensorInfo> device_ptr_tensor_map;

  TensorStats() {}

  void createTensor(int64_t uid) {
    this->tensor_uid_device_ptr_map.emplace(uid, nullptr);
  }

  void setTensorAddressData(int64_t uid, void* tensor_memory) {
    if (this->tensor_uid_device_ptr_map.count(uid) > 0) {
      this->tensor_uid_device_ptr_map[uid] = tensor_memory;
    } else {
      this->tensor_uid_device_ptr_map.emplace(uid, tensor_memory);
    }
    if (tensor_memory != nullptr) {
      if (this->device_ptr_tensor_map.count(tensor_memory) > 0) {
        TensorInfo tensorInfo = this->device_ptr_tensor_map[tensor_memory];
        tensorInfo.uid = uid;
        this->device_ptr_tensor_map[tensor_memory] = tensorInfo;
      } else {
        TensorInfo tensorInfo;
        tensorInfo.uid = uid;
        this->device_ptr_tensor_map.emplace(tensor_memory, tensorInfo);
      }
    }
  }

  void updateTensorAddressData(
      void* tensor_memory,
      std::string name,
      size_t req_size) {
    if (this->device_ptr_tensor_map.count(tensor_memory) > 0) {
      TensorInfo tensorInfo = this->device_ptr_tensor_map[tensor_memory];
      tensorInfo.name = std::string("") + name;
      tensorInfo.req_size = req_size;
      this->device_ptr_tensor_map[tensor_memory] = tensorInfo;
    } else {
      TensorInfo tensorInfo;
      tensorInfo.name = std::string("") + name;
      tensorInfo.req_size = req_size;
      this->device_ptr_tensor_map.emplace(tensor_memory, tensorInfo);
    }
  }

  void lockTensorAddressData(void* tensor_memory, size_t alloc_size) {
    if (this->device_ptr_tensor_map.count(tensor_memory) > 0) {
      TensorInfo tensorInfo = this->device_ptr_tensor_map[tensor_memory];
      tensorInfo.alloc_size = alloc_size;
      this->device_ptr_tensor_map[tensor_memory] = tensorInfo;
    } else {
      TensorInfo tensorInfo;
      tensorInfo.alloc_size = alloc_size;
      this->device_ptr_tensor_map.emplace(tensor_memory, tensorInfo);
    }
  }

  void removeTensor(int64_t uid) {
    if (this->tensor_uid_device_ptr_map.count(uid) > 0) {
      removeTensorByAddress(this->tensor_uid_device_ptr_map[uid]);
      this->tensor_uid_device_ptr_map.erase(uid);
    }
  }

  void removeTensorByAddress(void* tensor_memory) {
    if (this->device_ptr_tensor_map.count(tensor_memory) > 0) {
      this->device_ptr_tensor_map.erase(tensor_memory);
    }
  }

  /*
  Future Tensor   : storage not allocated + use_count() >= 1
  Dead Tensor     : storage not allocated + use_count() < 1 //TODO: Need to
  investigate further Ghost Tensor    : storage allocated + use_count() < 1 Live
  Tensor     : storage allocated + use_count() >= 1
  */

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") + "TensorStats\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"TensorStats\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT("NumberOfLiveTensor",
                        this->device_ptr_tensor_map.size());
    int i = 0;
    size_t live_tensor_size = 0;
    for (auto tensor_item : this->device_ptr_tensor_map) {
      event_chunk_begin += std::string(",");
      std::string header_begin_ =
          std::string("\"Tensor-") + std::to_string(i++) + std::string("\"");
      std::string header_end_ = std::string("");
      event_chunk_begin +=
          tensor_item.second.toJsonEvent(header_begin_, header_end_);
      live_tensor_size += tensor_item.second.alloc_size;
    }
    event_chunk_begin += std::string(", ") +
        TO_REPORT_EVENT_GB("LiveTensorSize", live_tensor_size);
    event_chunk_begin += std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "TensorStats\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"TensorStats\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};

struct MemoryReporter {
  MemoryConsumption mem_consum;
  MemoryAllocatorStats mem_alloc_stats;
  FragmentationStats frag_stats;
  GraphStats graph_stats;
  TensorStats tensor_stats;

  MemoryReporter() {}

  synapse_helpers::MemoryConsumption* getMemoryConsumption() {
    return &mem_consum;
  }

  synapse_helpers::MemoryAllocatorStats* getMemoryAllocatorStats() {
    return &mem_alloc_stats;
  }

  synapse_helpers::FragmentationStats* getFragmentationStats() {
    return &frag_stats;
  }

  synapse_helpers::GraphStats* getGraphStats() {
    return &graph_stats;
  }

  synapse_helpers::TensorStats* getTensorStats() {
    return &tensor_stats;
  }
};
} // namespace synapse_helpers
