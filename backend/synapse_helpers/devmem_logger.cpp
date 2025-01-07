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
#include <dlfcn.h>
#include <cinttypes>
#include <cstdlib>
#include <cstring>

#include <execinfo.h>
#include <unistd.h>
#include <cassert>

#include <cxxabi.h>
#include <cstdlib>
#include <memory>
#include <sstream>

#include <absl/strings/str_format.h>
#include "backend/synapse_helpers/device_mem_stats.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/util.h"
#include "devmem_logger.h"

namespace synapse_helpers {

deviceMallocData::deviceMallocData() {
  iteration_number = 0;
  running_memory = iteration_high_watermark = overall_high_watermark = 0;
  bt_depth = 40;
  std::string node_id = std::getenv("RANK") ? std::getenv("RANK") : "0";
  filename = absl::StrFormat(
      "%s_%s", GET_ENV_FLAG_NEW(PT_HABANA_MEM_LOG_FILENAME), node_id);
  memory_reporter_name = absl::StrFormat(
      "memory.reporter_%s.json", node_id);

  auto log_level = (mem_log_level)GET_ENV_FLAG_NEW(PT_HABANA_MEM_LOG_LEVEL);
  print_free_bt = false;
  print_alloc_bt = false;
  take_bt = false;
  enable_recording = false;
  print_memory_stats = false;
  logging_enabled_ = true;
  mem_reporter_enable_ = false;
  switch (log_level) {
    case MEM_LOG_ALL:
      print_free_bt = true;
      print_alloc_bt = true;
      take_bt = true;
      enable_recording = true;
      break;
    case MEM_LOG_ALLOC:
      print_alloc_bt = true;
      take_bt = true;
      break;
    case MEM_LOG_FREE:
      print_free_bt = true;
      take_bt = true;
      break;
    case MEM_LOG_ALLOC_FREE_NOBT:
      print_alloc_bt = true;
      print_free_bt = true;
      break;
    case MEM_LOG_MEMORY_STATS:
      enable_recording = true;
      print_memory_stats = true;
      break;
    case MEM_LOG_RECORD:
      enable_recording = true;
      break;
    case MEM_REPORTER:
      mem_reporter_enable_ = true;
      break;
    case MEM_LOG_DISABLE:
    default:
      logging_enabled_ = false;
      break;
  }
  dram_start_ = dram_size_ = 0;

  logging_enabled_ = (take_bt || print_free_bt || print_alloc_bt);
  if (logging_enabled_ || enable_recording || print_memory_stats) {
    out.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
  }

  if (mem_reporter_enable_) {
    SET_ENV_FLAG_NEW(PT_HPU_POOL_LOG_FRAGMENTATION_INFO, true, 1);
    // TODO: support .txt and .json, default .json support
    memory_reporter_out.open(
        memory_reporter_name.c_str() , std::ofstream::out | std::ofstream::trunc);
    memory_reporter_out << "[\n";
  }
  fragment_json_enabled_ = GET_ENV_FLAG_NEW(PT_HPU_POOL_MEM_FRAGMENT_JSON);
  if (fragment_json_enabled_) {
    memory_json_out.open(
        "memory.log.json", std::ofstream::out | std::ofstream::trunc);
    memory_json_out << "[\n";
  }
}

deviceMallocData::~deviceMallocData() {
  if (out.is_open()) {
    out.close();
  }
  if (memory_reporter_out.is_open()) {
    UNSET_ENV_FLAG_NEW(PT_HPU_POOL_LOG_FRAGMENTATION_INFO);
    memory_reporter_out << "]\n";
    memory_reporter_out.close();
  }
  if (memory_json_out.is_open()) {
    memory_json_out.close();
  }
}

deviceMallocData& deviceMallocData::singleton() {
  static deviceMallocData instance;
  return instance;
}

bool deviceMallocData::sort_by_size(
    std::pair<uint64_t, size_bt_pair_t>& a,
    std::pair<uint64_t, size_bt_pair_t>& b) {
  return a.second.first > b.second.first;
}

bool deviceMallocData::sort_by_ptr(
    std::pair<uint64_t, size_bt_pair_t>& a,
    std::pair<uint64_t, size_bt_pair_t>& b) {
  return a.first < b.first;
}
/*
 * Check if the stack frame contain functions/modules that
 * we would like to see.
 */
bool deviceMallocData::interesting_function(const std::string& name) {
  std::vector<std::string> list_of_interest = {
      "hpu",
      "HPU",
      "at::native::",
      "habana",
      "Habana",
      "HABANA",
      "hb_torch",
      "AllocateAndAddSynapseNode",
  };

  bool interesting = false;
  for (const auto& name_entry : list_of_interest) {
    if (name.find(name_entry) != std::string::npos) {
      interesting = true;
      break;
    }
  }

  return interesting;
}

std::string deviceMallocData::get_formatted_func_name(
    std::string string,
    bool print_all_frames,
    bool* dot_marker_placed) {
  const std::string dot_dot_dot = "...";

  // Find the mangled function name in the frame
  const auto start_of_func_name = string.find('(');
  std::string out_name = "";
  std::size_t end_of_func_name;
  bool formatted_name = true;
  if (start_of_func_name != std::string::npos) {
    end_of_func_name = string.find('+', start_of_func_name);
    if ((end_of_func_name == std::string::npos) ||
        (end_of_func_name == start_of_func_name + 1)) {
      formatted_name = false;
    }
  } else {
    formatted_name = false;
  }
  if (formatted_name) {
    // Print demangled name
    const auto len = end_of_func_name - start_of_func_name - 1;
    int status;
    const auto& name = string.substr(start_of_func_name + 1, len);
    const auto demangled_name =
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
    if (!print_all_frames &&
        !interesting_function((status == 0) ? demangled_name : name)) {
      // If the function isn't of interest, don't print the frame
      if (!*dot_marker_placed) {
        out_name += ("    " + dot_dot_dot + "\n");
        *dot_marker_placed = true;
      }
      return out_name;
    }
    *dot_marker_placed = false;
    if (status == 0) {
      std::string demang_name(demangled_name);
      demang_name = demang_name.substr(0, demang_name.find("("));
      out_name += ("    " + demang_name + "\n");
    } else {
      std::string name_(name);
      name_ = name_.substr(0, name_.find("("));
      out_name += ("    " + name_ + "\n");
    }
  } else {
    if (!print_all_frames && !interesting_function(string)) {
      // If the function isn't of interest, don't print the frame
      if (!*dot_marker_placed) {
        out_name += ("    " + dot_dot_dot + "\n");
        *dot_marker_placed = true;
      }
      return out_name;
    }
    *dot_marker_placed = false;
    out_name += ("    " + string + "\n");
  }
  return out_name;
}

/*
 * Print an entry from the log
 */
void deviceMallocData::print_an_entry(
    Lockedfstream& out_stream,
    const std::pair<uint64_t, size_bt_pair_t>& entry,
    bool print_all_frames) {
  // Print the data pointer
  out_stream << "ptr = 0x" << std::hex << entry.first;

  // Print the allocated size
  const auto& size_bt = entry.second;
  out_stream << ", size = " << std::dec << size_bt.first << "\n";

  // Print the backtrace
  if (take_bt) {
    const auto& bt_strings = size_bt.second;
    bool dot_marker_placed = false;
    for (const auto& string : bt_strings) {
      if (string.length()) {
        out_stream << get_formatted_func_name(
            string, print_all_frames, &dot_marker_placed);
      }
    }
  }
}

/*
 * Gather backtrace for a synDeviceMalloc/synDeviceFree
 */
void deviceMallocData::collect_backtrace(
    uint64_t ptr,
    bool alloc,
    size_t size,
    bool failure) {
  if (!logging_enabled_)
    return;

  int nptrs;
  std::vector<void*> vbuf;
  vbuf.reserve(bt_depth);
  void** buffer = vbuf.data();
  char** strings = nullptr;
  bool duplicate = false;
  std::vector<std::string> bt_string;
  // TODO: keep a maximum limit for the backtrace buffer to avoid host memory
  // exhaustion.

  // Take backtrace
  if (take_bt && (alloc || print_free_bt)) {
    nptrs = backtrace(buffer, bt_depth);

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == nullptr) {
      perror("backtrace_symbols");
      exit(EXIT_FAILURE);
    }

    for (int i = 2; i < nptrs; i++) {
      bt_string.emplace_back(strings[i]);
    }

    free(strings);
  }

  auto out_stream = get_out_stream();
  if (failure) {
    std::pair<uint64_t, size_bt_pair_t> entry =
        std::make_pair(ptr, std::make_pair(size, bt_string));

    out_stream << "=========================\n";
    if (alloc) {
      out_stream << "Allocation failed from\n";
    } else {
      out_stream << "Free failed from\n";
    }
    print_an_entry(out_stream, entry, true);
    out_stream << "=========================\n";
  } else {
    // synDeviceMalloc
    if (alloc) {
      auto existing_it = ptr_bt_map.find(ptr);
      if (existing_it != ptr_bt_map.end()) {
        std::pair<uint64_t, size_bt_pair_t> existing_entry = std::make_pair(
            existing_it->first,
            std::make_pair(
                existing_it->second.first, existing_it->second.second));

        duplicate = true;
        duplicate_ptr_bt_map[ptr] = std::make_pair(size, bt_string);
        out_stream << "=========================\n";
        out_stream << "Duplicate alloc detected - was a free missed?\n";
        out_stream
            << "NOTE: if allocations and free happen from multiple threads, then\n";
        out_stream
            << "      it is possible to have a scenario when the free followed by an\n";
        out_stream
            << "      allocation is seen by the logger in reverse order and the\n";
        out_stream
            << "      freed ptr is returned by alloc. It needs to be checked if a free\n";
        out_stream
            << "      follows this message with the same ptr, which then is most likely\n";
        out_stream
            << "      due to the logger seeing the free followed by alloc in revsere \n";
        out_stream << "      order as alloc followed by free.\n";
        out_stream << "Existing record for the allocated ptr\n";
        print_an_entry(out_stream, existing_entry, true);
        out_stream << "=========================\n";
        out_stream << "Now allocating from \n";

        std::pair<uint64_t, size_bt_pair_t> new_entry =
            std::make_pair(ptr, std::make_pair(size, bt_string));
        print_an_entry(out_stream, new_entry, true);
        out_stream << "=========================\n";
      }
      // Log the entry :
      //   ptr -> (size, backtrace)
      ptr_bt_map[ptr] = std::make_pair(size, bt_string);
      auto it = ptr_bt_map.find(ptr);
      // If we want a backtrace on each alloc (very verbose!)
      if (print_alloc_bt) {
        std::pair<uint64_t, size_bt_pair_t> entry = std::make_pair(
            it->first, std::make_pair(it->second.first, bt_string));

        out_stream << "=========================\n";
        out_stream << "Alloc record entry\n";
        print_an_entry(out_stream, entry);
        out_stream << "=========================\n";
      }

      // Stats update
      if (!duplicate) {
        running_memory += size;
        if (running_memory > iteration_high_watermark) {
          iteration_high_watermark = running_memory;
        }
      }

      if (iteration_high_watermark > overall_high_watermark) {
        overall_high_watermark = iteration_high_watermark;
        std::pair<uint64_t, size_bt_pair_t> entry =
            std::make_pair(ptr, std::make_pair(size, bt_string));

        out_stream << "=========================\n";
        out_stream << "Reached high watermark " << overall_high_watermark
                   << " from\n";
        print_an_entry(out_stream, entry, false);
        out_stream << "=========================\n";
      }
    } else {
      // synDeviceFree
      auto it = ptr_bt_map.find(ptr);
      if (it == ptr_bt_map.end()) {
        auto duplicate_it = duplicate_ptr_bt_map.find(ptr);
        if (duplicate_it == duplicate_ptr_bt_map.end()) {
          out_stream << "=========================\n";
          out_stream << "Unknwon pointer 0x" << std::hex << ptr << std::dec
                     << " free detected, not even in duplicates\n";
          out_stream << "=========================\n";
          out_stream << "Now Freeing from \n";

          std::pair<uint64_t, size_bt_pair_t> new_entry =
              std::make_pair(ptr, std::make_pair(0, bt_string));
          print_an_entry(out_stream, new_entry, true);
          out_stream << "=========================\n";
        } else {
          out_stream << "=========================\n";
          out_stream << "Duplicate pointer 0x" << std::hex << ptr << std::dec
                     << " free detected\n";
          running_memory -= duplicate_it->second.first;
          duplicate_ptr_bt_map.erase(duplicate_it);
        }
        // Log the entry :
      } else {
        // Update stat
        running_memory -= it->second.first;

        // If we want a backtrace on each free (very verbose!)
        if (print_free_bt) {
          std::pair<uint64_t, size_bt_pair_t> entry = std::make_pair(
              it->first, std::make_pair(it->second.first, bt_string));

          out_stream << "=========================\n";
          out_stream << "Free record entry\n";
          print_an_entry(out_stream, entry, true);
          out_stream << "Free entry was allocated from\n";
          entry = std::make_pair(
              it->first, std::make_pair(it->second.first, it->second.second));
          print_an_entry(out_stream, entry);
          out_stream << "=========================\n";
        }

        // Remove the entry from live allocations list
        ptr_bt_map.erase(it);
      }
    }
  }
}

void deviceMallocData::report_fragmentation(bool from_free) {
  if (!logging_enabled_)
    return;

  print_live_allocations(from_free ? "Free failure" : "Allocation failure");

  auto out_stream = get_out_stream();

  out_stream << "Fragmentation report\n";

  if ((dram_size_ == 0) || (dram_start_ == 0)) {
    out_stream << " No record of DRAM start or size!\n";
  } else {
    out_stream << "dram start" << dram_start_ << "\n";
    out_stream << "dram size" << dram_size_ << "\n";
    uint64_t current_head = dram_start_;

    std::vector<std::pair<uint64_t, size_bt_pair_t>> sorted_by_ptr_log(
        ptr_bt_map.begin(), ptr_bt_map.end());

    // Sort the entries by ptr address
    std::sort(sorted_by_ptr_log.begin(), sorted_by_ptr_log.end(), sort_by_ptr);

    std::vector<std::pair<uint64_t, uint64_t>> free_list;

    for (const auto& entry : sorted_by_ptr_log) {
      const auto& entry_addr = entry.first;
      if (current_head < entry_addr) {
        free_list.emplace_back(
            std::make_pair(current_head, entry_addr - current_head));
      } else {
        assert(current_head == entry_addr);
      }
      current_head = entry_addr + entry.second.first;
    }
    if (current_head >= dram_start_ + dram_size_) {
      out_stream << "WARNING: DRAM size data is probably wrong. "
                    "Last allocation exceeds DRAM size\n";
    } else if (current_head != dram_start_ + dram_size_) {
      free_list.emplace_back(std::make_pair(
          current_head, dram_start_ + dram_size_ - current_head));
    }

    out_stream << "Free List\n";
    for (const auto& entry : free_list) {
      out_stream << "0x" << std::hex << entry.first << ": " << std::dec
                 << entry.second << "\n";
    }

    std::ofstream csv_out;
    csv_out.open(fragment_csv_file, std::ofstream::out | std::ofstream::trunc);
    // Redirect output to logfile
    auto ptr_start = dram_start_;
    size_t running_size = 0;
    for (const auto& entry : free_list) {
      // ptr_start to free list entry is occupied
      if (entry.first > ptr_start) {
        auto occupied_size = entry.first - ptr_start - 1;
        // Mark occupied range with 1
        csv_out << running_size << ", 1\n" << std::flush;
        csv_out << running_size + occupied_size << ", 1\n" << std::flush;
        ptr_start = entry.first;
        running_size += occupied_size + 1;
      }
      assert(ptr_start == entry.first);
      // Mark free range with 0
      csv_out << running_size << ", 0\n" << std::flush;
      csv_out << running_size + entry.second << ", 0\n" << std::flush;
      running_size += entry.second + 1;
    }
    if (ptr_start < dram_start_ + dram_size_) {
      // Mark if the last part is occupied
      csv_out << dram_start_ + dram_size_ - ptr_start << ", 1\n" << std::flush;
    }
    csv_out.close();
  }
}

void deviceMallocData::print_to_file(const char* msg) {
  auto stream = get_out_stream();
  stream << msg << "\n";
}

/*
 * Print live allocation details at the given point.
 */
void deviceMallocData::print_live_allocations(const char* msg) {
  if (!logging_enabled_) {
    return print_to_file(msg);
  }

  std::string record_id_msg = msg;
  if (0 == record_id_msg.size()) {
    record_id_msg = "Instance " + std::to_string(iteration_number);
  }
  auto out_stream = get_out_stream();
  out_stream << "\n=========================\n";
  out_stream << "LIVE ALLOCATIONS DATA " << record_id_msg << "\n";
  out_stream << "=========================\n";
  out_stream << "DRAM start: 0x" << std::hex << dram_start_ << "\n";
  out_stream << "DRAM size: " << std::dec << dram_size_ << " ("
             << dram_size_ / (1024 * 1024 * 1024.) << " GB)\n";
  std::vector<std::pair<uint64_t, size_bt_pair_t>> sorted_by_size_log(
      ptr_bt_map.begin(), ptr_bt_map.end());

  // Sort the entries by size
  std::sort(sorted_by_size_log.begin(), sorted_by_size_log.end(), sort_by_size);

  // How many allocations are not freed yet?
  out_stream << "#Allocations live : " << sorted_by_size_log.size() << "\n";

  // How much memory is held by our live allocations now?
  size_t total_live_size = 0;
  for (const auto& entry : sorted_by_size_log) {
    total_live_size += entry.second.first;
  }
  out_stream << "Total memory held : " << total_live_size << " ("
             << total_live_size / (1024 * 1024.) << " MB)\n";

  // Stats on peak memory usage
  out_stream << "Peak memory usage : " << overall_high_watermark << " ("
             << overall_high_watermark / (1024 * 1024.) << " MB)\n";

  out_stream << "Peak memory usage from last log : " << iteration_high_watermark
             << " (" << iteration_high_watermark / (1024 * 1024.) << " MB)\n";

  ++iteration_number;
  iteration_high_watermark = 0;

  // Some allocations persist, what are the new live allocations from last
  // report?
  uint64_t new_allocs = 0;
  if (!ptr_bt_map_last.empty()) {
    for (const auto& entry : sorted_by_size_log) {
      if (ptr_bt_map_last.find(entry.first) == ptr_bt_map_last.end()) {
        ++new_allocs;
      }
    }
  }

  out_stream << "New allocations since last log " << record_id_msg << " : "
             << new_allocs << "\n";

  out_stream << "New allocations since last log\n";
  // Find entries that ae new for this log and print them
  if (!ptr_bt_map_last.empty()) {
    for (const auto& entry : sorted_by_size_log) {
      if (ptr_bt_map_last.find(entry.first) == ptr_bt_map_last.end()) {
        print_an_entry(out_stream, entry);
      }
    }
  }

  if (!mem_statuscheck_running) {
    // Print all entries that are live at this point
    out_stream << "All live allocations\n";
    int cnt = 0;
    for (const auto& entry : sorted_by_size_log) {
      out_stream << "Entry : " << ++cnt << " ";
      print_an_entry(out_stream, entry);
    }
  }
  ptr_bt_map_last = ptr_bt_map;
}

void deviceMallocData::record_graph_tensor_info(
    const std::string& name,
    const bool is_graph_input,
    const bool is_graph_output,
    const uint64_t index,
    uint64_t size) {
  if (is_graph_input) {
    graph_input_indices.insert({index, graph_input.size()});
    graph_input.emplace_back(std::make_tuple(index, size, name));
  } else if (is_graph_output) {
    graph_output_indices.insert({index, graph_output.size()});
    graph_output.emplace_back(std::make_tuple(index, size, name));
  }
}

void deviceMallocData::update_graph_tensor_info(
    const uint64_t index,
    uint64_t start) {
  auto is_input = graph_input_indices.find(index) != graph_input_indices.end();

  auto update_graph_entry =
      [&](const uint64_t index,
          const uint64_t list_index,
          std::vector<std::tuple<uint64_t, uint64_t, std::string>>&
              graph_tensors) {
        assert(list_index < graph_tensors.size());
        auto& entry = graph_tensors.at(list_index);
        if (index != std::get<0>(entry)) {
          return;
        }
        auto size = std::get<1>(entry);
        auto& name = std::get<2>(entry);
        graph_tensors[list_index] = std::make_tuple(start, start + size, name);
        // std::cout << "graph entry: " << name << " start = " << start << "end
        // = " << start + size << "\n";
      };

  if (is_input) {
    auto input_entry = graph_input_indices.find(index);
    update_graph_entry(input_entry->first, input_entry->second, graph_input);
  } else {
    auto output_entry = graph_output_indices.find(index);
    assert(output_entry != graph_output_indices.end());
    update_graph_entry(output_entry->first, output_entry->second, graph_output);
  }
}

void deviceMallocData::record_tensor_info(
    const std::string& name,
    const bool is_param,
    const bool is_grad,
    const bool is_optim_state,
    const bool is_graph_input,
    const bool is_graph_output,
    uint64_t start,
    uint64_t end) {
  if (is_param) {
    params.emplace_back(std::make_tuple(start, end, name));
  } else if (is_grad) {
    grads.emplace_back(std::make_tuple(start, end, name));
  } else if (is_optim_state) {
    optim_states.emplace_back(std::make_tuple(start, end, name));
  } else if (is_graph_input) {
    // Not adding via this.. graph_input.emplace_back(std::make_tuple(start,
    // end, name));
  } else if (is_graph_output) {
    // Not adding via this.. graph_output.emplace_back(std::make_tuple(start,
    // end, name));
  }
}

void deviceMallocData::create_fragment_json_entry(
    synapse_helpers::device& device,
    std::string& graph_name) {
  auto occupied_chunks_map =
      device.get_device_memory().get_occupied_chunk_map();
  static int stat_idx;

  std::unordered_set<std::string> json_params_printed;

  std::string frag_line_header = std::string("{ \"tid\":") +
      std::to_string(stat_idx++) + std::string(", \"pid\":") +
      std::to_string(getpid()) + std::string(", ");

  bool first_chunk_reported = false;

  auto json_out_stream = get_memory_json_out_stream();

  for (auto& chunk : occupied_chunks_map) {
    std::string frag_chunk_begin = frag_line_header;
    if (!first_chunk_reported) {
      frag_chunk_begin += std::string("\"ts\":") + std::to_string(0);
      frag_chunk_begin += std::string(", \"name\":\"") + graph_name +
          std::string("\", \"ph\":\"B\", \"func\":\"Graph") +
          std::string("\", \"args\":{\"graph name\":\"") + graph_name +
          std::string("\"}}\n");
      std::string frag_chunk_end = frag_line_header;
      frag_chunk_end +=
          std::string("\"ts\":") + std::to_string((chunk.first) / (1024));
      frag_chunk_end += std::string(", \"name\":\"") + graph_name +
          std::string("\", \"ph\":\"E\", \"func\":\"Graph") +
          std::string("\", \"args\":{\"graph name\":\"") + graph_name +
          std::string("\"}}\n");

      json_out_stream << frag_chunk_begin;
      json_out_stream << frag_chunk_end;
      first_chunk_reported = true;
    }

    frag_chunk_begin = frag_line_header;
    frag_chunk_begin +=
        std::string("\"ts\":") + std::to_string(chunk.first / (1024));
    frag_chunk_begin +=
        std::string(
            ", \"name\":\"Persistent\", \"ph\":\"B\", \"func\":\"Persistent\", \"args\":{\"Size_in_bytes\":\"") +
        std::to_string(chunk.second / (1024)) + std::string(" KB\"}}\n");
    std::string frag_chunk_end = frag_line_header;
    frag_chunk_end += std::string("\"ts\":") +
        std::to_string((chunk.first + chunk.second) / (1024));
    frag_chunk_end +=
        std::string(
            ", \"name\":\"Persistent\", \"ph\":\"E\", \"func\":\"Persistent\", \"args\":{\"Size_in_bytes\":\"") +
        std::to_string(chunk.second / (1024)) + std::string(" KB\"}}\n");

    json_out_stream << frag_chunk_begin;
    json_out_stream << frag_chunk_end;

    auto write_model_tensors =
        [&](std::vector<std::tuple<uint64_t, uint64_t, std::string>>
                model_tensors,
            const std::string& type) {
          for (auto p : model_tensors) {
            uint64_t start = std::get<0>(p);
            uint64_t end = std::get<1>(p);
            std::string name = std::get<2>(p);

            if ((json_params_printed.count(name) == 0) &&
                (start >= chunk.first) &&
                (end <= (chunk.first + chunk.second))) {
              std::string param_begin = frag_line_header;
              param_begin +=
                  std::string("\"ts\":") + std::to_string(start / (1024));
              param_begin += std::string(", \"name\":\"") + std::string(type) +
                  std::string("\", \"ph\":\"B\", \"func\":\"") +
                  std::string("\", \"args\":{\"Name\":\"") + std::string(name) +
                  std::string("\", \"Size_in_bytes\":\"") +
                  std::to_string(chunk.second / (1024)) +
                  std::string(" KB\"}}\n");
              std::string param_end = frag_line_header;
              param_end +=
                  std::string("\"ts\":") + std::to_string(end / (1024));
              param_end += std::string(", \"name\":\"") + std::string(type) +
                  std::string("\", \"ph\":\"E\", \"func\":\"") +
                  std::string("\", \"args\":{\"Name\":\"") + std::string(name) +
                  std::string("\", \"Size_in_bytes\":\"") +
                  std::to_string(chunk.second / (1024)) +
                  std::string(" KB\"}}\n");

              json_out_stream << param_begin;
              json_out_stream << param_end;
              json_params_printed.insert(name);
            }
          }
        };

    write_model_tensors(graph_input, "Graph Input");
    write_model_tensors(graph_output, "Graph Output");
    write_model_tensors(params, "Weight");
    write_model_tensors(grads, "Grad");
    write_model_tensors(optim_states, "Optimizer State");

    if ((workspace_start >= chunk.first) &&
        (workspace_end <= (chunk.first + chunk.second))) {
      std::string workspace_chunk_begin = frag_line_header;
      workspace_chunk_begin +=
          std::string("\"ts\":") + std::to_string(workspace_start / (1024));
      workspace_chunk_begin +=
          std::string(
              ", \"name\":\"Workspace\", \"ph\":\"B\", \"func\":\"Workspace\", \"args\":{\"Name\":\"") +
          std::string(graph_name) + std::string("\", \"Size_in_bytes\":\"") +
          std::to_string((workspace_end - workspace_start) / (1024)) +
          std::string(" KB\"}}\n");
      std::string workspace_chunk_end = frag_line_header;
      workspace_chunk_end +=
          std::string("\"ts\":") + std::to_string(workspace_end / (1024));
      workspace_chunk_end +=
          std::string(
              ", \"name\":\"Workspace\", \"ph\":\"E\", \"func\":\"Workspace\", \"args\":{\"Name\":\"") +
          std::string(graph_name) + std::string("\", \"Size_in_bytes\":\"") +
          std::to_string((workspace_end - workspace_start) / (1024)) +
          std::string(" KB\"}}\n");

      json_out_stream << workspace_chunk_begin;
      json_out_stream << workspace_chunk_end;
    }
  }

  graph_input.clear();
  graph_output.clear();

  graph_input_indices.clear();
  graph_output_indices.clear();
}

/**
 * log_synDeviceRecordTensorInfo
 */
void log_synDeviceRecordGraphTensorInfo(
    const std::string& name,
    const bool is_graph_input,
    const bool is_graph_output,
    const uint64_t index,
    uint64_t size) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_fragment_json_enabled()) {
    dmd.record_graph_tensor_info(
        name, is_graph_input, is_graph_output, index, size);
  }
}

void log_synDeviceUpdateGraphTensorInfo(const uint64_t index, uint64_t start) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_fragment_json_enabled()) {
    dmd.update_graph_tensor_info(index, start);
  }
}

/**
 * log_synDeviceRecordTensorInfo
 */
void log_synDeviceRecordTensorInfo(
    const std::string& name,
    const bool is_param,
    const bool is_grad,
    const bool is_optim_state,
    const bool is_graph_input,
    const bool is_graph_output,
    uint64_t start,
    uint64_t end) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_fragment_json_enabled()) {
    dmd.record_tensor_info(
        name,
        is_param,
        is_grad,
        is_optim_state,
        is_graph_input,
        is_graph_output,
        start,
        end);
  }
}
/*
 * log synDeviceMalloc
 */
void log_synDeviceMalloc(uint64_t ptr, size_t size, bool failed) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_logging_enabled()) {
    dmd.collect_backtrace(ptr, true, size, failed);
    if (failed) {
      dmd.report_fragmentation();
    }
  }
  if (dmd.is_recording_enabled()) {
    dmd.record("MALLOC", size, uint64_to_hex_string(ptr));
  }
}

/*
 * log synDeviceFree
 */
void log_synDeviceFree(uint64_t ptr, bool failed) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_logging_enabled()) {
    dmd.collect_backtrace(ptr, false, 0, failed);
    if (failed) {
      dmd.report_fragmentation(true);
    }
  }
  if (dmd.is_recording_enabled()) {
    dmd.record("FREE", uint64_to_hex_string(ptr));
  }
}

/*
 * log creating memory pool
 */
void log_synDevicePoolCreate(
    uint64_t free_mem,
    uint64_t mem_acquire_perc,
    uint64_t base_mem_ptr) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    dmd.record(
        "CREATE_POOL",
        free_mem,
        mem_acquire_perc,
        uint64_to_hex_string(base_mem_ptr));
  }
}

/*
 * log workspace memory
 */
void log_synDeviceWorkspace(uint64_t ptr, size_t size) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    dmd.record("WORKSPACE", size, uint64_to_hex_string(ptr));
  }
  if (dmd.is_fragment_json_enabled()) {
    dmd.update_workspace_record(ptr, ptr + size);
  }
}

/*
 * log Alloc device memory
 */
void log_synDeviceAlloc(uint64_t ptr, size_t size) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    dmd.record("ALLOCATE", size, uint64_to_hex_string(ptr));
  }
}

/*
 * log Deallocate device memory
 */
void log_synDeviceDeallocate(uint64_t ptr) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    dmd.record("DEALLOCATE", uint64_to_hex_string(ptr));
  }
}

/*
 * log Mem Stats
 */

void log_synDeviceMemStats(device_memory& dev_mem) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_mem_stats_log_enabled()) {
    synapse_helpers::MemoryStats stats;
    dev_mem.get_memory_stats(&stats);
    std::string updated_msg = "Memory stats \n" + stats.DebugString();
    synapse_helpers::print_live_allocations(updated_msg.c_str());
  }
}

/*
 * log lock memory
 */
void log_synDeviceLockMemory(
    absl::Span<const synapse_helpers::device_ptr> ptrs) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    dmd.record("LOCK", ptrs);
  }
}

/*
 * log graph info - name, total memory and size
 */
void log_graph_info(
    synapse_helpers::device& device,
    std::string graph_name,
    size_t size,
    size_t wsize,
    size_t used_ws_size) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    std::stringstream msg;
    msg << "GRAPH " << graph_name << " total Memory::" << size
        << " Required WS::" << wsize << " Used Size" << used_ws_size;
    synapse_helpers::print_live_allocations(msg.str().c_str());
  }
  if (dmd.is_fragment_json_enabled()) {
    dmd.create_fragment_json_entry(device, graph_name);
  }
}

/*
 * log tensor info - tensor name, virtual addr  and device_addr
 */
void log_tensor_info(
    std::string tensor_name,
    uint64_t index,
    uint64_t v_addr,
    uint64_t d_addr) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    std::stringstream msg;
    msg << "Tensor Name " << tensor_name << std::hex << " virtual addr::0x"
        << v_addr << " device_addr::0x" << d_addr;
    synapse_helpers::print_live_allocations(msg.str().c_str());
  }
  if (dmd.is_fragment_json_enabled()) {
    dmd.update_graph_tensor_info(index, v_addr);
  }
}

/*
 * log allocation failure stats
 */
void log_synDeviceAllocFail(
    synapse_helpers::device& device,
    bool is_workspace,
    size_t size) {
  auto& dmd = deviceMallocData::singleton();
  if (dmd.is_recording_enabled()) {
    synapse_helpers::MemoryStats stats;
    device.get_device_memory().get_memory_stats(&stats);
    std::stringstream msg;
    if (is_workspace) {
      msg << "Memory Allocation failure for workspace size::" << size
          << std::endl
          << stats.DebugString();
    } else {
      msg << "Memory Allocation failure for persistant Tensor size::" << size
          << std::endl
          << stats.DebugString();
    }
    synapse_helpers::print_live_allocations(msg.str().c_str());
  }
}

/*
 * Print live allocation data at the given point
 */
void print_live_allocations(const char* msg) {
  deviceMallocData::singleton().print_live_allocations(msg);
}

void log_DRAM_start(uint64_t dram_start) {
  deviceMallocData::singleton().set_dram_start(dram_start);
}

void log_DRAM_size(uint64_t dram_size) {
  deviceMallocData::singleton().set_dram_size(dram_size);
}

void set_back_trace(bool enable) {
  deviceMallocData::singleton().set_back_trace(enable);
}

void set_memstats_check_flag(bool flag) {
  deviceMallocData::singleton().set_memstats_check_flag(flag);
}

void memstats_dump(synapse_helpers::device& device, const char* msg) {
  if (deviceMallocData::singleton().get_memstats_check_flag() ||
      GET_ENV_FLAG_NEW(PT_HPU_MEM_STATS_DUMP)) {
    synapse_helpers::MemoryStats stats;
    device.get_device_memory().get_memory_stats(&stats);
    std::string updated_msg = msg;
    updated_msg = updated_msg + "\n" + stats.DebugString();
    synapse_helpers::print_live_allocations(updated_msg.c_str());
  }
}

bool memory_reporter_enable() {
  auto& dmd = deviceMallocData::singleton();
  return dmd.is_mem_reporter_enabled();
}

void deviceMallocData::create_memory_reporter_event(
    synapse_helpers::device& device,
    std::string& event_name) {
  synapse_helpers::MemoryStats mem_stats;
  device.get_device_memory().get_memory_stats(&mem_stats);
  synapse_helpers::MemoryReporter* reporter =
      device.get_device_memory().get_memory_reporter();
  // Redirect output to logfile
  static int status_id = 0;
  int event_ts = status_id++;
  std::string event_line_begin_header = std::string("{ \"tid\":") +
      std::to_string(event_ts) + std::string(", \"pid\":") +
      std::to_string(getpid()) + std::string(", \"ts\":") +
      std::to_string(event_ts);
  std::string event_line_end_header = std::string("{ \"tid\":") +
      std::to_string(event_ts) + std::string(", \"pid\":") +
      std::to_string(getpid()) + std::string(", \"ts\":") +
      std::to_string(event_ts + 1);
  std::string report_event_begin = event_line_begin_header;
  report_event_begin += std::string(", \"name\":\"") + event_name +
      std::string("\", \"ph\":\"B\", \"cat\":\"ReportEvent\"") +
      std::string("},\n");
  std::string report_event_end = event_line_end_header;
  report_event_end += std::string(", \"name\":\"") + event_name +
      std::string("\", \"ph\":\"E\", \"cat\":\"ReportEvent\"") +
      std::string("},\n");

  auto reporter_out_stream = get_memory_reporter_out_stream();

  reporter_out_stream << report_event_begin;

  reporter_out_stream << event_line_begin_header +
          std::string(", \"name\":\"") + "TotalMemoryAvailable" +
          std::string("\", \"ph\":\"B\", \"cat\":\"TotalMemoryAvailable\"") +
          std::string(", \"args\": { ") +
          TO_REPORT_EVENT_GB("TotalMemoryAvailable", mem_stats.memory_limit) +
          std::string("}},\n");
  reporter_out_stream << event_line_end_header + std::string(", \"name\":\"") +
          "TotalMemoryAvailable" +
          std::string("\", \"ph\":\"E\", \"cat\":\"TotalMemoryAvailable\"") +
          std::string("},\n");

  // memory consumption event create
  synapse_helpers::MemoryConsumption* mem_consume =
      reporter->getMemoryConsumption();
  mem_consume->update(mem_stats);
  reporter_out_stream << mem_consume->toJsonEvent(
      event_line_begin_header, event_line_end_header);

  // memory allocator stats event create
  synapse_helpers::MemoryAllocatorStats* mem_alloc_stats =
      reporter->getMemoryAllocatorStats();
  mem_alloc_stats->update(mem_stats);
  reporter_out_stream << mem_alloc_stats->toJsonEvent(
      event_line_begin_header, event_line_end_header);

  // fragmentation stats event create
  synapse_helpers::FragmentationStats* frag_stats =
      reporter->getFragmentationStats();
  frag_stats->update(mem_stats);
  reporter_out_stream << frag_stats->toJsonEvent(
      event_line_begin_header, event_line_end_header);

  // graph stats event create
  synapse_helpers::GraphStats* graph_stats = reporter->getGraphStats();
  reporter_out_stream << graph_stats->toJsonEvent(
      event_line_begin_header, event_line_end_header);

  // tensor stats event create
  synapse_helpers::TensorStats* tensor_stats = reporter->getTensorStats();
  reporter_out_stream << tensor_stats->toJsonEvent(
      event_line_begin_header, event_line_end_header);

  reporter_out_stream << report_event_end;
}

void memory_reporter_event_create(
    synapse_helpers::device& device,
    synapse_helpers::mem_reporter_type event_type) {
  if (memory_reporter_enable()) {
    std::string event_name = "";
    switch (event_type) {
      case MEM_REPORTER_GRAPH_BEFORE_LAUNCH:
        event_name = "GRAPH_BEFORE_LAUNCH_EVENT";
        break;
      case MEM_REPORTER_GRAPH_AFTER_LAUNCH:
        event_name = "GRAPH_AFTER_LAUNCH_EVENT";
        break;
      case MEM_REPORTER_ALLOC_FAILS:
        event_name = "ALLOC_FAIL_EVENT";
        break;
      case MEM_REPORTER_OOM:
        event_name = "OOM_EVENT";
        break;
      case MEM_REPORTER_USER_CALL:
        event_name = "USER_REQUEST_EVENT";
        break;
      case MEM_DEFRAGMENT_START:
        event_name = "DEFRAGMENT_START";
        break;
      case MEM_DEFRAGMENT_SUCCESS:
        event_name = "DEFRAGMENT_SUCCESS";
        break;
      case MEM_DEFRAGMENT_FAIL:
        event_name = "DEFRAGMENT_FAIL";
        break;
    }
    auto& dmd = deviceMallocData::singleton();
    dmd.create_memory_reporter_event(device, event_name);
  }
}

} // namespace synapse_helpers
