/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <dlfcn.h>
#include <stdlib.h>
#include <inttypes.h>

#include <execinfo.h>
#include <unistd.h>
#include <stdbool.h>
#include <assert.h>

#include <cstdlib>
#include <memory>
#include <cxxabi.h>

#include <HPUAllocate_wrapper.h>

deviceMallocData::deviceMallocData() {
  iteration_number = 0;
  running_memory = iteration_high_watermark = overall_high_watermark = 0;
  bt_depth = 40;
  take_bt = true;
  // Disable backtrace with HABANA_MEM_COLLECT_BT=0
  if (const char* env_p = std::getenv("HABANA_MEM_COLLECT_BT")) {
    take_bt = !(*env_p == '0');
  }

  // Modify log filename with HABANA_MEM_LIVEALLOC_LOGNAME=filename
  if (const char* env_p = std::getenv("HABANA_MEM_LIVEALLOC_LOGNAME")) {
    filename = env_p;
  }

  print_free_bt = std::getenv("HABANA_MEM_PRINT_FREE");

  dram_start_ = dram_size_ = 0;

  out.open(filename, std::ofstream::out);
}

deviceMallocData::~deviceMallocData() {
  if (out.is_open()) {
    out.close();
  }
}

deviceMallocData& deviceMallocData::singleton() {
  static deviceMallocData* instance = new deviceMallocData();
  return *instance;
}

bool deviceMallocData::sort_by_size(
  std::pair<uint64_t, size_bt_pair_t> a,
  std::pair<uint64_t, size_bt_pair_t> b) {
    return a.second.first > b.second.first;
}

bool deviceMallocData::sort_by_ptr(
  std::pair<uint64_t, size_bt_pair_t> a,
  std::pair<uint64_t, size_bt_pair_t> b) {
    return a.first < b.first;
}
/*
 * Check if the stack frame contain functions/modules that
 * we would like to see.
 */
bool deviceMallocData::interesting_function(
  const std::string& name) {
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
  for(auto name_entry : list_of_interest) {
    if (name.find(name_entry) != std::string::npos) {
      interesting = true;
      break;
    }
  }

  return interesting;
}

/*
 * Print an entry from the log
 */
void deviceMallocData::print_an_entry(
  const std::pair<uint64_t, size_bt_pair_t>& entry) {

  // Print the data pointer
  std::cout << "ptr = 0x" << std::hex << entry.first << "\n" << std::flush;

  // Print the allocated size
  const auto& size_bt = entry.second;
  std::cout << "    size = " << std::dec << size_bt.first << "\n" << std::flush;

  // Print the backtrace
  const auto& bt_strings = size_bt.second;
  std::cout << "    BACKTRACE\n" << std::flush;
  const std::string dot_dot_dot = "...";
  bool dot_marker_placed = false;
  for(const auto& string : bt_strings) {
    // Find the mangled function name in the frame
    const auto start_of_func_name = string.find("(");
    std::size_t end_of_func_name;
    bool formatted_name = true;
    if (start_of_func_name != std::string::npos) {
      end_of_func_name = string.find("+", start_of_func_name);
      if (   (end_of_func_name == std::string::npos)
          || (end_of_func_name == start_of_func_name + 1)) {
        formatted_name = false;
      }
    } else {
      formatted_name = false;
    }
    if (formatted_name) {
      // Print demangled name
      const auto len = end_of_func_name - start_of_func_name - 1;
      int status;
      const auto& name = string.substr(start_of_func_name+1, len);
      const auto demangled_name = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);
      if (!interesting_function((status == 0) ? demangled_name : name)) {
        // If the function isn't of interest, don't print the frame
        if (!dot_marker_placed) {
          std::cout << "    " << dot_dot_dot << "\n" << std::flush;
          dot_marker_placed = true;
        }
        continue;
      }
      dot_marker_placed = false;
      if (status == 0) {
        std::cout << "    " << demangled_name << "\n" << std::flush;
      } else {
        std::cout << "    " << name << "\n" << std::flush;
      }
    } else {
      if (!interesting_function(string)) {
        // If the function isn't of interest, don't print the frame
        if (!dot_marker_placed) {
          std::cout << "    " << dot_dot_dot << "\n" << std::flush;
          dot_marker_placed = true;
        }
        continue;
      }
      dot_marker_placed = false;
      std::cout << "    " << string << "\n" << std::flush;
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
    bool alloc_failure)
{
  int nptrs;
  void *buffer[bt_depth];
  char **strings;

  std::vector<std::string> bt_string;
  // Take backtrace
  if (take_bt && (alloc || print_free_bt)) {
    nptrs = backtrace(buffer, bt_depth);

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL) {
      perror("backtrace_symbols");
      exit(EXIT_FAILURE);
    }

    for (int i = 2; i < nptrs; i++)
    {
      bt_string.push_back(strings[i]);
    }

    free(strings);
  }

  if (alloc_failure) {
    std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
    std::cout.rdbuf(out.rdbuf());

    std::pair<uint64_t, size_bt_pair_t> entry
      = std::make_pair(ptr, std::make_pair(size, bt_string));

    std::cout << "=========================\n" << std::flush;
    std::cout << "Allocation failed from\n" << std::flush;
    print_an_entry(entry);
    std::cout << "=========================\n" << std::flush;

    std::cout.rdbuf(coutbuf); //reset to standard output again
  } else {
    // synDeviceMalloc
    if (alloc) {
      assert(ptr_bt_map.find(ptr) == ptr_bt_map.end());
      // Log the entry :
      //   ptr -> (size, backtrace)
      ptr_bt_map[ptr] = std::make_pair(size, bt_string);

      // Stats update
      running_memory += size;
      if (running_memory > iteration_high_watermark) {
        iteration_high_watermark = running_memory;
      }

      if (iteration_high_watermark > overall_high_watermark) {
        overall_high_watermark = iteration_high_watermark;
      }
    } else {
      // synDeviceFree
      auto it = ptr_bt_map.find(ptr);
      assert(it != ptr_bt_map.end());

      // Update stat
      running_memory -= it->second.first;

      // If we want a backtrace on each free (very verbose!)
      if (print_free_bt) {
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        std::cout.rdbuf(out.rdbuf());

        std::pair<uint64_t, size_bt_pair_t> entry
          = std::make_pair(it->first, std::make_pair(it->second.first, bt_string));

        std::cout << "=========================\n" << std::flush;
        std::cout << "Free record entry\n" << std::flush;
        print_an_entry(entry);
        std::cout << "=========================\n" << std::flush;

        std::cout.rdbuf(coutbuf); //reset to standard output again
      }

      // Remove the entry from live allocations list
      ptr_bt_map.erase(it);
    }
  }
}

void deviceMallocData::report_fragmentation() {
  print_live_allocations();
  // Redirect output to logfile
  std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
  std::cout.rdbuf(out.rdbuf());

  std::cout << "Fragmentation report\n" << std::flush;

  if ((dram_size_ == 0) || (dram_start_ == 0)) {
    std::cout << " No record of DRAM start or size!\n" << std::flush;
  } else {
    uint64_t current_head = dram_start_;

    std::vector< std::pair<uint64_t, size_bt_pair_t> >
      sorted_by_ptr_log(ptr_bt_map.begin(), ptr_bt_map.end());

    // Sort the entries by ptr address
    std::sort(sorted_by_ptr_log.begin(), sorted_by_ptr_log.end(), sort_by_ptr);

    std::vector< std::pair<uint64_t, uint64_t> > free_list;

    for(const auto& entry : sorted_by_ptr_log) {
      const auto& entry_addr = entry.first;
      if (current_head < entry_addr) {
        free_list.emplace_back(std::make_pair(current_head, entry_addr - current_head));
      } else {
        assert(current_head == entry_addr);
      }
      current_head = entry_addr + entry.second.first;
    }
    if (current_head >= dram_start_ + dram_size_) {
      std::cout << "WARNING: DRAM size data is probably wrong. "
                   "Last allocation exceeds DRAM size\n" << std::flush;
    } else if (current_head != dram_start_ + dram_size_) {
      free_list.emplace_back(std::make_pair(current_head,
                                            dram_start_ + dram_size_ - current_head));
    }

    std::cout << "Free List\n" << std::flush;
    for(const auto& entry : free_list) {
      std::cout << "0x" << std::hex << entry.first << ": "
                << std::dec << entry.second << "\n" << std::flush;
    }

    std::cout.rdbuf(coutbuf); //reset to standard output again

    std::ofstream csv_out;
    csv_out.open(fragment_csv_file, std::ofstream::out);
    // Redirect output to logfile
    coutbuf = std::cout.rdbuf(); //save old buf
    std::cout.rdbuf(csv_out.rdbuf());
    auto ptr_start = dram_start_;
    size_t running_size = 0;
    for(const auto& entry : free_list) {
      // ptr_start to free list entry is occupied
      if (entry.first > ptr_start) {
        auto occupied_size = entry.first - ptr_start - 1;
        // Mark occupied range with 1
        std::cout << running_size << ", 1\n" << std::flush;
        std::cout << running_size + occupied_size << ", 1\n" << std::flush;
        ptr_start = entry.first;
        running_size += occupied_size + 1;
      }
      assert(ptr_start == entry.first);
      // Mark free range with 0
      std::cout << running_size << ", 0\n" << std::flush;
      std::cout << running_size + entry.second << ", 0\n" << std::flush;
      running_size += entry.second + 1;
    }
    if (ptr_start < dram_start_ + dram_size_) {
      // Mark if the last part is occupied
      std::cout << dram_start_ + dram_size_ - ptr_start << ", 1\n" << std::flush;
    }

    std::cout.rdbuf(coutbuf); //reset to standard output again
    csv_out.close();
  }
}

/*
 * Print live allocation details at the given point.
 */
void deviceMallocData::print_live_allocations() {
  // Redirect output to logfile
  std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
  std::cout.rdbuf(out.rdbuf());

  std::cout << "\n=========================\n" << std::flush;
  std::cout << "LIVE ALLOCATIONS DATA (instance = " << iteration_number << ")\n" << std::flush;
  std::cout << "=========================\n" << std::flush;
  std::cout << "DRAM start: 0x" << std::hex << dram_start_ << "\n" << std::flush;
  std::cout << "DRAM size: " << std::dec << dram_size_ << " (" << dram_size_/(1024*1024*1024.) << " GB)\n" << std::flush;
  std::vector< std::pair<uint64_t, size_bt_pair_t> >
    sorted_by_size_log(ptr_bt_map.begin(), ptr_bt_map.end());

  // Sort the entries by size
  std::sort(sorted_by_size_log.begin(), sorted_by_size_log.end(), sort_by_size);

  // How many allocations are not freed yet?
  std::cout << "#Allocations live (instance = " << iteration_number << ") : "
            << sorted_by_size_log.size() << "\n" << std::flush;

  // How much memory is held by our live allocations now?
  size_t total_live_size = 0;
  for(const auto& entry : sorted_by_size_log) {
    total_live_size += entry.second.first;
  }
  std::cout << "Total memory held (instance = " << iteration_number << ") : "
            << total_live_size << " (" << total_live_size/(1024*1024.)
            << " MB)\n" << std::flush;

  // Stats on peak memory usage
  std::cout << "Peak memory usage (instance = " << iteration_number << ") : "
            << overall_high_watermark << " (" << overall_high_watermark/(1024*1024.)
            << " MB)\n" << std::flush;

  std::cout << "Peak memory usage from last log (instance = " << iteration_number << ") : "
            << iteration_high_watermark << " (" << iteration_high_watermark/(1024*1024.)
            << " MB)\n" << std::flush;

  ++iteration_number;
  iteration_high_watermark = 0;

  // Some allocations persist, what are the new live allocations from last report?
  std::cout << "New allocations since last log\n" << std::flush;
  // Find entries that ae new for this log and print them
  if (!ptr_bt_map_last.empty()) {
    for(const auto& entry : sorted_by_size_log) {
      if (ptr_bt_map_last.find(entry.first) == ptr_bt_map_last.end()) {
        print_an_entry(entry);
      }
    }
  }

  // Print all entries that are live at this point
  std::cout << "All live allocations\n" << std::flush;
  for(const auto& entry : sorted_by_size_log) {
    print_an_entry(entry);
  }

  std::cout.rdbuf(coutbuf); //reset to standard output again

  // Save current log to compare against next time log
  ptr_bt_map_last = ptr_bt_map;
}

/*
 * log synDeviceMalloc
 */
void log_synDeviceMalloc(uint64_t ptr, size_t size, bool failed)
{
  deviceMallocData::singleton().collect_backtrace(ptr, true, size, failed);
  if (failed) {
    deviceMallocData::singleton().report_fragmentation();
  }
}

/*
 * log synDeviceFree
 */
void log_synDeviceFree(uint64_t ptr)
{
  deviceMallocData::singleton().collect_backtrace(ptr, false);
}

/*
 * Print live allocation data at the given point
 */
void print_live_allocations() {
  deviceMallocData::singleton().print_live_allocations();
}

void log_DRAM_start(uint64_t dram_start) {
  deviceMallocData::singleton().set_dram_start(dram_start);
}

void log_DRAM_size(uint64_t dram_size) {
  deviceMallocData::singleton().set_dram_size(dram_size);
}

