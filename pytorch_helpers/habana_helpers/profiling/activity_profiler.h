#pragma once
#include <string.h>

namespace habana {

void export_profiler_logs(const std::string_view& path);
void start_profiler_session();
void stop_profiler_session();
uint64_t add_custom_tag_begin(const std::string& tag);
void add_custom_tag_end(uint64_t id);

}; // namespace habana