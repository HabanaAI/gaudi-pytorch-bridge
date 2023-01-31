#pragma once
#include <string.h>
#include <string_view>

namespace habana {

void export_profiler_logs(const std::string_view& path);
void start_profiler_session();
void stop_profiler_session();
}; // namespace habana