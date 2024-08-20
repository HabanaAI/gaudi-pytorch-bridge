/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */

#pragma once

#include <unordered_map>
#include "backend/profiling/profiling.h"
#include "backend/profiling/trace_sources/trace_parser.h"

namespace habana {
namespace profile {

class SynapseProfilerSource : public TraceSource {
 public:
  SynapseProfilerSource();
  ~SynapseProfilerSource() = default;

  void start(TraceSink& output) override;
  void stop() override;
  void extract(TraceSink& output) override;
  TraceSourceVariant get_variant() override;
  void set_offset(unsigned offset) override;

 private:
  void convertLogs(TraceSink& output);
  void getLogsSize(size_t& size, size_t& count);
  bool getEntries(size_t& size, size_t& count, void* out);
  void initHpuDetails(TraceSink& output);

  std::unique_ptr<HpuTraceParser> parser_;
  long double wall_stop_time_;
  unsigned offset_{};
};
} // namespace profile
} // namespace habana