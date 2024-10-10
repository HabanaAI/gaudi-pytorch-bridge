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
#include "backend/profiling/profiling.h"

namespace habana {
namespace profile {

class MemorySource : public TraceSource {
 public:
  ~MemorySource() override;
  void start(TraceSink&) override;
  void stop() override;
  void extract(TraceSink& output) override;
  TraceSourceVariant get_variant() override;
  void set_offset(unsigned offset) override;
};
}; // namespace profile
}; // namespace habana