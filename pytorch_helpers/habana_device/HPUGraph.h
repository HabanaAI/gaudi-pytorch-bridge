/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "HPUStream.h"

namespace at {
namespace hpu {

struct HPUGraph {
  HPUGraph();
  ~HPUGraph();

  void capture_begin();
  void capture_end();
  void replay();

 protected:
  // Stream on which capture began
  c10::hpu::HPUStream capture_stream_;
};

} // namespace hpu
} // namespace at
