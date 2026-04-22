/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "pass_utils.h"
#include "backend/jitgraph_utils.h"
using namespace jitgraph_utils;
namespace habana_lazy {
at::IntArrayRef getDimsForLayout5d(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order) {
  at::IntArrayRef dims;

  // using NCHW/NHWC/HWCK since synapse 5d layout nomenclature
  // is not clear. Note that for 5d layout channel dim is 1 for
  // NCDHW and 4 for NDHWC
  if (current_order == habana::LayoutFormat::NCHW) {
    if (channel_order == habana::LayoutFormat::NHWC) {
      static const std::array<int64_t, 5> dimarr = {
          habana::LayoutFormatWithDepthDims::N,
          habana::LayoutFormatWithDepthDims::D,
          habana::LayoutFormatWithDepthDims::H,
          habana::LayoutFormatWithDepthDims::W,
          habana::LayoutFormatWithDepthDims::C};
      dims = c10::makeArrayRef(dimarr);
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const std::array<int64_t, 5> dimarr = {
          habana::LayoutFormatWithDepthDims::D,
          habana::LayoutFormatWithDepthDims::H,
          habana::LayoutFormatWithDepthDims::W,
          habana::LayoutFormatWithDepthDims::C,
          habana::LayoutFormatWithDepthDims::N};
      dims = c10::makeArrayRef(dimarr);
    } else {
      HABANA_ASSERT(
          0,
          " InsertPermute_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::NHWC) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const std::array<int64_t, 5> dimarr = {
          habana::LayoutFormatWithDepthDims::N,
          habana::LayoutFormatWithDepthDims::W,
          habana::LayoutFormatWithDepthDims::C,
          habana::LayoutFormatWithDepthDims::D,
          habana::LayoutFormatWithDepthDims::H};
      dims = c10::makeArrayRef(dimarr);
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const std::array<int64_t, 5> dimarr = {
          habana::LayoutFormatWithDepthDims::C,
          habana::LayoutFormatWithDepthDims::D,
          habana::LayoutFormatWithDepthDims::H,
          habana::LayoutFormatWithDepthDims::W,
          habana::LayoutFormatWithDepthDims::N};
      dims = c10::makeArrayRef(dimarr);
    } else {
      HABANA_ASSERT(
          0,
          " InsertPermute_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::HWCK) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const std::array<int64_t, 5> dimarr = {
          habana::LayoutFormatWithDepthDims::W,
          habana::LayoutFormatWithDepthDims::H,
          habana::LayoutFormatWithDepthDims::N,
          habana::LayoutFormatWithDepthDims::C,
          habana::LayoutFormatWithDepthDims::D};
      dims = c10::makeArrayRef(dimarr);
    } else if (channel_order == habana::LayoutFormat::NHWC) {
      static const std::array<int64_t, 5> dimarr = {
          habana::LayoutFormatWithDepthDims::W,
          habana::LayoutFormatWithDepthDims::N,
          habana::LayoutFormatWithDepthDims::C,
          habana::LayoutFormatWithDepthDims::D,
          habana::LayoutFormatWithDepthDims::H};
      dims = c10::makeArrayRef(dimarr);
    } else {
      HABANA_ASSERT(
          0,
          " InsertPermute_graph: permute called for unsupported channel order");
    }
  } else {
    HABANA_ASSERT(
        0,
        " InsertPermute_graph: permute called for unsupported channel order");
  }

  return dims;
}

at::IntArrayRef getDimsForLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order) {
  at::IntArrayRef dims;

  if (current_order == habana::LayoutFormat::NCHW) {
    if (channel_order == habana::LayoutFormat::NHWC) {
      static const std::array<int64_t, 4> dimarr = {
          habana::LayoutFormatDims::N,
          habana::LayoutFormatDims::H,
          habana::LayoutFormatDims::W,
          habana::LayoutFormatDims::C};
      dims = c10::makeArrayRef(dimarr);
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const std::array<int64_t, 4> dimarr = {
          habana::LayoutFormatDims::H,
          habana::LayoutFormatDims::W,
          habana::LayoutFormatDims::C,
          habana::LayoutFormatDims::N};
      dims = c10::makeArrayRef(dimarr);
    } else {
      HABANA_ASSERT(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::NHWC) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const std::array<int64_t, 4> dimarr = {
          habana::LayoutFormatDims::N,
          habana::LayoutFormatDims::W,
          habana::LayoutFormatDims::C,
          habana::LayoutFormatDims::H};
      dims = c10::makeArrayRef(dimarr);
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const std::array<int64_t, 4> dimarr = {
          habana::LayoutFormatDims::C,
          habana::LayoutFormatDims::H,
          habana::LayoutFormatDims::W,
          habana::LayoutFormatDims::N};
      dims = c10::makeArrayRef(dimarr);
    } else {
      HABANA_ASSERT(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::HWCK) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const std::array<int64_t, 4> dimarr = {
          habana::LayoutFormatDims::W,
          habana::LayoutFormatDims::H,
          habana::LayoutFormatDims::N,
          habana::LayoutFormatDims::C};
      dims = c10::makeArrayRef(dimarr);
    } else if (channel_order == habana::LayoutFormat::NHWC) {
      static const std::array<int64_t, 4> dimarr = {
          habana::LayoutFormatDims::W,
          habana::LayoutFormatDims::N,
          habana::LayoutFormatDims::C,
          habana::LayoutFormatDims::H};
      dims = c10::makeArrayRef(dimarr);
    } else {
      HABANA_ASSERT(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else {
    HABANA_ASSERT(
        0,
        " InsertPermtue_graph: permute called for unsupported channel order");
  }

  return dims;
}
} // namespace habana_lazy
