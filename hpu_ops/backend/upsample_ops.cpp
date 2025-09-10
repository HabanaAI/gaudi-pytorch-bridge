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

#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/upsample_bicubic2d.h"
#include "generated/backend/upsample_bilinear2d.h"
#include "generated/backend/upsample_linear1d.h"
#include "generated/backend/upsample_nearest1d.h"
#include "generated/backend/upsample_nearest2d.h"
#include "generated/backend/upsample_nearest3d.h"
#include "hpu_ops/hpu_op_helper.h"
namespace habana {
struct UpsampleNearest1dVec : UpsampleNearest1DFwdOperator {
  UpsampleNearest1dVec(int device_id, c10::ScalarType scalar_type)
      : UpsampleNearest1DFwdOperator(
            device_id,
            "resize_fwd",
            scalar_type,
            {0},
            {},
            {},
            false) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    SetOutputMetaFn(UpsampleNearest1DFwdMeta);
  }
};

struct UpsampleLinear1dVec : UpsampleLinear1DFwdOperator {
  UpsampleLinear1dVec(int device_id, c10::ScalarType scalar_type)
      : UpsampleLinear1DFwdOperator(
            device_id,
            "resize_fwd",
            scalar_type,
            {0},
            {},
            {},
            false) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    SetOutputMetaFn(UpsampleLinear1DFwdMeta);
  }
};

struct UpsampleBilinear2dVec : OpBackend {
  UpsampleBilinear2dVec(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "resize_fwd", scalar_type, {0}, {}, {}, false) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    SetOutputMetaFn(UpsampleBilinear2DFwdMeta);
    SetFillParams(FillBilinearFwdParams);
  }
};

struct UpsampleNearest2dVec : UpSampleNearest2DOperator {
  UpsampleNearest2dVec(int device_id, c10::ScalarType scalar_type)
      : UpSampleNearest2DOperator(
            device_id,
            "resize_fwd",
            scalar_type,
            {0},
            {},
            {},
            false) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    SetOutputMetaFn(UpsampleNearest2DFwdMeta);
    SetFillParams(FillNearestFwdParams);
  }
};

struct UpsampleNearest3dVec : UpSampleNearest3DFwdOperator {
  UpsampleNearest3dVec(int device_id, c10::ScalarType scalar_type)
      : UpSampleNearest3DFwdOperator(
            device_id,
            "resize_fwd",
            scalar_type,
            {0},
            {},
            {},
            false) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
    SetOutputMetaFn(UpsampleNearest3DFwdMeta);
  }
};

static const auto& UpsampleKernelRegistry =
    KernelRegistry()
        .REGISTER_HPU_BACKEND(
            "aten::upsample_nearest1d.vec",
            UpsampleNearest1dVec)
        .REGISTER_HPU_BACKEND(
            "aten::upsample_linear1d.vec",
            UpsampleLinear1dVec)
        .REGISTER_HPU_BACKEND(
            "aten::upsample_bilinear2d.vec",
            UpsampleBilinear2dVec)
        .REGISTER_HPU_BACKEND(
            "aten::upsample_nearest2d.vec",
            UpsampleNearest2dVec)
        .REGISTER_HPU_BACKEND(
            "aten::upsample_nearest3d.vec",
            UpsampleNearest3dVec);
} // namespace habana
