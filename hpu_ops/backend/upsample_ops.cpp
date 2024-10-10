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
 *******************************************************************************/
#include <pytorch_helpers/habana_helpers/pt_version_check.h>

#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/upsample_bicubic2d.h"
#include "generated/backend/upsample_bilinear2d.h"
#include "generated/backend/upsample_linear1d.h"
#include "generated/backend/upsample_nearest1d.h"
#include "generated/backend/upsample_nearest2d.h"
#include "generated/backend/upsample_nearest3d.h"

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

struct UpsampleBicubic2dVec : OpBackend {
  UpsampleBicubic2dVec(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "resize_fwd", scalar_type, {0}, {}, {}, false) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    SetOutputMetaFn(UpsampleBicubic2DFwdMeta);
    SetFillParams(FillBicubicFwdParams);
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
        .add(
            "aten::upsample_nearest1d.vec",
            KERNEL_FN_GLOBAL(UpsampleNearest1dVec))
        .add(
            "aten::upsample_linear1d.vec",
            KERNEL_FN_GLOBAL(UpsampleLinear1dVec))
        .add(
            "aten::upsample_bilinear2d.vec",
            KERNEL_FN_GLOBAL(UpsampleBilinear2dVec))
        .add(
            "aten::upsample_bicubic2d.vec",
            KERNEL_FN_GLOBAL(UpsampleBicubic2dVec))
        .add(
            "aten::upsample_nearest2d.vec",
            KERNEL_FN_GLOBAL(UpsampleNearest2dVec))
        .add(
            "aten::upsample_nearest3d.vec",
            KERNEL_FN_GLOBAL(UpsampleNearest3dVec));
} // namespace habana
