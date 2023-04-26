/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "status_conversion.h"
#include <synapse_api.h>
#include "backend/synapse_helpers/synapse_error.h"
#include "habana_helpers/logging.h"

namespace hccl_integration {

hcclResult_t to_hccl_result(const synapse_helpers::synapse_error& status) {
  if (status.status != synSuccess) {
    PT_DISTRIBUTED_FATAL(
        Logger::formatStatusMsg(status.status),
        ": error occurred, msg: ",
        status.error);
  }
  return to_hccl_result(status.status);
}

// TODO: move to hccl public headers
hcclResult_t to_hccl_result(synStatus status) {
  switch (status) {
    case synSuccess:
      return hcclSuccess;
    case synInvalidArgument:
      return hcclInvalidArgument;
    case synCbFull:
      return hcclInternalError;
    case synOutOfHostMemory:
    case synOutOfDeviceMemory:
    case synOutOfResources:
      return hcclOutOfMemory;
    case synObjectAlreadyInitialized:
    case synObjectNotInitialized:
      return hcclInvalidUsage;
    case synCommandSubmissionFailure:
      return hcclInternalError;
    case synNoDeviceFound:
    case synDeviceTypeMismatch:
      return hcclInvalidUsage;
    case synFailedToInitializeCb:
    case synFailedToFreeCb:
    case synFailedToMapCb:
    case synFailedToUnmapCb:
    case synFailedToAllocateDeviceMemory:
    case synFailedToFreeDeviceMemory:
      return hcclInternalError;
    case synFailedNotEnoughDevicesFound:
      return hcclInvalidUsage;
    case synDeviceReset:
      return hcclUnhandledSynapseError;
    case synUnsupported:
      return hcclInternalError;
    case synWrongParamsFile:
    case synDeviceAlreadyAcquired:
    case synUninitialized:
    case synAlreadyInitialized:
      return hcclInvalidUsage;
    case synNameIsAlreadyUsed:
    case synBusy:
    case synAllResourcesTaken:
      return hcclSystemError;
    case synUnavailable:
      return hcclUnhandledSynapseError;
    case synInvalidTensorDimensions:
      return hcclInvalidArgument;
    case synFail:
      return hcclInternalError;
    default:
      PT_DISTRIBUTED_FATAL(
          "Encountered unrecognized synStatus enum value of: ",
          static_cast<int>(status));
  }

  return hcclInternalError;
}

} // namespace hccl_integration
