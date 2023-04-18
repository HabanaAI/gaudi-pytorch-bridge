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
#pragma once

#include <c10/util/Exception.h>

#include <synapse_common_types.h>

// const std::vector<const std::string> synStatusToStr{
//     "synSuccess",
//     "synInvalidArgument",
//     "synCbFull",
//     "synOutOfHostMemory",
//     "synOutOfDeviceMemory",
//     "synObjectAlreadyInitialized",
//     "synObjectNotInitialized",
//     "synCommandSubmissionFailure",
//     "synNoDeviceFound",
//     "synDeviceTypeMismatch",
//     "synFailedToInitializeCb",
//     "synFailedToFreeCb",
//     "synFailedToMapCb",
//     "synFailedToUnmapCb",
//     "synFailedToAllocateDeviceMemory",
//     "synFailedToFreeDeviceMemory",
//     "synFailedNotEnoughDevicesFound",
//     "synDeviceReset",
//     "synUnsupported",
//     "synWrongParamsFile",
//     "synDeviceAlreadyAcquired",
//     "synNameIsAlreadyUsed",
//     "synBusy",
//     "synFail"};

// TODO add "Returned synStatus: ", synStatusToStr[__err] to logger

#define TORCH_HABANA_CHECK(EXPR, ...)   \
  do {                                  \
    synStatus __err = EXPR;             \
    if (__err != synStatus::synSuccess) \
      TORCH_CHECK(false, __VA_ARGS__);  \
  } while (0)

#define TORCH_HABANA_CHECK_WARN(EXPR, ...) \
  do {                                     \
    synStatus __err = EXPR;                \
    if (__err != synStatus::synSuccess)    \
      TORCH_WARN(__VA_ARGS__);             \
  } while (0)
