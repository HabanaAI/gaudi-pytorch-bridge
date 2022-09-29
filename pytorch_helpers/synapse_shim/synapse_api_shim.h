/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/
#pragma once

#include <functional>

#include <synapse.h>
#include <synapse_api.h> // IWYU pragma: keep

#define SYN_API_SYMBOL_VISIT(visitor)          \
  visitor(synSetCfg);                          \
  visitor(synGetCfg);                          \
  visitor(synDeviceSynchronize);               \
  visitor(synStreamCreate);                    \
  visitor(synStreamDestroy);                   \
  visitor(synStreamWaitEvent);                 \
  visitor(synStreamSynchronize);               \
  visitor(synStreamQuery);                     \
  visitor(synEventCreate);                     \
  visitor(synEventDestroy);                    \
  visitor(synEventRecord);                     \
  visitor(synEventQuery);                      \
  visitor(synEventSynchronize);                \
  visitor(synEventElapsedTime);                \
  visitor(synLaunchExt);                       \
  visitor(synWorkspaceGetSize);                \
  visitor(synMemCopyAsync);                    \
  visitor(synMemCopyAsyncMultiple);            \
  visitor(synDeviceGetCount);                  \
  visitor(synDeviceGetCountByDeviceType);      \
  visitor(synDeviceAcquireByDeviceType);       \
  visitor(synDeviceAcquireByModuleId);         \
  visitor(synDeviceAcquire);                   \
  visitor(synDriverGetVersion);                \
  visitor(synDeviceGetName);                   \
  visitor(synTensorRetrieveIds);               \
  visitor(synTensorDestroy);                   \
  visitor(synTensorHandleCreate);              \
  visitor(synNodeCreate);                      \
  visitor(synNodeCreateWithId);                \
  visitor(synNodeSetDeterministic);            \
  visitor(synNodeDependencySet);               \
  visitor(synGetVersion);                      \
  visitor(synGraphCompile);                    \
  visitor(synGraphCreate);                     \
  visitor(synGraphCreateEager);                \
  visitor(synGraphDuplicate);                  \
  visitor(synGraphDestroy);                    \
  visitor(synMemsetD32Async);                  \
  visitor(synMemsetD8Async);                   \
  visitor(synMemsetD16Async);                  \
  visitor(synHostMalloc);                      \
  visitor(synHostFree);                        \
  visitor(synHostMap);                         \
  visitor(synHostUnmap);                       \
  visitor(synDeviceMalloc);                    \
  visitor(synDeviceFree);                      \
  visitor(synInitialize);                      \
  visitor(synDestroy);                         \
  visitor(synDeviceRelease);                   \
  visitor(synDeviceGetMemoryInfo);             \
  visitor(synDeviceGetInfo);                   \
  visitor(synProfilerStart);                   \
  visitor(synProfilerStop);                    \
  visitor(synProfilerGetTrace);                \
  visitor(synConfigurationSet);                \
  visitor(synConfigurationGet);                \
  visitor(synSectionCreate);                   \
  visitor(synSectionGetRMW);                   \
  visitor(synSectionSetRMW);                   \
  visitor(synSectionGetPersistent);            \
  visitor(synSectionSetPersistent);            \
  visitor(synSectionDestroy);                  \
  visitor(synSectionSetConst);                 \
  visitor(synRecipeSectionGetProp);            \
  visitor(synRecipeSerialize);                 \
  visitor(synRecipeDeSerialize);               \
  visitor(synRecipeGetAttribute);              \
  visitor(synDeviceGetAttribute);              \
  visitor(synRecipeDestroy);                   \
  visitor(synSectionSetGroup);                 \
  visitor(synProfilerGetCurrentTimeNS);        \
  visitor(synProfilerAddCustomMeasurement);    \
  visitor(synTensorExtExtractExecutionOrder);  \
  visitor(synEventMapTensorExt);               \
  visitor(synLaunchWithExternalEventsExt);     \
  visitor(synTensorSetExternal);               \
  visitor(synTensorGetExternal);               \
  visitor(synTensorAssignToSection);           \
  visitor(synTensorSetHostPtr);                \
  visitor(synTensorSetGeometryExt);            \
  visitor(synTensorSetDeviceFullLayout);       \
  visitor(synTensorSetQuantizationData);       \
  visitor(synTensorGetQuantizationData);       \
  visitor(synTensorCreate);                    \
  visitor(synTensorSetPermutation);            \
  visitor(synTensorRetrieveLaunchInfoByIdExt); \
  visitor(synConstTensorCreate);               \
  visitor(synDestroyTensor);                   \
  visitor(synTensorSetAllowPermutation);       \
  visitor(synTensorGetHostPtr);                \
  visitor(synTensorSetDeviceDataType);

#define DECL_SYN_FN(func)                       \
  using func##_pfn_t = decltype(::func);        \
  using func##_t = std::function<func##_pfn_t>; \
  func##_t func{};

struct synapse_api_t {
  SYN_API_SYMBOL_VISIT(DECL_SYN_FN)
};

extern synapse_api_t* syn_api;
synapse_api_t* GetSynapseApi();
void EnableSynapseApi();
void EnableSynapseApiLogger();
void EnableSynapseApiStub();
void EnableNullHw();
