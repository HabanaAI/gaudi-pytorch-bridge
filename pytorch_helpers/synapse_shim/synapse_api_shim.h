/**
* Copyright (c) 2021-2024 Intel Corporation
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
#pragma once

#include <functional>

#include <synapse_api.h> // IWYU pragma: keep

#include "synapse_logger/synapse_logger_observer.h"

#define SYN_API_SYMBOL_VISIT(visitor)         \
  visitor(synDeviceSynchronize);              \
  visitor(synStreamCreateGeneric);            \
  visitor(synStreamDestroy);                  \
  visitor(synDeviceGetNextStreamAffinity);    \
  visitor(synStreamSetAffinity);              \
  visitor(synStreamWaitEvent);                \
  visitor(synStreamSynchronize);              \
  visitor(synStreamQuery);                    \
  visitor(synEventCreate);                    \
  visitor(synEventDestroy);                   \
  visitor(synEventRecord);                    \
  visitor(synEventQuery);                     \
  visitor(synEventSynchronize);               \
  visitor(synEventElapsedTime);               \
  visitor(synLaunch);                         \
  visitor(synWorkspaceGetSize);               \
  visitor(synMemCopyAsync);                   \
  visitor(synMemCopyAsyncMultiple);           \
  visitor(synDeviceGetCount);                 \
  visitor(synDeviceGetCountByDeviceType);     \
  visitor(synDeviceAcquireByDeviceType);      \
  visitor(synDeviceAcquireByModuleId);        \
  visitor(synDeviceAcquire);                  \
  visitor(synDriverGetVersion);               \
  visitor(synDeviceGetName);                  \
  visitor(synTensorRetrieveIds);              \
  visitor(synTensorDestroy);                  \
  visitor(synTensorHandleCreate);             \
  visitor(synNodeCreate);                     \
  visitor(synNodeCreateWithId);               \
  visitor(synNodeSetDeterministic);           \
  visitor(synNodeDependencySet);              \
  visitor(synNodeSetUserProgrammability);     \
  visitor(synGraphCompile);                   \
  visitor(synGraphCreate);                    \
  visitor(synGraphCreateEager);               \
  visitor(synGraphSetAttribute);              \
  visitor(synGraphGetAttribute);              \
  visitor(synGraphDuplicate);                 \
  visitor(synGraphInferShapes);               \
  visitor(synGraphDestroy);                   \
  visitor(synMemsetD32Async);                 \
  visitor(synMemsetD8Async);                  \
  visitor(synMemsetD16Async);                 \
  visitor(synHostMalloc);                     \
  visitor(synHostFree);                       \
  visitor(synHostMap);                        \
  visitor(synHostUnmap);                      \
  visitor(synDeviceMalloc);                   \
  visitor(synDeviceFree);                     \
  visitor(synInitialize);                     \
  visitor(synDestroy);                        \
  visitor(synDeviceRelease);                  \
  visitor(synDeviceGetMemoryInfo);            \
  visitor(synDeviceGetInfo);                  \
  visitor(synDeviceGetInfoV2);                \
  visitor(synProfilerStart);                  \
  visitor(synProfilerStop);                   \
  visitor(synProfilerGetTrace);               \
  visitor(synProfilerQueryRequiredMemory);    \
  visitor(synProfilerSetUserBuffer);          \
  visitor(synConfigurationSet);               \
  visitor(synConfigurationGet);               \
  visitor(synSectionCreate);                  \
  visitor(synSectionGetRMW);                  \
  visitor(synSectionSetRMW);                  \
  visitor(synSectionGetPersistent);           \
  visitor(synSectionSetPersistent);           \
  visitor(synSectionDestroy);                 \
  visitor(synSectionSetConst);                \
  visitor(synRecipeSectionHostBuffersClear);  \
  visitor(synRecipeSectionGetProp);           \
  visitor(synRecipeSerialize);                \
  visitor(synRecipeDeSerialize);              \
  visitor(synRecipeGetAttribute);             \
  visitor(synDeviceGetAttribute);             \
  visitor(synRecipeDestroy);                  \
  visitor(synSectionSetGroup);                \
  visitor(synProfilerGetCurrentTimeNS);       \
  visitor(synProfilerAddCustomMeasurement);   \
  visitor(synTensorExtExtractExecutionOrder); \
  visitor(synEventMapTensor);                 \
  visitor(synLaunchWithExternalEvents);       \
  visitor(synTensorSetExternal);              \
  visitor(synTensorGetExternal);              \
  visitor(synTensorAssignToSection);          \
  visitor(synTensorSetSectionOffset);         \
  visitor(synNodeGetUserParams);              \
  visitor(synNodeSetUserParams);              \
  visitor(synTensorSetHostPtr);               \
  visitor(synTensorGetGeometry);              \
  visitor(synTensorSetGeometry);              \
  visitor(synTensorSetDeviceFullLayout);      \
  visitor(synTensorSetQuantizationData);      \
  visitor(synTensorCreate);                   \
  visitor(synTensorGetName);                  \
  visitor(synTensorSetPermutation);           \
  visitor(synTensorRetrieveLaunchAmount);     \
  visitor(synTensorRetrieveLaunchIds);        \
  visitor(synTensorRetrieveLaunchInfoById);   \
  visitor(synConstTensorCreate);              \
  visitor(synTensorSetAllowPermutation);      \
  visitor(synTensorGetHostPtr);               \
  visitor(synTensorSetDeviceDataType);        \
  visitor(synStatusGetBriefDescription);      \
  visitor(synDumpStateAndTerminate);

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
void EnableSynapseApiLogger(synapse_logger::SynapseLoggerObserver*);
void EnableSynapseApiStub();
void EnableNullHw();
