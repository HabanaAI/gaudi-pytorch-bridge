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

#include <synapse_api.h> // IWYU pragma: keep
#include "logging.h"
#include "partial_event_emulation.h"
#include "synapse_shim/synapse_api_shim.h"

synStatus SYN_API_CALL synRecipeDestroy(synRecipeHandle recipeHandle) {
  return syn_api->synRecipeDestroy(recipeHandle);
}

synStatus SYN_API_CALL
synSectionSetGroup(synSectionHandle sectionHandle, uint64_t sectionGroup) {
  return syn_api->synSectionSetGroup(sectionHandle, sectionGroup);
}

synStatus SYN_API_CALL synInitialize() {
  return syn_api->synInitialize();
}

synStatus SYN_API_CALL synDestroy() {
  return syn_api->synDestroy();
}

synStatus SYN_API_CALL synDeviceSynchronize(const synDeviceId deviceId) {
  return syn_api->synDeviceSynchronize(deviceId);
}

synStatus SYN_API_CALL synStreamCreateGeneric(
    synStreamHandle* pStreamHandle,
    const synDeviceId deviceId,
    const uint32_t flags) {
  return syn_api->synStreamCreateGeneric(pStreamHandle, deviceId, flags);
}

synStatus SYN_API_CALL synStreamDestroy(const synStreamHandle streamHandle) {
  return syn_api->synStreamDestroy(streamHandle);
}

synStatus SYN_API_CALL synDeviceGetNextStreamAffinity(
    const synDeviceId deviceId,
    uint64_t* availAffinity) {
  return syn_api->synDeviceGetNextStreamAffinity(deviceId, availAffinity);
}

synStatus SYN_API_CALL synStreamSetAffinity(
    const synDeviceId deviceId,
    const synStreamHandle pStreamHandle,
    uint64_t availAffinity) {
  return syn_api->synStreamSetAffinity(deviceId, pStreamHandle, availAffinity);
}
synStatus SYN_API_CALL synStreamWaitEvent(
    const synStreamHandle streamHandle,
    synEventHandle eventHandle,
    const uint32_t flags) {
  return syn_api->synStreamWaitEvent(streamHandle, eventHandle, flags);
}

synStatus SYN_API_CALL
synStreamSynchronize(const synStreamHandle streamHandle) {
  return syn_api->synStreamSynchronize(streamHandle);
}

synStatus SYN_API_CALL synStreamQuery(const synStreamHandle streamHandle) {
  return syn_api->synStreamQuery(streamHandle);
}

synStatus SYN_API_CALL synEventCreate(
    synEventHandle* pEventHandle,
    const synDeviceId deviceId,
    const uint32_t flags) {
  return syn_api->synEventCreate(pEventHandle, deviceId, flags);
}

synStatus SYN_API_CALL synEventDestroy(synEventHandle eventHandle) {
  return syn_api->synEventDestroy(eventHandle);
}

synStatus SYN_API_CALL
synEventRecord(synEventHandle eventHandle, const synStreamHandle streamHandle) {
  return syn_api->synEventRecord(eventHandle, streamHandle);
}

synStatus SYN_API_CALL synEventQuery(const synEventHandle eventHandle) {
  return syn_api->synEventQuery(eventHandle);
}

synStatus SYN_API_CALL synEventSynchronize(const synEventHandle eventHandle) {
  return syn_api->synEventSynchronize(eventHandle);
}

synStatus SYN_API_CALL synEventElapsedTime(
    uint64_t* pMilliseconds,
    const synEventHandle eventHandleStart,
    const synEventHandle eventHandleEnd) {
  return syn_api->synEventElapsedTime(
      pMilliseconds, eventHandleStart, eventHandleEnd);
}

synStatus SYN_API_CALL synLaunch(
    const synStreamHandle streamHandle,
    const synLaunchTensorInfo* launchTensorsInfo,
    uint32_t numberTensors,
    uint64_t pWorkspace,
    const synRecipeHandle pRecipehandle,
    uint32_t flags) {
  return syn_api->synLaunch(
      streamHandle,
      launchTensorsInfo,
      numberTensors,
      pWorkspace,
      pRecipehandle,
      flags);
}

synStatus SYN_API_CALL synWorkspaceGetSize(
    uint64_t* pWorkspaceSize,
    const synRecipeHandle recipeHandle) {
  return syn_api->synWorkspaceGetSize(pWorkspaceSize, recipeHandle);
}

synStatus SYN_API_CALL synMemCopyAsync(
    const synStreamHandle streamHandle,
    const uint64_t src,
    const uint64_t size,
    const uint64_t dst,
    const synDmaDir direction) {
  return syn_api->synMemCopyAsync(streamHandle, src, size, dst, direction);
}

synStatus SYN_API_CALL synMemCopyAsyncMultiple(
    const synStreamHandle streamHandle,
    const uint64_t* src,
    const uint64_t* size,
    const uint64_t* dst,
    const synDmaDir direction,
    const size_t numCopies) {
  return syn_api->synMemCopyAsyncMultiple(
      streamHandle, src, size, dst, direction, numCopies);
}

synStatus SYN_API_CALL synDeviceGetCount(uint32_t* pCount) {
  return syn_api->synDeviceGetCount(pCount);
}

synStatus SYN_API_CALL synDeviceGetCountByDeviceType(
    uint32_t* pCount,
    const synDeviceType deviceType) {
  return syn_api->synDeviceGetCountByDeviceType(pCount, deviceType);
}

synStatus SYN_API_CALL synDeviceAcquireByDeviceType(
    synDeviceId* pDeviceId,
    const synDeviceType deviceType) {
  return syn_api->synDeviceAcquireByDeviceType(pDeviceId, deviceType);
}

synStatus SYN_API_CALL
synDeviceAcquireByModuleId(synDeviceId* pDeviceId, const synModuleId moduleId) {
  return syn_api->synDeviceAcquireByModuleId(pDeviceId, moduleId);
}

synStatus SYN_API_CALL
synDeviceAcquire(synDeviceId* pDeviceId, const char* pciBus) {
  return syn_api->synDeviceAcquire(pDeviceId, pciBus);
}

synStatus SYN_API_CALL
synDriverGetVersion(char* pDriverVersion, const int len) {
  return syn_api->synDriverGetVersion(pDriverVersion, len);
}

synStatus SYN_API_CALL
synDeviceGetName(char* pName, const int len, const synDeviceId deviceId) {
  return syn_api->synDeviceGetName(pName, len, deviceId);
}

synStatus SYN_API_CALL synTensorDestroy(const synTensor tensor) {
  return syn_api->synTensorDestroy(tensor);
}

synStatus SYN_API_CALL synTensorRetrieveIds(
    const synRecipeHandle pRecipeHandle,
    const char** tensorNames,
    uint64_t* tensorIds,
    const uint32_t numOfTensors) {
  return syn_api->synTensorRetrieveIds(
      pRecipeHandle, tensorNames, tensorIds, numOfTensors);
}

synStatus SYN_API_CALL synSectionCreate(
    synSectionHandle* sectionHandle,
    uint64_t memoryAttributes,
    const synGraphHandle graph) {
  return syn_api->synSectionCreate(sectionHandle, memoryAttributes, graph);
}

synStatus SYN_API_CALL synSectionGetPersistent(
    synSectionHandle sectionHandle,
    bool* sectionIsPersistent) {
  return syn_api->synSectionGetPersistent(sectionHandle, sectionIsPersistent);
}

synStatus SYN_API_CALL synSectionSetPersistent(
    synSectionHandle sectionHandle,
    bool sectionIsPersistent) {
  return syn_api->synSectionSetPersistent(sectionHandle, sectionIsPersistent);
}

synStatus SYN_API_CALL
synSectionGetRMW(synSectionHandle sectionHandle, bool* sectionIsRMW) {
  return syn_api->synSectionGetRMW(sectionHandle, sectionIsRMW);
}

synStatus SYN_API_CALL
synSectionSetRMW(synSectionHandle sectionHandle, bool sectionIsRMW) {
  return syn_api->synSectionSetRMW(sectionHandle, sectionIsRMW);
}

synStatus SYN_API_CALL synSectionDestroy(synSectionHandle sectionHandle) {
  return syn_api->synSectionDestroy(sectionHandle);
}

synStatus SYN_API_CALL
synSectionSetConst(synSectionHandle sectionHandle, bool sectionIsConst) {
  return syn_api->synSectionSetConst(sectionHandle, sectionIsConst);
}

synStatus SYN_API_CALL synRecipeSectionHostBuffersClear(
    synRecipeHandle recipeHandle,
    const synSectionId* sectionIds,
    size_t numOfSections) {
  return syn_api->synRecipeSectionHostBuffersClear(
      recipeHandle, sectionIds, numOfSections);
}

synStatus SYN_API_CALL synRecipeSectionGetProp(
    const synRecipeHandle pRecipeHandle,
    const synSectionId sectionId,
    const synSectionProp prop,
    uint64_t* propertyPtr) {
  return syn_api->synRecipeSectionGetProp(
      pRecipeHandle, sectionId, prop, propertyPtr);
}

synStatus SYN_API_CALL synNodeCreate(
    const synGraphHandle graphHandle,
    const synTensor* pInputsTensorList,
    const synTensor* pOutputsTensorList,
    const uint32_t numberInputs,
    const uint32_t numberOutputs,
    const void* pUserParams,
    const unsigned paramsSize,
    const char* pGuid,
    const char* pName,
    const char** inputLayouts,
    const char** outputLayouts) {
  return syn_api->synNodeCreate(
      graphHandle,
      pInputsTensorList,
      pOutputsTensorList,
      numberInputs,
      numberOutputs,
      pUserParams,
      paramsSize,
      pGuid,
      pName,
      inputLayouts,
      outputLayouts);
}

synStatus SYN_API_CALL synNodeCreateWithId(
    const synGraphHandle graphHandle,
    const synTensor* pInputsTensorList,
    const synTensor* pOutputsTensorList,
    const uint32_t numberInputs,
    const uint32_t numberOutputs,
    const void* pUserParams,
    const unsigned paramsSize,
    const char* pGuid,
    const char* pName,
    synNodeId* nodeUniqueId,
    const char** inputLayouts,
    const char** outputLayouts) {
  return syn_api->synNodeCreateWithId(
      graphHandle,
      pInputsTensorList,
      pOutputsTensorList,
      numberInputs,
      numberOutputs,
      pUserParams,
      paramsSize,
      pGuid,
      pName,
      nodeUniqueId,
      inputLayouts,
      outputLayouts);
}

synStatus SYN_API_CALL synNodeSetDeterministic(
    const synGraphHandle graphHandle,
    const synNodeId nodeId,
    const bool useDeterministic) {
  return syn_api->synNodeSetDeterministic(
      graphHandle, nodeId, useDeterministic);
}

synStatus synNodeDependencySet(
    const synGraphHandle graphHandle,
    const synNodeId* pBlockingNodesIdList,
    const synNodeId* pBlockedNodesIdList,
    const uint32_t numberblocking,
    const uint32_t numberblocked) {
  return syn_api->synNodeDependencySet(
      graphHandle,
      pBlockingNodesIdList,
      pBlockedNodesIdList,
      numberblocking,
      numberblocked);
}

synStatus SYN_API_CALL synNodeSetUserProgrammability(
    const synGraphHandle graphHandle,
    const synNodeId nodeId,
    const synUserProgrammability* userProgrammability) {
  return syn_api->synNodeSetUserProgrammability(
      graphHandle, nodeId, userProgrammability);
}

synStatus SYN_API_CALL synGraphCompile(
    synRecipeHandle* pRecipeHandle,
    const synGraphHandle graphHandle,
    const char* pRecipeName,
    const char* pBuildLog) {
  return syn_api->synGraphCompile(
      pRecipeHandle, graphHandle, pRecipeName, pBuildLog);
}

synStatus SYN_API_CALL synGraphSetAttribute(
    synGraphHandle GraphHandle,
    const synGraphAttribute* attributes,
    const uint64_t* values,
    const uint32_t size) {
  return syn_api->synGraphSetAttribute(GraphHandle, attributes, values, size);
}

synStatus SYN_API_CALL synGraphGetAttribute(
    synGraphHandle GraphHandle,
    const synGraphAttribute* attributes,
    uint64_t* values,
    const uint32_t size) {
  return syn_api->synGraphGetAttribute(GraphHandle, attributes, values, size);
}

synStatus SYN_API_CALL
synGraphCreate(synGraphHandle* pGraphHandle, const synDeviceType deviceType) {
  return syn_api->synGraphCreate(pGraphHandle, deviceType);
}

synStatus SYN_API_CALL synGraphCreateEager(
    synGraphHandle* pGraphHandle,
    const synDeviceType deviceType) {
  return syn_api->synGraphCreateEager(pGraphHandle, deviceType);
}

synStatus SYN_API_CALL synGraphDuplicate(
    synGraphHandle graphHandle,
    synGraphHandle* newGraphHandle,
    synTensorHandleMap* tensorsMap,
    uint32_t* numTensors,
    synNodeHandleMap* nodesMap,
    uint32_t* numNodes) {
  return syn_api->synGraphDuplicate(
      graphHandle, newGraphHandle, tensorsMap, numTensors, nodesMap, numNodes);
}

synStatus SYN_API_CALL synGraphInferShapes(synGraphHandle graphHandle) {
  return syn_api->synGraphInferShapes(graphHandle);
}

synStatus SYN_API_CALL synGraphDestroy(const synGraphHandle graphHandle) {
  return syn_api->synGraphDestroy(graphHandle);
}

synStatus SYN_API_CALL synMemsetD32Async(
    uint64_t pDeviceMem,
    const uint32_t value,
    const size_t numOfElements,
    const synStreamHandle streamHandle) {
  return syn_api->synMemsetD32Async(
      pDeviceMem, value, numOfElements, streamHandle);
}

synStatus SYN_API_CALL synMemsetD8Async(
    uint64_t pDeviceMem,
    const unsigned char value,
    const size_t numOfElements,
    const synStreamHandle streamHandle) {
  return syn_api->synMemsetD8Async(
      pDeviceMem, value, numOfElements, streamHandle);
}

synStatus SYN_API_CALL synMemsetD16Async(
    uint64_t pDeviceMem,
    const uint16_t value,
    const size_t numOfElements,
    const synStreamHandle streamHandle) {
  return syn_api->synMemsetD16Async(
      pDeviceMem, value, numOfElements, streamHandle);
}

synStatus SYN_API_CALL synHostMalloc(
    const synDeviceId deviceId,
    const uint64_t size,
    const uint32_t flags,
    void** buffer) {
  return syn_api->synHostMalloc(deviceId, size, flags, buffer);
}

synStatus SYN_API_CALL synHostFree(
    const synDeviceId deviceId,
    const void* buffer,
    const uint32_t flags) {
  return syn_api->synHostFree(deviceId, buffer, flags);
}

synStatus SYN_API_CALL synHostMap(
    const synDeviceId deviceId,
    const uint64_t size,
    const void* buffer) {
  return syn_api->synHostMap(deviceId, size, buffer);
}

synStatus SYN_API_CALL
synHostUnmap(const synDeviceId deviceId, const void* buffer) {
  return syn_api->synHostUnmap(deviceId, buffer);
}

synStatus SYN_API_CALL synDeviceMalloc(
    const synDeviceId deviceId,
    const uint64_t size,
    uint64_t reqAddr,
    const uint32_t flags,
    uint64_t* buffer) {
  return syn_api->synDeviceMalloc(deviceId, size, reqAddr, flags, buffer);
}

synStatus SYN_API_CALL synDeviceFree(
    const synDeviceId deviceId,
    const uint64_t buffer,
    const uint32_t flags) {
  return syn_api->synDeviceFree(deviceId, buffer, flags);
}

synStatus SYN_API_CALL synDeviceRelease(const synDeviceId deviceId) {
  return syn_api->synDeviceRelease(deviceId);
}

synStatus SYN_API_CALL synDeviceGetMemoryInfo(
    const synDeviceId deviceId,
    uint64_t* free,
    uint64_t* total) {
  return syn_api->synDeviceGetMemoryInfo(deviceId, free, total);
}

synStatus SYN_API_CALL
synDeviceGetInfo(const synDeviceId deviceId, synDeviceInfo* pDeviceInfo) {
  return syn_api->synDeviceGetInfo(deviceId, pDeviceInfo);
}

synStatus SYN_API_CALL
synDeviceGetInfoV2(const synDeviceId deviceId, synDeviceInfoV2* pDeviceInfo) {
  return syn_api->synDeviceGetInfoV2(deviceId, pDeviceInfo);
}

synStatus SYN_API_CALL
synProfilerStart(const synTraceType type, const synDeviceId deviceId) {
  return syn_api->synProfilerStart(type, deviceId);
}

synStatus SYN_API_CALL
synProfilerStop(const synTraceType type, const synDeviceId deviceId) {
  return syn_api->synProfilerStop(type, deviceId);
}

synStatus SYN_API_CALL synProfilerGetTrace(
    const synTraceType type,
    const synDeviceId deviceId,
    const synTraceFormat format,
    void* buffer,
    size_t* size,
    size_t* numEntries) {
  return syn_api->synProfilerGetTrace(
      type, deviceId, format, buffer, size, numEntries);
}

synStatus SYN_API_CALL synProfilerQueryRequiredMemory(
    const synDeviceId deviceId,
    uint32_t* bytesRequired) {
  return syn_api->synProfilerQueryRequiredMemory(deviceId, bytesRequired);
}

synStatus SYN_API_CALL
synProfilerSetUserBuffer(const synDeviceId deviceId, void* userBuffer) {
  return syn_api->synProfilerSetUserBuffer(deviceId, userBuffer);
}

synStatus SYN_API_CALL synConfigurationSet(
    const char* configurationName,
    const char* configurationValue) {
  return syn_api->synConfigurationSet(configurationName, configurationValue);
}

synStatus SYN_API_CALL synConfigurationGet(
    const char* configurationName,
    char* configurationValue,
    uint64_t size) {
  return syn_api->synConfigurationGet(
      configurationName, configurationValue, size);
}

synStatus SYN_API_CALL synRecipeSerialize(
    const synRecipeHandle recipeHandle,
    const char* recipeFileName) {
  return syn_api->synRecipeSerialize(recipeHandle, recipeFileName);
}

synStatus SYN_API_CALL synRecipeDeSerialize(
    synRecipeHandle* pRecipeHandle,
    const char* recipeFileName) {
  return syn_api->synRecipeDeSerialize(pRecipeHandle, recipeFileName);
}

synStatus SYN_API_CALL synRecipeGetAttribute(
    uint64_t* retVal,
    const synRecipeAttribute* recipeAttr,
    const unsigned querySize,
    const synRecipeHandle recipeHandle) {
  return syn_api->synRecipeGetAttribute(
      retVal, recipeAttr, querySize, recipeHandle);
}

synStatus SYN_API_CALL synDeviceGetAttribute(
    uint64_t* retVal,
    const synDeviceAttribute* deviceAttr,
    const unsigned querySize,
    const synDeviceId deviceId) {
  return syn_api->synDeviceGetAttribute(
      retVal, deviceAttr, querySize, deviceId);
}

synStatus SYN_API_CALL synTensorHandleCreate(
    synTensor* tensor,
    synGraphHandle graph,
    synTensorType type,
    const char* tensorName) {
  return syn_api->synTensorHandleCreate(tensor, graph, type, tensorName);
}

synStatus SYN_API_CALL synProfilerGetCurrentTimeNS(uint64_t* nanoTime) {
  return syn_api->synProfilerGetCurrentTimeNS(nanoTime);
}

synStatus SYN_API_CALL
synProfilerAddCustomMeasurement(const char* description, uint64_t nanoTime) {
  return syn_api->synProfilerAddCustomMeasurement(description, nanoTime);
}

synStatus SYN_API_CALL synTensorAssignToSection(
    synTensor tensor,
    synSectionHandle section,
    uint64_t byteOffset) {
  return syn_api->synTensorAssignToSection(tensor, section, byteOffset);
}

synStatus SYN_API_CALL
synTensorSetSectionOffset(synTensor tensor, uint64_t byteOffset) {
  return syn_api->synTensorSetSectionOffset(tensor, byteOffset);
}

synStatus SYN_API_CALL synNodeGetUserParams(
    const synGraphHandle graphHandle,
    const synNodeId nodeId,
    void* userParams,
    unsigned* paramsSize) {
  return syn_api->synNodeGetUserParams(
      graphHandle, nodeId, userParams, paramsSize);
}

synStatus SYN_API_CALL synNodeSetUserParams(
    const synGraphHandle graphHandle,
    const synNodeId nodeId,
    const void* userParams,
    const unsigned paramsSize) {
  return syn_api->synNodeSetUserParams(
      graphHandle, nodeId, userParams, paramsSize);
}

synStatus SYN_API_CALL synTensorSetHostPtr(
    synTensor tensor,
    void* hostPtr,
    uint64_t size,
    synDataType dataType,
    bool copyBuffer) {
  return syn_api->synTensorSetHostPtr(
      tensor, hostPtr, size, dataType, copyBuffer);
}

synStatus SYN_API_CALL synTensorGetHostPtr(
    synTensor tensor,
    void** hostPtr,
    uint64_t* size,
    synDataType* dataType) {
  return syn_api->synTensorGetHostPtr(tensor, hostPtr, size, dataType);
}

synStatus SYN_API_CALL synConstTensorCreate(
    synTensor* pTensor,
    const synTensorDescriptor* descriptor) {
  return syn_api->synConstTensorCreate(pTensor, descriptor);
}

synStatus SYN_API_CALL synTensorCreate(
    synTensor* pTensor,
    const synTensorDescriptor* descriptor,
    const synSectionHandle pSectionHandle,
    const uint64_t sectionOffset) {
  return syn_api->synTensorCreate(
      pTensor, descriptor, pSectionHandle, sectionOffset);
}

synStatus SYN_API_CALL synTensorSetPermutation(
    synTensor tensor,
    const synTensorPermutation* permutation) {
  return syn_api->synTensorSetPermutation(tensor, permutation);
}

synStatus SYN_API_CALL
synTensorSetAllowPermutation(synTensor tensor, int8_t allowPermutation) {
  return syn_api->synTensorSetAllowPermutation(tensor, allowPermutation);
}

synStatus SYN_API_CALL
synTensorGetName(const synTensor tensor, const uint64_t size, char* name) {
  return syn_api->synTensorGetName(tensor, size, name);
}

synStatus SYN_API_CALL synTensorRetrieveLaunchAmount(
    const synRecipeHandle pRecipeHandle,
    uint32_t* numOfTensors) {
  return syn_api->synTensorRetrieveLaunchAmount(pRecipeHandle, numOfTensors);
}

synStatus SYN_API_CALL synTensorRetrieveLaunchIds(
    const synRecipeHandle pRecipeHandle,
    uint64_t* tensorsIds,
    const uint32_t numOfTensors) {
  return syn_api->synTensorRetrieveLaunchIds(
      pRecipeHandle, tensorsIds, numOfTensors);
}

synStatus SYN_API_CALL synTensorRetrieveLaunchInfoById(
    const synRecipeHandle pRecipeHandle,
    const uint32_t numOfTensors,
    synRetrievedLaunchTensorInfo* tensorsLaunchInfo) {
  return syn_api->synTensorRetrieveLaunchInfoById(
      pRecipeHandle, numOfTensors, tensorsLaunchInfo);
}

synStatus SYN_API_CALL synTensorGetGeometry(
    const synTensor tensor,
    synTensorGeometry* geometry,
    synGeometryType geometryType) {
  return syn_api->synTensorGetGeometry(tensor, geometry, geometryType);
}

synStatus SYN_API_CALL synTensorSetGeometry(
    synTensor tensor,
    const synTensorGeometry* geometry,
    synGeometryType geometryType) {
  return syn_api->synTensorSetGeometry(tensor, geometry, geometryType);
}

synStatus SYN_API_CALL synTensorSetDeviceFullLayout(
    synTensor tensor,
    const synTensorDeviceFullLayout* layout) {
  return syn_api->synTensorSetDeviceFullLayout(tensor, layout);
}

synStatus SYN_API_CALL
synTensorSetDeviceDataType(synTensor tensor, synDataType deviceDataType) {
  return syn_api->synTensorSetDeviceDataType(tensor, deviceDataType);
}

synStatus SYN_API_CALL synTensorSetQuantizationData(
    synTensor tensor,
    synQuantizationProperty prop,
    void* propVal,
    uint64_t propSize) {
  return syn_api->synTensorSetQuantizationData(tensor, prop, propVal, propSize);
}

synStatus SYN_API_CALL synTensorExtExtractExecutionOrder(
    const synRecipeHandle recipeHandle,
    uint32_t numOfExternalTensors,
    uint64_t* tensorIds) {
  if (UsePartialEventEmulation()) {
    auto& partial_event_emulation = PartialEventEmulation::Instance();
    return partial_event_emulation.synTensorExtExtractExecutionOrder(
        recipeHandle, numOfExternalTensors, tensorIds);
  } else
    return syn_api->synTensorExtExtractExecutionOrder(
        recipeHandle, numOfExternalTensors, tensorIds);
}

synStatus SYN_API_CALL synEventMapTensor(
    synEventHandle* eventHandle,
    size_t numOfEvents,
    const synLaunchTensorInfo* launchTensorsInfo,
    const synRecipeHandle recipeHandle) {
  if (UsePartialEventEmulation()) {
    auto& partial_event_emulation = PartialEventEmulation::Instance();
    return partial_event_emulation.synEventMapTensor(
        eventHandle, numOfEvents, launchTensorsInfo, recipeHandle);
  } else
    return syn_api->synEventMapTensor(
        eventHandle, numOfEvents, launchTensorsInfo, recipeHandle);
}

synStatus SYN_API_CALL synLaunchWithExternalEvents(
    const synStreamHandle streamHandle,
    const synLaunchTensorInfo* launchTensorsInfo,
    const uint32_t numberOfTensors,
    uint64_t pWorkspace,
    const synRecipeHandle pRecipeHandle,
    synEventHandle* eventHandleList,
    const uint32_t numberOfEvents,
    uint32_t flags) {
  if (UsePartialEventEmulation()) {
    auto& partial_event_emulation = PartialEventEmulation::Instance();
    return partial_event_emulation.synLaunchWithExternalEvents(
        streamHandle,
        launchTensorsInfo,
        numberOfTensors,
        pWorkspace,
        pRecipeHandle,
        eventHandleList,
        numberOfEvents,
        flags);
  } else
    return syn_api->synLaunchWithExternalEvents(
        streamHandle,
        launchTensorsInfo,
        numberOfTensors,
        pWorkspace,
        pRecipeHandle,
        eventHandleList,
        numberOfEvents,
        flags);
}

synStatus SYN_API_CALL synTensorSetExternal(synTensor tensor, bool isExternal) {
  if (UsePartialEventEmulation()) {
    auto& partial_event_emulation = PartialEventEmulation::Instance();
    return partial_event_emulation.synTensorSetExternal(tensor, isExternal);
  } else
    return syn_api->synTensorSetExternal(tensor, isExternal);
}

synStatus SYN_API_CALL
synTensorGetExternal(const synTensor tensor, bool* isExternal) {
  if (UsePartialEventEmulation()) {
    auto& partial_event_emulation = PartialEventEmulation::Instance();
    return partial_event_emulation.synTensorGetExternal(tensor, isExternal);
  } else
    return syn_api->synTensorGetExternal(tensor, isExternal);
}

synStatus SYN_API_CALL synStatusGetBriefDescription(
    synStatus status,
    char* statusDescription,
    size_t len) {
  return syn_api->synStatusGetBriefDescription(status, statusDescription, len);
}

synStatus SYN_API_CALL
synDumpStateAndTerminate(const char* msg, uint64_t flags) {
  return syn_api->synDumpStateAndTerminate(msg, flags);
}
