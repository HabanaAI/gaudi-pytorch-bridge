/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <dlfcn.h>
#include <synapse_api.h> // IWYU pragma: keep
#include <synapse_api_types.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arg_utils.h"
#include "object_dump.h"
#include "synapse_common_types.h"
#include "synapse_logger.h"

#define CALL_SYN_FUNC(func, ...)                          \
  if (synapse_logger::logger.should_use_null_backend()) { \
    status = synSuccess;                                  \
  } else {                                                \
    status = func(__VA_ARGS__);                           \
  }

#define CALL_SYN_FUNC_SET_VAL(func, val, ...)             \
  if (synapse_logger::logger.should_use_null_backend()) { \
    val = 1;                                              \
    status = synSuccess;                                  \
  } else {                                                \
    status = func(__VA_ARGS__);                           \
  }
#define LOG_TRACE(x, y, z)
#define SYN_API_PTR(func) decltype(::func)* func
#define SYN_API_INIT_PTR(func) \
  CHECK_NULL(func = (decltype(func))dlsym(lib_handle, #func))
namespace lib_synapse {
SYN_API_PTR(synDeviceSynchronize);
SYN_API_PTR(synStreamCreate);
SYN_API_PTR(synStreamDestroy);
SYN_API_PTR(synStreamWaitEvent);
SYN_API_PTR(synStreamSynchronize);
SYN_API_PTR(synStreamQuery);
SYN_API_PTR(synEventCreate);
SYN_API_PTR(synEventDestroy);
SYN_API_PTR(synEventMapTensorExt);
SYN_API_PTR(synEventRecord);
SYN_API_PTR(synEventQuery);
SYN_API_PTR(synEventSynchronize);
SYN_API_PTR(synEventElapsedTime);
SYN_API_PTR(synLaunchExt);
SYN_API_PTR(synLaunchWithExternalEventsExt);
SYN_API_PTR(synWorkspaceGetSize);
SYN_API_PTR(synMemCopyAsync);
SYN_API_PTR(synMemCopyAsyncMultiple);
SYN_API_PTR(synDeviceGetCount);
SYN_API_PTR(synDeviceGetCountByDeviceType);
SYN_API_PTR(synDeviceAcquireByDeviceType);
SYN_API_PTR(synDeviceAcquire);
SYN_API_PTR(synDriverGetVersion);
SYN_API_PTR(synDeviceGetName);
SYN_API_PTR(synTensorRetrieveIds);
SYN_API_PTR(synTensorCreate);
SYN_API_PTR(synTensorDestroy);
SYN_API_PTR(synConstTensorCreate);
SYN_API_PTR(synNodeCreate);
SYN_API_PTR(synNodeCreateWithId);
SYN_API_PTR(synNodeDependencySet);
SYN_API_PTR(synGraphCompile);
SYN_API_PTR(synGraphCreate);
SYN_API_PTR(synGraphCreateEager);
SYN_API_PTR(synGraphDestroy);
SYN_API_PTR(synMemsetD32Async);
SYN_API_PTR(synMemsetD8Async);
SYN_API_PTR(synMemsetD16Async);
SYN_API_PTR(synHostMalloc);
SYN_API_PTR(synHostFree);
SYN_API_PTR(synHostMap);
SYN_API_PTR(synHostUnmap);
SYN_API_PTR(synDeviceMalloc);
SYN_API_PTR(synDeviceFree);
SYN_API_PTR(synDeviceGetAttribute);
SYN_API_PTR(synInitialize);
SYN_API_PTR(synDestroy);
SYN_API_PTR(synDeviceRelease);
SYN_API_PTR(synDeviceGetMemoryInfo);
SYN_API_PTR(synDeviceGetInfo);
SYN_API_PTR(synProfilerStart);
SYN_API_PTR(synProfilerStop);
SYN_API_PTR(synProfilerGetTrace);
SYN_API_PTR(synConfigurationSet);
SYN_API_PTR(synConfigurationGet);
SYN_API_PTR(synSectionCreate);
SYN_API_PTR(synSectionDestroy);
SYN_API_PTR(synTensorAssignToSection);
SYN_API_PTR(synTensorHandleCreate);
SYN_API_PTR(synTensorGetName);
SYN_API_PTR(synTensorSetExternal);
SYN_API_PTR(synTensorRetrieveLaunchAmount);
SYN_API_PTR(synTensorRetrieveLaunchIds);
SYN_API_PTR(synTensorRetrieveLaunchInfoById);
SYN_API_PTR(synTensorRetrieveLaunchInfoByIdExt);
SYN_API_PTR(synTensorSetGeometryExt);
SYN_API_PTR(synTensorSetDeviceDataType);
SYN_API_PTR(synTensorSetHostPtr);
SYN_API_PTR(synTensorSetPermutation);
SYN_API_PTR(synSectionSetConst);
SYN_API_PTR(synSectionSetGroup);
SYN_API_PTR(synRecipeSectionGetProp);

void LoadSymbols(void* lib_handle) {
  SYN_API_INIT_PTR(synDeviceSynchronize);
  SYN_API_INIT_PTR(synStreamCreate);
  SYN_API_INIT_PTR(synStreamDestroy);
  SYN_API_INIT_PTR(synStreamWaitEvent);
  SYN_API_INIT_PTR(synStreamSynchronize);
  SYN_API_INIT_PTR(synStreamQuery);
  SYN_API_INIT_PTR(synEventCreate);
  SYN_API_INIT_PTR(synEventDestroy);
  SYN_API_INIT_PTR(synEventMapTensorExt);
  SYN_API_INIT_PTR(synEventRecord);
  SYN_API_INIT_PTR(synEventQuery);
  SYN_API_INIT_PTR(synEventSynchronize);
  SYN_API_INIT_PTR(synEventElapsedTime);
  SYN_API_INIT_PTR(synLaunchExt);
  SYN_API_INIT_PTR(synLaunchWithExternalEventsExt);
  SYN_API_INIT_PTR(synWorkspaceGetSize);
  SYN_API_INIT_PTR(synMemCopyAsync);
  SYN_API_INIT_PTR(synMemCopyAsyncMultiple);
  SYN_API_INIT_PTR(synDeviceGetCount);
  SYN_API_INIT_PTR(synDeviceGetCountByDeviceType);
  SYN_API_INIT_PTR(synDeviceAcquireByDeviceType);
  SYN_API_INIT_PTR(synDeviceAcquire);
  SYN_API_INIT_PTR(synDriverGetVersion);
  SYN_API_INIT_PTR(synDeviceGetName);
  SYN_API_INIT_PTR(synTensorRetrieveIds);
  SYN_API_INIT_PTR(synTensorCreate);
  SYN_API_INIT_PTR(synTensorDestroy);
  SYN_API_INIT_PTR(synConstTensorCreate);
  SYN_API_INIT_PTR(synNodeCreate);
  SYN_API_INIT_PTR(synNodeCreateWithId);
  SYN_API_INIT_PTR(synNodeDependencySet);
  SYN_API_INIT_PTR(synGraphCompile);
  SYN_API_INIT_PTR(synGraphCreate);
  SYN_API_INIT_PTR(synGraphCreateEager);
  SYN_API_INIT_PTR(synGraphDestroy);
  SYN_API_INIT_PTR(synMemsetD32Async);
  SYN_API_INIT_PTR(synMemsetD8Async);
  SYN_API_INIT_PTR(synMemsetD16Async);
  SYN_API_INIT_PTR(synHostMalloc);
  SYN_API_INIT_PTR(synHostFree);
  SYN_API_INIT_PTR(synHostMap);
  SYN_API_INIT_PTR(synHostUnmap);
  SYN_API_INIT_PTR(synDeviceMalloc);
  SYN_API_INIT_PTR(synDeviceFree);
  SYN_API_INIT_PTR(synDeviceGetAttribute);
  SYN_API_INIT_PTR(synInitialize);
  SYN_API_INIT_PTR(synDestroy);
  SYN_API_INIT_PTR(synDeviceRelease);
  SYN_API_INIT_PTR(synDeviceGetMemoryInfo);
  SYN_API_INIT_PTR(synDeviceGetInfo);
  SYN_API_INIT_PTR(synProfilerStart);
  SYN_API_INIT_PTR(synProfilerStop);
  SYN_API_INIT_PTR(synProfilerGetTrace);
  SYN_API_INIT_PTR(synConfigurationSet);
  SYN_API_INIT_PTR(synConfigurationGet);
  SYN_API_INIT_PTR(synSectionCreate);
  SYN_API_INIT_PTR(synSectionDestroy);
  SYN_API_INIT_PTR(synTensorAssignToSection);
  SYN_API_INIT_PTR(synTensorHandleCreate);
  SYN_API_INIT_PTR(synTensorGetName);
  SYN_API_INIT_PTR(synTensorSetExternal);
  SYN_API_INIT_PTR(synTensorRetrieveLaunchAmount);
  SYN_API_INIT_PTR(synTensorRetrieveLaunchIds);
  SYN_API_INIT_PTR(synTensorRetrieveLaunchInfoById);
  SYN_API_INIT_PTR(synTensorRetrieveLaunchInfoByIdExt);
  SYN_API_INIT_PTR(synTensorSetGeometryExt);
  SYN_API_INIT_PTR(synTensorSetDeviceDataType);
  SYN_API_INIT_PTR(synTensorSetHostPtr);
  SYN_API_INIT_PTR(synTensorSetPermutation);
  SYN_API_INIT_PTR(synSectionSetConst);
  SYN_API_INIT_PTR(synSectionSetGroup);
  SYN_API_INIT_PTR(synRecipeSectionGetProp);
}

} // namespace lib_synapse

namespace synapse_logger {

class LogStreamName {
 public:
  const std::string get(const synStreamType sType) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return streamNameMap.at(sType);
  }
  const std::string get(const synStreamHandle sHandle) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_streamNameFromStreamHandleMap.find(sHandle) !=
        m_streamNameFromStreamHandleMap.end()) {
      return m_streamNameFromStreamHandleMap.at(sHandle);
    } else {
      return "\"NULL\"";
    }
  }
  const std::string get(const synEventHandle eHandle) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_streamNameFromEventHandleMap.find(eHandle) !=
        m_streamNameFromEventHandleMap.end()) {
      return m_streamNameFromEventHandleMap.at(eHandle);
    } else {
      return "\"NULL\"";
    }
  }
  synStreamHandle getHandle(const synEventHandle eHandle) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_streamHandleFromEventHandleMap.find(eHandle) !=
        m_streamHandleFromEventHandleMap.end()) {
      return m_streamHandleFromEventHandleMap.at(eHandle);
    } else {
      return NULL;
    }
  }

  void link(synStreamHandle sHandle, const std::string streamName) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_streamNameFromStreamHandleMap[sHandle] = streamName;
  }
  void link(synEventHandle eHandle, const std::string streamName) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_streamNameFromEventHandleMap[eHandle] = streamName;
  }
  void link(synEventHandle eHandle, const synStreamHandle sHandle) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_streamHandleFromEventHandleMap[eHandle] = sHandle;
  }

 private:
  std::map<synStreamType, const std::string> streamNameMap = {
      {STREAM_TYPE_COPY_DEVICE_TO_HOST, "\"STREAM_TYPE_COPY_DEVICE_TO_HOST\""},
      {STREAM_TYPE_COPY_HOST_TO_DEVICE, "\"STREAM_TYPE_COPY_HOST_TO_DEVICE\""},
      {STREAM_TYPE_COPY_DEVICE_TO_DEVICE,
       "\"STREAM_TYPE_COPY_DEVICE_TO_DEVICE\""},
      {STREAM_TYPE_COMPUTE, "\"STREAM_TYPE_COMPUTE\""},
      {STREAM_TYPE_NETWORK_COLLECTIVE, "\"STREAM_TYPE_NETWORK_COLLECTIVE\""},
      {STREAM_TYPE_MAX_USER_TYPES, "\"STREAM_TYPE_RESERVED_1\""},
      {STREAM_TYPE_MAX, "\"STREAM_TYPE_MAX\""}};

  std::mutex m_mutex;
  std::map<synStreamHandle, std::string> m_streamNameFromStreamHandleMap;
  std::map<synEventHandle, std::string> m_streamNameFromEventHandleMap;
  std::map<synEventHandle, synStreamHandle> m_streamHandleFromEventHandleMap;
};

} // namespace synapse_logger

synapse_logger::LogStreamName strLog;

synStatus synInitialize() {
  API_LOG_CALL();
  // Directly calling synInitialize() due to null backend functionality
  synStatus status = lib_synapse::synInitialize();
  API_LOG_RESULT();
  return status;
}

synStatus synDeviceSynchronize(const synDeviceId deviceId) {
  API_LOG_CALL(ARG(deviceId));
  synStatus status = lib_synapse::synDeviceSynchronize(deviceId);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synStreamCreate(
    synStreamHandle* pStreamHandle,
    const synDeviceId deviceId,
    const synStreamType streamType,
    const uint32_t flags) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamType);
  API_LOG_CALL(
      ARG(pStreamHandle),
      ARG(deviceId),
      ARG(streamType),
      ARG(flags),
      ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synStreamCreate, pStreamHandle, deviceId, streamType, flags)
  API_LOG_RESULT(S_ARG(pStreamHandle));
  strLog.link(pStreamHandle[0], streamName);

  return status;
}

synStatus SYN_API_CALL synStreamDestroy(const synStreamHandle streamHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(ARG(streamHandle), ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synStreamDestroy, streamHandle)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synStreamWaitEvent(
    const synStreamHandle streamHandle,
    synEventHandle eventHandle,
    const uint32_t flags) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(
      ARG(streamHandle), ARG(eventHandle), ARG(flags), ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synStreamWaitEvent, streamHandle, eventHandle, flags)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synStreamSynchronize(const synStreamHandle streamHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(ARG(streamHandle), ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synStreamSynchronize, streamHandle)
  API_LOG_RESULT();
  synapse_logger::logger.stream_synchronized(streamHandle);
  return status;
}

synStatus SYN_API_CALL synStreamQuery(const synStreamHandle streamHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(ARG(streamHandle), ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synStreamQuery, streamHandle)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synEventCreate(
    synEventHandle* pEventHandle,
    const synDeviceId deviceId,
    const uint32_t flags) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pEventHandle), ARG(deviceId), ARG_Q(flags));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synEventCreate, pEventHandle, deviceId, flags)
  API_LOG_RESULT(S_ARG(pEventHandle));
  return status;
}

synStatus SYN_API_CALL synEventDestroy(synEventHandle eventHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(eventHandle));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synEventDestroy, eventHandle)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synEventMapTensor(
    synEventHandle* eventHandle,
    size_t numOfEvents,
    const synLaunchTensorInfoExt* launchTensorsInfo,
    const synRecipeHandle recipeHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(
      ARG_X(eventHandle),
      ARG(numOfEvents),
      ARG_X(launchTensorsInfo),
      ARG(recipeHandle));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synEventMapTensorExt,
      eventHandle,
      numOfEvents,
      launchTensorsInfo,
      recipeHandle)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synEventRecord(synEventHandle eventHandle, const synStreamHandle streamHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(ARG(eventHandle), ARG(streamHandle), ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synEventRecord, eventHandle, streamHandle)
  API_LOG_RESULT();
  synapse_logger::logger.event_recorded(streamHandle, eventHandle);
  strLog.link(eventHandle, streamName);
  strLog.link(eventHandle, streamHandle);

  return status;
}

synStatus SYN_API_CALL synEventQuery(const synEventHandle eventHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(eventHandle));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synEventQuery, eventHandle)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synEventSynchronize(const synEventHandle eventHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(eventHandle);
  const synStreamHandle streamHandle = strLog.getHandle(eventHandle);
  API_LOG_CALL(ARG(eventHandle), ARG(streamHandle), ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synEventSynchronize, eventHandle)
  API_LOG_RESULT();
  synapse_logger::logger.event_synchronized(eventHandle);
  return status;
}

synStatus SYN_API_CALL synEventElapsedTime(
    uint64_t* pNanoSeconds,
    const synEventHandle eventHandleStart,
    const synEventHandle eventHandleEnd) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG_Q(pNanoSeconds), ARG(eventHandleStart), ARG(eventHandleEnd));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synEventElapsedTime,
      pNanoSeconds,
      eventHandleStart,
      eventHandleEnd)
  API_LOG_RESULT();
  return status;
}

inline std::ostream& operator<<(
    std::ostream& out,
    const synLaunchTensorInfoExt& v) {
  return out << '"' << (v.tensorName ? v.tensorName : "nullptr") << "\", \""
             << v.tensorId << "\", \"" << (void*)v.pTensorAddress << '"';
}

synStatus SYN_API_CALL synLaunchExt(
    const synStreamHandle streamHandle,
    const synLaunchTensorInfoExt* launchTensorsInfo,
    uint32_t numberTensors,
    uint64_t pWorkspace,
    const synRecipeHandle pRecipeHandle,
    uint32_t flags) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(
      ARG(streamHandle),
      ARG(launchTensorsInfo),
      ARG(numberTensors),
      ARG(numberTensors),
      ARG_X(pWorkspace),
      ARG(pRecipeHandle),
      ARG(flags),
      ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synLaunchExt,
      streamHandle,
      launchTensorsInfo,
      numberTensors,
      pWorkspace,
      pRecipeHandle,
      flags)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synLaunchWithExternalEvents(
    const synStreamHandle streamHandle,
    const synLaunchTensorInfoExt* launchTensorsInfo,
    uint32_t numberOfTensors,
    uint64_t pWorkspace,
    const synRecipeHandle pRecipeHandle,
    synEventHandle* eventHandleList,
    const uint32_t numberOfEvents,
    uint32_t flags) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(
      ARG(streamHandle),
      ARG(launchTensorsInfo),
      ARG(numberOfTensors),
      ARG_X(pWorkspace),
      ARG(pRecipeHandle),
      ARG(eventHandleList),
      ARG(numberOfEvents),
      ARG(flags),
      ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synLaunchWithExternalEventsExt,
      streamHandle,
      launchTensorsInfo,
      numberOfTensors,
      pWorkspace,
      pRecipeHandle,
      eventHandleList,
      numberOfEvents,
      flags)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synWorkspaceGetSize(
    uint64_t* pWorkspaceSize,
    const synRecipeHandle recipeHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pWorkspaceSize), ARG(recipeHandle));
  synStatus status;
  CALL_SYN_FUNC_SET_VAL(
      lib_synapse::synWorkspaceGetSize,
      *pWorkspaceSize,
      pWorkspaceSize,
      recipeHandle)
  API_LOG_RESULT(S_ARG_X(pWorkspaceSize));
  return status;
}

synStatus SYN_API_CALL synMemCopyAsync(
    const synStreamHandle streamHandle,
    const uint64_t src,
    const uint64_t size,
    const uint64_t dst,
    const synDmaDir direction) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  switch (direction) {
    case HOST_TO_DRAM:
      synapse_logger::logger.dump_host_data(reinterpret_cast<void*>(src), size);
      break;
    case DRAM_TO_HOST:
      synapse_logger::logger.store_transfer_to_host(
          streamHandle, src, size, dst);
      break;
    default:
      break;
  }

  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(
      ARG(streamHandle),
      ARG_X(src),
      ARG_X(size),
      ARG_X(dst),
      ARG(direction),
      ARG(streamName));

  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synMemCopyAsync, streamHandle, src, size, dst, direction)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synMemCopyAsyncMultiple(
    const synStreamHandle streamHandle,
    const uint64_t* src,
    const uint64_t* size,
    const uint64_t* dst,
    const synDmaDir direction,
    const size_t numCopies) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  switch (direction) {
    case HOST_TO_DRAM:
      for (std::size_t i = 0; i < numCopies; ++i) {
        synapse_logger::logger.dump_host_data(
            reinterpret_cast<void*>(src[i]), size[i]);
      }
      break;
    case DRAM_TO_HOST:
      for (std::size_t i = 0; i < numCopies; ++i) {
        synapse_logger::logger.store_transfer_to_host(
            streamHandle, src[i], size[i], dst[i]);
      }
      break;
    default:
      break;
  }

  const std::string streamName = strLog.get(streamHandle);
  API_LOG_CALL(
      ARG(streamHandle),
      M_ARG_X(src, numCopies),
      M_ARG_X(size, numCopies),
      M_ARG_X(dst, numCopies),
      ARG(direction),
      ARG(numCopies),
      ARG(streamName));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synMemCopyAsyncMultiple,
      streamHandle,
      src,
      size,
      dst,
      direction,
      numCopies)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synDeviceGetCount(uint32_t* pCount) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pCount));
  synStatus status = lib_synapse::synDeviceGetCount(pCount);
  API_LOG_RESULT(S_ARG(pCount));
  return status;
}

synStatus SYN_API_CALL synDeviceGetCountByDeviceType(
    uint32_t* pCount,
    const synDeviceType deviceType) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pCount), ARG(deviceType));
  synStatus status =
      lib_synapse::synDeviceGetCountByDeviceType(pCount, deviceType);
  API_LOG_RESULT(S_ARG(pCount));
  return status;
}

synStatus SYN_API_CALL synDeviceAcquireByDeviceType(
    synDeviceId* pDeviceId,
    const synDeviceType deviceType) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG(pDeviceId), ARG(deviceType));
  synStatus status =
      lib_synapse::synDeviceAcquireByDeviceType(pDeviceId, deviceType);
  API_LOG_RESULT(S_ARG(pDeviceId));
  synapse_logger::logger.last_acquired_id(*pDeviceId);
  return status;
}

synStatus SYN_API_CALL
synDeviceAcquire(synDeviceId* pDeviceId, const char* pciBus) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pDeviceId), ARG_Q(pciBus));
  synStatus status = lib_synapse::synDeviceAcquire(pDeviceId, pciBus);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synDriverGetVersion(char* pDriverVersion, const int len) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG_Q(pDriverVersion), ARG(len));
  synStatus status = lib_synapse::synDriverGetVersion(pDriverVersion, len);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synDeviceGetName(char* pName, const int len, const synDeviceId deviceId) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG_Q(pName), ARG(len), ARG(deviceId));
  synStatus status = lib_synapse::synDeviceGetName(pName, len, deviceId);
  API_LOG_RESULT(ARG_Q(pName));
  return status;
}

inline void log_synTensorDescriptor(const synTensorDescriptor* obj) {
  if (!logger_is_enabled(
          synapse_logger::data_dump_category::SYNAPSE_API_CALL) ||
      obj == nullptr) {
    return;
  }

  synapse_logger::ostr_t out{synapse_logger::get_ostr()};
  out << R"("name":"object", "args":{"at":")" << (void*)obj
      << R"(", "type":"synTensorDescriptor", "fields":{)"
      << R"("m_dataType":)" << obj->m_dataType << R"(, "m_dims":)"
      << obj->m_dims << R"(, "m_sizes":[)"
      << absl::Span<const unsigned>(obj->m_sizes) << R"(], "m_ptr":")"
      << obj->m_ptr << R"(", "isWeights":)" << obj->m_isWeights
      << R"(, "m_name":")" << (obj->m_name ? obj->m_name : "nullptr") << R"(")"
      << "}}";
  synapse_logger::log(out.str());
}

inline void log_synConstTensorDescriptor(const synTensorDescriptor* obj) {
  if (!logger_is_enabled(
          synapse_logger::data_dump_category::SYNAPSE_API_CALL) ||
      obj == nullptr) {
    return;
  }

  auto dataSize = synapse_logger::size_of_syn_data_type(obj->m_dataType);
  for (auto i = 0UL; i < obj->m_dims; i++) {
    dataSize *= obj->m_sizes[i];
  }
  synapse_logger::ostr_t out{synapse_logger::get_ostr()};
  out << R"("name":"object", "args":{"at":")" << (void*)obj
      << R"(", "type":"synTensorDescriptor", "fields":{)"
      << R"("m_dataType":)" << obj->m_dataType << R"(, "m_dims":)"
      << obj->m_dims << R"(, "m_sizes":[)"
      << absl::Span<const unsigned>(obj->m_sizes) << R"(], "m_ptr":")"
      << obj->m_ptr << R"(", "isWeights":)" << obj->m_isWeights
      << R"(, "m_name":")" << (obj->m_name ? obj->m_name : "nullptr") << R"(")"
      << R"(, "m_batchPos":)" << obj->m_batchPos << R"(, "m_isQuantized":)"
      << obj->m_isQuantized << "}"
      << R"(, "const":1,)";
  if (synapse_logger::logger.is_enabled(
          synapse_logger::data_dump_category::CONST_TENSOR_DATA)) {
    auto offset = synapse_logger::logger.dump_data(obj->m_ptr, dataSize);
    out << R"("data_offset":)" << offset << ",";
  }
  out << R"("byte_size":)" << dataSize << "}";
  synapse_logger::log(out.str());
}

inline void log_synTensorSetGeometry(
    const synTensorGeometryExt* obj,
    synTensor& tensor) {
  if (!logger_is_enabled(
          synapse_logger::data_dump_category::SYNAPSE_API_CALL) ||
      obj == nullptr) {
    return;
  }

  synapse_logger::ostr_t out{synapse_logger::get_ostr()};
  out << R"("name":"object", "args":{"at":")" << (void*)obj << R"(", "tensor":)"
      << tensor << R"(, "type":"synTensorGeometryExt", "fields":{)"
      << R"( "m_sizes":[)"
      << absl::Span<const tensor_size_t>(obj->sizes, HABANA_DIM_MAX)
      << R"(], "m_dims":)" << obj->dims << "}}";
  synapse_logger::log(out.str());
}

synStatus SYN_API_CALL synTensorCreate(
    synTensor* pTensor,
    const synTensorDescriptor* descriptor,
    const synSectionHandle pSectionHandle,
    const uint64_t sectionOffset) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  log_synTensorDescriptor(descriptor);
  API_LOG_CALL(
      ARG(pTensor), ARG(descriptor), ARG(pSectionHandle), ARG(sectionOffset));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorCreate,
      pTensor,
      descriptor,
      pSectionHandle,
      sectionOffset)
  API_LOG_RESULT(S_ARG(pTensor));
  return status;
}

synStatus SYN_API_CALL synConstTensorCreate(
    synTensor* pTensor,
    const synTensorDescriptor* descriptor) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  log_synConstTensorDescriptor(descriptor);
  API_LOG_CALL(ARG(pTensor), ARG(descriptor));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synConstTensorCreate, pTensor, descriptor)
  API_LOG_RESULT(S_ARG(pTensor));
  return status;
}

synStatus SYN_API_CALL synTensorRetrieveIds(
    const synRecipeHandle recipeHandle,
    const char** tensorNames,
    uint64_t* tensorIds,
    const uint32_t numOfTensors) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(
      ARG(recipeHandle),
      ARG(tensorNames),
      ARG(numOfTensors),
      ARG(tensorIds),
      ARG(numOfTensors));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorRetrieveIds,
      recipeHandle,
      tensorNames,
      tensorIds,
      numOfTensors);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorDestroy(const synTensor tensor) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(tensor));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synTensorDestroy, tensor)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synSectionCreate(
    synSectionHandle* sectionHandle,
    uint64_t memoryAttributes,
    const synGraphHandle graph) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(sectionHandle), ARG_Q(memoryAttributes), ARG(graph));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synSectionCreate, sectionHandle, memoryAttributes, graph)
  API_LOG_RESULT(S_ARG(sectionHandle));
  return status;
}

synStatus SYN_API_CALL synSectionDestroy(synSectionHandle sectionHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(sectionHandle));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synSectionDestroy, sectionHandle)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorAssignToSection(
    synTensor tensor,
    synSectionHandle section,
    uint64_t byteOffset) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(tensor), ARG(section), ARG(byteOffset));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorAssignToSection, tensor, section, byteOffset)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synSectionSetConst(synSectionHandle sectionHandle, bool sectionIsConst) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(sectionHandle), ARG(sectionIsConst));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synSectionSetConst, sectionHandle, sectionIsConst)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synSectionSetGroup(synSectionHandle sectionHandle, uint64_t sectionGroup) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(sectionHandle), ARG(sectionGroup));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synSectionSetGroup, sectionHandle, sectionGroup)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synRecipeSectionGetProp(
    const synRecipeHandle pRecipeHandle,
    const synSectionId sectionId,
    const synSectionProp prop,
    uint64_t* propertyPtr) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pRecipeHandle), ARG(sectionId), ARG(prop), ARG(propertyPtr));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synRecipeSectionGetProp,
      pRecipeHandle,
      sectionId,
      prop,
      propertyPtr)
  API_LOG_RESULT();
  return status;
}

namespace synapse_logger {

template <>
inline void dump_object<std::vector<TransposePermutationDim>>(
    const std::vector<TransposePermutationDim>* obj,
    UNUSED unsigned count) {
  if (!logger_is_enabled(data_dump_category::SYNAPSE_API_CALL) ||
      obj == nullptr) {
    return;
  }
  static auto type_name{type_name_from_pretty_function(__PRETTY_FUNCTION__)};
  synapse_logger::ostr_t out{synapse_logger::get_ostr()};
  out << R"("name":"object", "args":{"at":")" << (void*)obj << R"(", "type":")"
      << type_name << R"(", "fields":[)" << absl::MakeSpan(*obj) << "]}";
  synapse_logger::log(out.str());
}

void dump_node_create_params(
    const char* pGuid,
    const void* pUserParams,
    const unsigned paramsSize) {
  if (absl::string_view(pGuid) == "transpose_logic") {
    dump_object(reinterpret_cast<const std::vector<TransposePermutationDim>*>(
        pUserParams));
  } else {
    dump_object((uint8_t*)(pUserParams), paramsSize);
  }
}

} // namespace synapse_logger

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
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  synapse_logger::dump_node_create_params(pGuid, pUserParams, paramsSize);
  API_LOG_CALL(
      ARG(graphHandle),
      M_ARG(pInputsTensorList, numberInputs),
      M_ARG(pOutputsTensorList, numberOutputs),
      ARG(numberInputs),
      ARG(numberOutputs),
      ARG_Q(pUserParams),
      ARG_X(paramsSize),
      ARG_Q(pGuid),
      ARG_Q(pName),
      M_ARG(inputLayouts, numberInputs),
      M_ARG(outputLayouts, numberOutputs));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synNodeCreate,
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
      outputLayouts)
  API_LOG_RESULT();
  return status;
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
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  synapse_logger::dump_node_create_params(pGuid, pUserParams, paramsSize);
  API_LOG_CALL(
      ARG(graphHandle),
      M_ARG(pInputsTensorList, numberInputs),
      M_ARG(pOutputsTensorList, numberOutputs),
      ARG(numberInputs),
      ARG(numberOutputs),
      ARG_Q(pUserParams),
      ARG_X(paramsSize),
      ARG_Q(pGuid),
      ARG_Q(pName),
      ARG(nodeUniqueId),
      M_ARG_Q(inputLayouts, numberInputs),
      M_ARG_Q(outputLayouts, numberOutputs));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synNodeCreateWithId,
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
      outputLayouts)
  API_LOG_RESULT(S_ARG_Q(nodeUniqueId));
  return status;
}

synStatus synNodeDependencySet(
    const synGraphHandle graphHandle,
    const synNodeId* pBlockingNodesIdList,
    const synNodeId* pBlockedNodesIdList,
    const uint32_t numberblocking,
    const uint32_t numberblocked) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(
      ARG(graphHandle),
      M_ARG(pBlockingNodesIdList, numberblocking),
      M_ARG(pBlockedNodesIdList, numberblocked),
      ARG(numberblocking),
      ARG(numberblocked));
#if 1
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synNodeDependencySet,
      graphHandle,
      pBlockingNodesIdList,
      pBlockedNodesIdList,
      numberblocking,
      numberblocked)
#else
  synStatus status = synSuccess;
#endif
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synGraphCompile(
    synRecipeHandle* pRecipeHandle,
    const synGraphHandle graphHandle,
    const char* pRecipeName,
    const char* pBuildLog) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(
      ARG(pRecipeHandle),
      ARG(graphHandle),
      ARG_Q(pRecipeName),
      ARG_Q(pBuildLog));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synGraphCompile,
      pRecipeHandle,
      graphHandle,
      pRecipeName,
      pBuildLog)

  API_LOG_RESULT(S_ARG(pRecipeHandle));
  return status;
}

synStatus SYN_API_CALL
synGraphCreate(synGraphHandle* pGraphHandle, const synDeviceType deviceType) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pGraphHandle), ARG(deviceType));
  synStatus status = lib_synapse::synGraphCreate(pGraphHandle, deviceType);
  API_LOG_RESULT(S_ARG(pGraphHandle));
  return status;
}

synStatus SYN_API_CALL synGraphCreateEager(
    synGraphHandle* pGraphHandle,
    const synDeviceType deviceType) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pGraphHandle), ARG(deviceType));
  synStatus status = lib_synapse::synGraphCreateEager(pGraphHandle, deviceType);
  API_LOG_RESULT(S_ARG(pGraphHandle));
  return status;
}

synStatus SYN_API_CALL synGraphDestroy(const synGraphHandle graphHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(graphHandle));
  synStatus status = lib_synapse::synGraphDestroy(graphHandle);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synMemsetD32Async(
    uint64_t pDeviceMem,
    const uint32_t value,
    const size_t numOfElements,
    const synStreamHandle streamHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(
      ARG_X(pDeviceMem), ARG_X(value), ARG(numOfElements), ARG(streamHandle));
  synStatus status = lib_synapse::synMemsetD32Async(
      pDeviceMem, value, numOfElements, streamHandle);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synMemsetD8Async(
    uint64_t pDeviceMem,
    const unsigned char value,
    const size_t numOfElements,
    const synStreamHandle streamHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(
      ARG_X(pDeviceMem), ARG_X(value), ARG(numOfElements), ARG(streamHandle));
  synStatus status = lib_synapse::synMemsetD8Async(
      pDeviceMem, value, numOfElements, streamHandle);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synMemsetD16Async(
    uint64_t pDeviceMem,
    const uint16_t value,
    const size_t numOfElements,
    const synStreamHandle streamHandle) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(
      ARG_X(pDeviceMem), ARG_X(value), ARG(numOfElements), ARG(streamHandle));
  synStatus status = lib_synapse::synMemsetD16Async(
      pDeviceMem, value, numOfElements, streamHandle);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synHostMalloc(
    const synDeviceId deviceId,
    const uint64_t size,
    const uint32_t flags,
    void** buffer) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(deviceId), ARG_X(size), ARG_X(flags), ARG(buffer));
  synStatus status = lib_synapse::synHostMalloc(deviceId, size, flags, buffer);
  API_LOG_RESULT(S_ARG_Q(buffer));
  return status;
}

synStatus SYN_API_CALL synHostFree(
    const synDeviceId deviceId,
    const void* buffer,
    const uint32_t flags) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(deviceId), ARG_Q(buffer), ARG_X(flags));
  synStatus status = lib_synapse::synHostFree(deviceId, buffer, flags);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synHostMap(
    const synDeviceId deviceId,
    const uint64_t size,
    const void* buffer) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG(deviceId), ARG_X(size), ARG_Q(buffer));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synHostMap, deviceId, size, buffer)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synHostUnmap(const synDeviceId deviceId, const void* buffer) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(deviceId), ARG_Q(buffer));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synHostUnmap, deviceId, buffer)
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synDeviceMalloc(
    const synDeviceId deviceId,
    const uint64_t size,
    uint64_t reqAddr,
    const uint32_t flags,
    uint64_t* buffer) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(
      ARG(deviceId), ARG_X(size), ARG_X(reqAddr), ARG(flags), ARG(buffer));
  synStatus status =
      lib_synapse::synDeviceMalloc(deviceId, size, reqAddr, flags, buffer);
  API_LOG_RESULT(S_ARG_X(buffer));
  return status;
}

synStatus SYN_API_CALL synDeviceFree(
    const synDeviceId deviceId,
    const uint64_t buffer,
    const uint32_t flags) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(deviceId), ARG_X(buffer), ARG_X(flags));
  synStatus status = lib_synapse::synDeviceFree(deviceId, buffer, flags);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synDeviceGetAttribute(
    uint64_t* retVal,
    const synDeviceAttribute* deviceAttr,
    const unsigned querySize,
    const synDeviceId deviceId) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(retVal), ARG(deviceAttr), ARG_X(querySize), ARG(deviceId));
  synStatus status = lib_synapse::synDeviceGetAttribute(
      retVal, deviceAttr, querySize, deviceId);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synDeviceRelease(const synDeviceId deviceId) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(deviceId));
  synStatus status = lib_synapse::synDeviceRelease(deviceId);
  API_LOG_RESULT();
  synapse_logger::logger.dump_trace_info();
  synapse_logger::logger.last_acquired_id(
      synapse_logger::SynapseLogger::SYN_DEVICE_ID_UNASSIGNED);
  return status;
}

synStatus SYN_API_CALL synDeviceGetMemoryInfo(
    const synDeviceId deviceId,
    uint64_t* free,
    uint64_t* total) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(deviceId), ARG(free), ARG(total));
  synStatus status = lib_synapse::synDeviceGetMemoryInfo(deviceId, free, total);
  API_LOG_RESULT(S_ARG(free), S_ARG(total));
  return status;
}

synStatus SYN_API_CALL
synDeviceGetInfo(const synDeviceId deviceId, synDeviceInfo* pDeviceInfo) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG(deviceId), ARG(pDeviceInfo));
  synStatus status = lib_synapse::synDeviceGetInfo(deviceId, pDeviceInfo);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synProfilerStart(const synTraceType type, const synDeviceId deviceId) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG(type), ARG(deviceId));
  synStatus status = lib_synapse::synProfilerStart(type, deviceId);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synProfilerStop(const synTraceType type, const synDeviceId deviceId) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG(type), ARG(deviceId));
  synStatus status = lib_synapse::synProfilerStop(type, deviceId);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synProfilerGetTrace(
    const synTraceType type,
    const synDeviceId deviceId,
    const synTraceFormat format,
    void* buffer,
    size_t* size,
    size_t* numEntries) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(
      ARG(type),
      ARG(deviceId),
      ARG(format),
      ARG(buffer),
      ARG(size),
      ARG(numEntries));
  synStatus status = lib_synapse::synProfilerGetTrace(
      type, deviceId, format, buffer, size, numEntries);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synConfigurationSet(
    const char* configurationName,
    const char* configurationValue) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG_Q(configurationName), ARG_Q(configurationValue));
  synStatus status =
      lib_synapse::synConfigurationSet(configurationName, configurationValue);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synConfigurationGet(
    const char* configurationName,
    char* configurationValue,
    uint64_t size) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG_Q(configurationName), ARG_Q(configurationValue), ARG(size));
  synStatus status = lib_synapse::synConfigurationGet(
      configurationName, configurationValue, size);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorHandleCreate(
    synTensor* pTensor,
    synGraphHandle graph,
    synTensorType type,
    const char* tensorName) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG(pTensor), ARG(graph), ARG(type), ARG_Q(tensorName));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorHandleCreate, pTensor, graph, type, tensorName);
  API_LOG_RESULT(S_ARG(pTensor));
  return status;
}

synStatus SYN_API_CALL
synTensorGetName(const synTensor tensor, const uint64_t size, char* name) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(tensor), ARG(size), ARG(name));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synTensorGetName, tensor, size, name);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorSetExternal(synTensor tensor, bool isExternal) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(tensor), ARG(isExternal));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synTensorSetExternal, tensor, isExternal);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL
synTensorSetDeviceDataType(synTensor tensor, const synDataType dataType) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synTensorSetDeviceDataType, tensor, dataType);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorRetrieveLaunchAmount(
    const synRecipeHandle pRecipeHandle,
    uint32_t* numOfTensors) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pRecipeHandle), ARG(numOfTensors));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorRetrieveLaunchAmount, pRecipeHandle, numOfTensors);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorRetrieveLaunchIds(
    const synRecipeHandle pRecipeHandle,
    uint64_t* tensorsIds,
    const uint32_t numOfTensors) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pRecipeHandle), ARG(tensorsIds), ARG(numOfTensors));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorRetrieveLaunchIds,
      pRecipeHandle,
      tensorsIds,
      numOfTensors);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorRetrieveLaunchInfoById(
    const synRecipeHandle pRecipeHandle,
    const uint32_t numOfTensors,
    synRetrievedLaunchTensorInfo* tensorsLaunchInfo) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pRecipeHandle), ARG(numOfTensors), ARG(tensorsLaunchInfo));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorRetrieveLaunchInfoById,
      pRecipeHandle,
      numOfTensors,
      tensorsLaunchInfo);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorRetrieveLaunchInfoByIdExt(
    const synRecipeHandle pRecipeHandle,
    const uint32_t numOfTensors,
    synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);
  API_LOG_CALL(ARG(pRecipeHandle), ARG(numOfTensors), ARG(tensorsLaunchInfo));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorRetrieveLaunchInfoByIdExt,
      pRecipeHandle,
      numOfTensors,
      tensorsLaunchInfo);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorSetGeometryExt(
    synTensor tensor,
    const synTensorGeometryExt* geometry,
    synGeometryType geometryType) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  log_synTensorSetGeometry(geometry, tensor);
  API_LOG_CALL(ARG(tensor), ARG(geometry), ARG(geometryType));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorSetGeometryExt, tensor, geometry, geometryType);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorSetHostPtr(
    synTensor tensor,
    void* hostPtr,
    uint64_t size,
    synDataType dataType,
    bool copyBuffer) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(
      ARG(tensor), ARG(hostPtr), ARG(size), ARG(dataType), ARG(copyBuffer));
  synStatus status;
  CALL_SYN_FUNC(
      lib_synapse::synTensorSetHostPtr,
      tensor,
      hostPtr,
      size,
      dataType,
      copyBuffer);
  API_LOG_RESULT();
  return status;
}

synStatus SYN_API_CALL synTensorSetPermutation(
    synTensor tensor,
    const synTensorPermutation* permutation) {
  LOG_TRACE("SYN_API", "{}", __FUNCTION__);

  API_LOG_CALL(ARG(tensor), ARG(permutation));
  synStatus status;
  CALL_SYN_FUNC(lib_synapse::synTensorSetPermutation, tensor, permutation);
  API_LOG_RESULT();
  return status;
}
