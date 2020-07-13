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
#include <cstdint>
#include <ostream>

#include "arg_utils.h"
#include "hcl_api_types.h"
#include "object_dump.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "synapse_logger.h"

namespace lib_hcl {

HCLStatus (*HCL_Init)(const synDeviceId deviceId, const char* configFileName);
HCLStatus (*HCL_Destroy)();
HCLStatus (*HCL_Comm_Size)(HCL_Comm comm, int* size);
HCLStatus (*HCL_Comm_Rank)(HCL_Comm comm, HCL_Rank* rank);
HCLStatus (*HCL_Comm_Ranks)(HCL_Comm comm, HCL_Rank* rankList, int count);
HCLStatus (*HCL_Wait)(HCL_Request phRequest, uint64_t microSeconds);
HCLStatus (*HCL_Allreduce)(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr,
                           uint64_t count, synDataType dataType, uint64_t intermediateBufferAddr,
                           uint64_t intermediateSize, HCL_Op op, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_IAllreduce)(HCL_Request* phRequest, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                            synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize,
                            HCL_Op op, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_Get_Intermediate_Buffer_size)(uint64_t* intermediateSize, const HCL_CollectiveOp collectiveOp,
                                              const uint64_t count, synDataType dataType, const HCL_Comm communicator);
HCLStatus (*HCL_Bcast)(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                       synDataType dataType, HCL_Rank root, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_IBcast)(HCL_Request* phRequest, uint64_t Address, uint64_t count, synDataType dataType, HCL_Rank root,
                        HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_Reduce)(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                        synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize,
                        uint16_t destRank, HCL_Op op, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_IReduce)(HCL_Request* phRequest, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                         synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize,
                         uint16_t destRank, HCL_Op op, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_Reduce_Scatter)(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr,
                                uint64_t count, synDataType dataType, uint64_t intermediateBufferAddr,
                                uint64_t intermediateSize, HCL_Op op, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_IReduce_Scatter)(HCL_Request* phRequest, uint64_t sendBufAddr, uint64_t receiveBuffAddr,
                                 uint64_t count, synDataType dataType, uint64_t intermediateBufferAddr,
                                 uint64_t intermediateSize, HCL_Op op, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_AllGather)(synStreamHandle streamHandle, uint64_t sendBufAddr, uint64_t receiveBuffAddr,
                           uint64_t count, synDataType dataType, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_IAllGather)(HCL_Request* phRequest, uint64_t sendBufAddr, uint64_t receiveBuffAddr, uint64_t count,
                            synDataType dataType, HCL_Comm communicator, bool sameAddress);
HCLStatus (*HCL_NetworkFlush)(HCL_Request* phRequest, synStreamHandle streamHandle);

void LoadSymbols(void* lib_handle) {
  CHECK_NULL(HCL_Init = (decltype(HCL_Init))dlsym(lib_handle, "HCL_Init"));
  CHECK_NULL(HCL_Destroy = (decltype(HCL_Destroy))dlsym(lib_handle, "HCL_Destroy"));
  CHECK_NULL(HCL_Comm_Size = (decltype(HCL_Comm_Size))dlsym(lib_handle, "HCL_Comm_Size"));
  CHECK_NULL(HCL_Comm_Rank = (decltype(HCL_Comm_Rank))dlsym(lib_handle, "HCL_Comm_Rank"));
  CHECK_NULL(HCL_Comm_Ranks = (decltype(HCL_Comm_Ranks))dlsym(lib_handle, "HCL_Comm_Ranks"));
  CHECK_NULL(HCL_Wait = (decltype(HCL_Wait))dlsym(lib_handle, "HCL_Wait"));
  CHECK_NULL(HCL_Allreduce = (decltype(HCL_Allreduce))dlsym(lib_handle, "HCL_Allreduce"));
  CHECK_NULL(HCL_IAllreduce = (decltype(HCL_IAllreduce))dlsym(lib_handle, "HCL_IAllreduce"));
  CHECK_NULL(HCL_Get_Intermediate_Buffer_size =
                 (decltype(HCL_Get_Intermediate_Buffer_size))dlsym(lib_handle, "HCL_Get_Intermediate_Buffer_size"));
  CHECK_NULL(HCL_Bcast = (decltype(HCL_Bcast))dlsym(lib_handle, "HCL_Bcast"));
  CHECK_NULL(HCL_IBcast = (decltype(HCL_IBcast))dlsym(lib_handle, "HCL_Bcast"));
  CHECK_NULL(HCL_Reduce = (decltype(HCL_Reduce))dlsym(lib_handle, "HCL_Reduce"));
  CHECK_NULL(HCL_IReduce = (decltype(HCL_IReduce))dlsym(lib_handle, "HCL_IReduce"));
  CHECK_NULL(HCL_Reduce_Scatter = (decltype(HCL_Reduce_Scatter))dlsym(lib_handle, "HCL_Reduce_Scatter"));
  CHECK_NULL(HCL_IReduce_Scatter = (decltype(HCL_IReduce_Scatter))dlsym(lib_handle, "HCL_IReduce_Scatter"));
  CHECK_NULL(HCL_AllGather = (decltype(HCL_AllGather))dlsym(lib_handle, "HCL_AllGather"));
  CHECK_NULL(HCL_IAllGather = (decltype(HCL_IAllGather))dlsym(lib_handle, "HCL_IAllGather"));
  CHECK_NULL(HCL_NetworkFlush = (decltype(HCL_NetworkFlush))dlsym(lib_handle, "HCL_NetworkFlush"));
}

}  // namespace lib_hcl

extern "C" {
HCLStatus HCL_Init(const synDeviceId deviceId, const char* configFileName) {
  API_LOG_CALL(ARG(deviceId), ARG_Q(configFileName));
  HCLStatus status = lib_hcl::HCL_Init(deviceId, configFileName);
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_Destroy() {
  API_LOG_CALL();
  HCLStatus status = lib_hcl::HCL_Destroy();
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_Comm_Size(HCL_Comm comm, int* size) {
  API_LOG_CALL(ARG_Q(comm), ARG(size));
  HCLStatus status = lib_hcl::HCL_Comm_Size(comm, size);
  API_LOG_RESULT(S_ARG_X(size));
  return status;
}

HCLStatus HCL_Comm_Rank(HCL_Comm comm, HCL_Rank* rank) {
  API_LOG_CALL(ARG_Q(comm), ARG(rank));
  HCLStatus status = lib_hcl::HCL_Comm_Rank(comm, rank);
  API_LOG_RESULT(S_ARG_X(rank));
  return status;
}

HCLStatus HCL_Comm_Ranks(HCL_Comm comm, HCL_Rank* rankList, int count) {
  API_LOG_CALL(ARG_Q(comm), ARG(rankList), ARG_X(count));
  HCLStatus status = lib_hcl::HCL_Comm_Ranks(comm, rankList, count);
  synapse_logger::dump_object(rankList, count);
  API_LOG_RESULT(M_ARG(rankList, count));
  return status;
}

std::ostream& operator<<(std::ostream& os, const HCL_Request request) {
  os << "\"{" << request.event << ", " << request.index << ", " << request.pIndex << "}\"";
  return os;
}

HCLStatus HCL_Wait(HCL_Request phRequest, uint64_t microSeconds) {
  API_LOG_CALL(ARG(phRequest), ARG_X(microSeconds));
  HCLStatus status = lib_hcl::HCL_Wait(phRequest, microSeconds);
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_Allreduce(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                        synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize, HCL_Op op,
                        HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(streamHandle), ARG_X(sendBuffAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_X(intermediateBufferAddr), ARG_X(intermediateSize), ARG_X(op), ARG_Q(communicator),
               ARG_X(sameAddress));
  HCLStatus status = lib_hcl::HCL_Allreduce(streamHandle, sendBuffAddr, receiveBuffAddr, count, dataType,
                                            intermediateBufferAddr, intermediateSize, op, communicator, sameAddress);
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_IAllreduce(HCL_Request* phRequest, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                         synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize, HCL_Op op,
                         HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(phRequest), ARG_X(sendBuffAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_X(intermediateBufferAddr), ARG_X(intermediateSize), ARG_X(op), ARG_Q(communicator),
               ARG_X(sameAddress));
  HCLStatus status = lib_hcl::HCL_IAllreduce(phRequest, sendBuffAddr, receiveBuffAddr, count, dataType,
                                             intermediateBufferAddr, intermediateSize, op, communicator, sameAddress);
  API_LOG_RESULT(S_ARG_X(phRequest));
  return status;
}

HCLStatus HCL_Get_Intermediate_Buffer_size(uint64_t* intermediateSize, const HCL_CollectiveOp collectiveOp,
                                           const uint64_t count, synDataType dataType, const HCL_Comm communicator) {
  API_LOG_CALL(ARG(intermediateSize), ARG_X(collectiveOp), ARG_X(count), ARG_X(dataType), ARG_Q(communicator));
  HCLStatus status =
      lib_hcl::HCL_Get_Intermediate_Buffer_size(intermediateSize, collectiveOp, count, dataType, communicator);
  API_LOG_RESULT(S_ARG_X(intermediateSize));
  return status;
}

HCLStatus HCL_Bcast(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                    synDataType dataType, HCL_Rank root, HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(streamHandle), ARG_X(sendBuffAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_X(root), ARG_Q(communicator), ARG_X(sameAddress));
  HCLStatus status = lib_hcl::HCL_Bcast(streamHandle, sendBuffAddr, receiveBuffAddr, count, dataType, root,
                                        communicator, sameAddress);
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_IBcast(HCL_Request* phRequest, uint64_t Address, uint64_t count, synDataType dataType, HCL_Rank root,
                     HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(phRequest), ARG_X(Address), ARG_X(count), ARG_X(dataType), ARG_X(root), ARG_Q(communicator),
               ARG_X(sameAddress));
  HCLStatus status = lib_hcl::HCL_IBcast(phRequest, Address, count, dataType, root, communicator, sameAddress);
  API_LOG_RESULT(S_ARG_X(phRequest));
  return status;
}

HCLStatus HCL_Reduce(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                     synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize,
                     uint16_t destRank, HCL_Op op, HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(streamHandle), ARG_X(sendBuffAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_X(intermediateBufferAddr), ARG_X(intermediateSize), ARG_X(destRank), ARG_X(op), ARG_Q(communicator),
               ARG_X(sameAddress));
  HCLStatus status =
      lib_hcl::HCL_Reduce(streamHandle, sendBuffAddr, receiveBuffAddr, count, dataType, intermediateBufferAddr,
                          intermediateSize, destRank, op, communicator, sameAddress);
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_IReduce(HCL_Request* phRequest, uint64_t sendBuffAddr, uint64_t receiveBuffAddr, uint64_t count,
                      synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize,
                      uint16_t destRank, HCL_Op op, HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(phRequest), ARG_X(sendBuffAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_X(intermediateBufferAddr), ARG_X(intermediateSize), ARG_X(destRank), ARG_X(op), ARG_Q(communicator),
               ARG_X(sameAddress));
  HCLStatus status =
      lib_hcl::HCL_IReduce(phRequest, sendBuffAddr, receiveBuffAddr, count, dataType, intermediateBufferAddr,
                           intermediateSize, destRank, op, communicator, sameAddress);
  API_LOG_RESULT(S_ARG_X(phRequest));
  return status;
}

HCLStatus HCL_Reduce_Scatter(synStreamHandle streamHandle, uint64_t sendBuffAddr, uint64_t receiveBuffAddr,
                             uint64_t count, synDataType dataType, uint64_t intermediateBufferAddr,
                             uint64_t intermediateSize, HCL_Op op, HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(streamHandle), ARG_X(sendBuffAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_X(intermediateBufferAddr), ARG_X(intermediateSize), ARG_X(op), ARG_Q(communicator),
               ARG_X(sameAddress));
  HCLStatus status =
      lib_hcl::HCL_Reduce_Scatter(streamHandle, sendBuffAddr, receiveBuffAddr, count, dataType, intermediateBufferAddr,
                                  intermediateSize, op, communicator, sameAddress);
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_IReduce_Scatter(HCL_Request* phRequest, uint64_t sendBufAddr, uint64_t receiveBuffAddr, uint64_t count,
                              synDataType dataType, uint64_t intermediateBufferAddr, uint64_t intermediateSize,
                              HCL_Op op, HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(phRequest), ARG_X(sendBufAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_X(intermediateBufferAddr), ARG_X(intermediateSize), ARG_X(op), ARG_Q(communicator),
               ARG_X(sameAddress));
  HCLStatus status =
      lib_hcl::HCL_IReduce_Scatter(phRequest, sendBufAddr, receiveBuffAddr, count, dataType, intermediateBufferAddr,
                                   intermediateSize, op, communicator, sameAddress);
  API_LOG_RESULT(S_ARG_X(phRequest));
  return status;
}

HCLStatus HCL_AllGather(synStreamHandle streamHandle, uint64_t sendBufAddr, uint64_t receiveBuffAddr, uint64_t count,
                        synDataType dataType, HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(streamHandle), ARG_X(sendBufAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_Q(communicator), ARG_X(sameAddress));
  HCLStatus status =
      lib_hcl::HCL_AllGather(streamHandle, sendBufAddr, receiveBuffAddr, count, dataType, communicator, sameAddress);
  API_LOG_RESULT();
  return status;
}

HCLStatus HCL_IAllGather(HCL_Request* phRequest, uint64_t sendBufAddr, uint64_t receiveBuffAddr, uint64_t count,
                         synDataType dataType, HCL_Comm communicator, bool sameAddress) {
  API_LOG_CALL(ARG(phRequest), ARG_X(sendBufAddr), ARG_X(receiveBuffAddr), ARG_X(count), ARG_X(dataType),
               ARG_Q(communicator), ARG_X(sameAddress));
  HCLStatus status =
      lib_hcl::HCL_IAllGather(phRequest, sendBufAddr, receiveBuffAddr, count, dataType, communicator, sameAddress);
  API_LOG_RESULT(S_ARG_X(phRequest));
  return status;
}

HCLStatus HCL_NetworkFlush(HCL_Request* phRequest, synStreamHandle streamHandle) {
  API_LOG_CALL(ARG(phRequest), ARG(streamHandle));
  HCLStatus status = lib_hcl::HCL_NetworkFlush(phRequest, streamHandle);
  API_LOG_RESULT(S_ARG_X(phRequest));
  return status;
}

}  // extern "C"
