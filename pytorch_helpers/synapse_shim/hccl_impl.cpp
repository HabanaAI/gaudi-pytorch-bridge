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

#include <hccl.h>
#include <hccl_types.h>
#include "synapse_shim/hccl_api_shim.h"

namespace shim_hccl {

extern "C" {

hcclResult_t hcclGetVersion(int* version) {
  return hccl_api->hcclGetVersion(version);
}

hcclResult_t hcclGetUniqueId(hcclUniqueId* uniqueId) {
  return hccl_api->hcclGetUniqueId(uniqueId);
}

hcclResult_t hcclCommInitRank(
    hcclComm_t* comm,
    int nranks,
    hcclUniqueId commId,
    int rank) {
  return hccl_api->hcclCommInitRank(comm, nranks, commId, rank);
}

hcclResult_t hcclCommInitAll(hcclComm_t* comm, int ndev, const int* devlist) {
  return hccl_api->hcclCommInitAll(comm, ndev, devlist);
}

hcclResult_t hcclCommDestroy(hcclComm_t comm) {
  return hccl_api->hcclCommDestroy(comm);
}

hcclResult_t hcclCommAbort(hcclComm_t comm) {
  return hccl_api->hcclCommAbort(comm);
}

const char* hcclGetErrorString(hcclResult_t result) {
  return hccl_api->hcclGetErrorString(result);
}

hcclResult_t hcclCommGetAsyncError(hcclComm_t comm, hcclResult_t* asyncError) {
  return hccl_api->hcclCommGetAsyncError(comm, asyncError);
}

hcclResult_t hcclCommCount(hcclComm_t comm, int* count) {
  return hccl_api->hcclCommCount(comm, count);
}

hcclResult_t hcclCommSynDevice(hcclComm_t, int*) {
  return hcclUnsupported;
}

hcclResult_t hcclCommUserRank(hcclComm_t comm, int* rank) {
  return hccl_api->hcclCommCount(comm, rank);
}

int hcclLookupDMABuff(uint64_t addr, uint64_t size, int* fd) {
  return hccl_api->hcclLookupDMABuff(addr, size, fd);
}

hcclResult_t hcclReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    hcclDataType_t datatype,
    hcclRedOp_t op,
    int root,
    hcclComm_t comm,
    synStreamHandle stream_handle) {
  return hccl_api->hcclReduce(
      sendbuff, recvbuff, count, datatype, op, root, comm, stream_handle);
}

hcclResult_t hcclBroadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    hcclDataType_t datatype,
    int root,
    hcclComm_t comm,
    synStreamHandle stream_handle) {
  return hccl_api->hcclBroadcast(
      sendbuff, recvbuff, count, datatype, root, comm, stream_handle);
}

hcclResult_t hcclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    hcclDataType_t datatype,
    hcclRedOp_t op,
    hcclComm_t comm,
    synStreamHandle stream_handle) {
  return hccl_api->hcclAllReduce(
      sendbuff, recvbuff, count, datatype, op, comm, stream_handle);
}

hcclResult_t hcclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    hcclDataType_t datatype,
    hcclRedOp_t op,
    hcclComm_t comm,
    synStreamHandle stream_handle) {
  return hccl_api->hcclReduceScatter(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream_handle);
}

hcclResult_t hcclAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    hcclDataType_t datatype,
    hcclComm_t comm,
    synStreamHandle stream_handle) {
  return hccl_api->hcclAllGather(
      sendbuff, recvbuff, sendcount, datatype, comm, stream_handle);
}

hcclResult_t hcclAlltoAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    hcclDataType_t datatype,
    hcclComm_t comm,
    synStreamHandle stream_handle) {
  return hccl_api->hcclAlltoAll(
      sendbuff, recvbuff, count, datatype, comm, stream_handle);
}

hcclResult_t hcclSend(
    const void* sendbuff,
    size_t count,
    hcclDataType_t datatype,
    int peer,
    hcclComm_t comm,
    synStreamHandle stream) {
  return hccl_api->hcclSend(sendbuff, count, datatype, peer, comm, stream);
}

hcclResult_t hcclRecv(
    void* recvbuff,
    size_t count,
    hcclDataType_t datatype,
    int peer,
    hcclComm_t comm,
    synStreamHandle stream) {
  return hccl_api->hcclRecv(recvbuff, count, datatype, peer, comm, stream);
}

hcclResult_t hcclGroupStart() {
  return hccl_api->hcclGroupStart();
}

hcclResult_t hcclGroupEnd() {
  return hccl_api->hcclGroupEnd();
}

} // extern "C"
} // namespace shim_hccl
