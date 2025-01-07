#pragma once
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <queue>
#include "synapse_api.h"

#ifndef TEMPORARY_MEDIA_PYTORCH_API_FLAG
#define TEMPORARY_MEDIA_PYTORCH_API_FLAG
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef enum mediaPytFwProxyDtype {
  MEDIA_PYTFWPROXY_FLOAT32,
  MEDIA_PYTFWPROXY_BFLOAT16,
  MEDIA_PYTFWPROXY_UINT8,
  MEDIA_PYTFWPROXY_UINT32,
  MEDIA_PYTFWPROXY_UINT64,
  MEDIA_PYTFWPROXY_INT32,
  MEDIA_PYTFWPROXY_INT64,
} mediaPytFwProxyDtype;

typedef uint64_t mediaFwProxy_f_allocFwOutTensor(
    void* impl,
    const uint64_t* shape,
    size_t shape_size,
    int dtype);
typedef uint64_t mediaFwProxy_f_allocHostFwOutTensor(
    void* impl,
    const uint64_t* shape,
    size_t shape_size,
    int dtype);
typedef void mediaFwProxy_f_freeFwOutTensor(void* impl, uint64_t addr);
typedef uint64_t mediaFwProxy_f_allocBuffer(void* impl, size_t size);
typedef void mediaFwProxy_f_freeBuffer(void* impl, uint64_t addr);
typedef synDeviceId mediaFwProxy_f_getSynDeviceId(void* impl);
typedef synStreamHandle mediaFwProxy_f_getComputeStream(void* impl);

typedef struct mediaFwProxy {
  // Pointer to the framework implementation of the proxy
  void* impl;

  // Allocates framework tensor and returns its device address.
  // The address to the buffer will be final output of MediaAPI.
  // And will be exchanged by framework-bridge back to framework Tensor.
  // Ownership of the buffer/tensor will be transfered to framework.
  mediaFwProxy_f_allocFwOutTensor* allocDeviceFwOutTensor;

  // Allocates framework tensor and returns its host address.
  // The address to the buffer will be final output of MediaAPI.
  // And will be exchanged by framework-bridge back to framework host Tensor.
  // Ownership of the buffer/tensor will be transfered to framework.
  mediaFwProxy_f_allocHostFwOutTensor* allocHostFwOutTensor;

  // Frees output tensors not yet transfered to framework.
  mediaFwProxy_f_freeFwOutTensor* freeFwOutTensor;

  // Allocate persistent buffer and returns its device address.
  // MediaAPI/proxy is owner of this buffer and it must be explicitly
  // deallocated.
  mediaFwProxy_f_allocBuffer* allocDeviceBuffer;

  // Release persistent buffer.
  mediaFwProxy_f_freeBuffer* freeDeviceBuffer;

  // Get synapse device id.
  mediaFwProxy_f_getSynDeviceId* getSynDeviceId;

  // Get compute stream handle
  mediaFwProxy_f_getComputeStream* getComputeStream;

} mediaFwProxy;

void mediaPytFwProxy_init(
    mediaFwProxy* proxy,
    void* impl,
    mediaFwProxy_f_allocFwOutTensor* allocDeviceFwOutTensor,
    mediaFwProxy_f_allocHostFwOutTensor* allocHostFwOutTensor,
    mediaFwProxy_f_freeFwOutTensor* freeFwOutTensor,
    mediaFwProxy_f_allocBuffer* allocDeviceBuffer,
    mediaFwProxy_f_freeBuffer* freeDeviceBuffer,
    mediaFwProxy_f_getSynDeviceId* getSynDeviceId,
    mediaFwProxy_f_getComputeStream* getComputeStream);

uint64_t mediaPytFwProxy_allocDeviceFwOutTensor(
    mediaFwProxy* proxy,
    const uint64_t* shape,
    size_t shape_size,
    int dtype);

uint64_t mediaPytFwProxy_allocHostFwOutTensor(
    mediaFwProxy* proxy,
    const uint64_t* shape,
    size_t shape_size,
    int dtype);

void mediaPytFwProxy_freeFwOutTensor(mediaFwProxy* proxy, uint64_t addr);

uint64_t mediaPytFwProxy_allocDeviceBuffer(mediaFwProxy* proxy, size_t size);

void mediaPytFwProxy_freeDeviceBuffer(mediaFwProxy* proxy, uint64_t addr);

uint64_t mediaPytFwProxy_getSynDeviceId(mediaFwProxy* proxy);

synStreamHandle mediaPytFwProxy_getComputeStream(mediaFwProxy* proxy);

#ifdef __cplusplus
} /* extern "C" */
#endif
