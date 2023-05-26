#include <torch/extension.h>
#include <iostream>

#include <media_pytorch_proxy.h>
#include "habana_helpers/logging.h"
#include "pyt_media_proxy.h"

namespace torch_hpu {
namespace {
torch::ScalarType toTorchDType(mediaPytFwProxyDtype media_dtype) {
  switch (media_dtype) {
    case MEDIA_PYTFWPROXY_BFLOAT16:
      return torch::kFloat16;
    case MEDIA_PYTFWPROXY_FLOAT32:
      return torch::kFloat32;
    case MEDIA_PYTFWPROXY_UINT8:
      return torch::kUInt8;
    case MEDIA_PYTFWPROXY_UINT32:
      return torch::kInt32;
    case MEDIA_PYTFWPROXY_UINT64:
      return torch::kInt64;
    default:
      PT_BRIDGE_FATAL("Unsupported mediaPytFwProxyDtype dtype = ", media_dtype);
  }
}

habana_helpers::TensorShape toTorchShape(
    const uint64_t* shape,
    size_t shape_size) {
  habana_helpers::TensorShape tensor_shape;
  for (size_t i = 0; i < shape_size; i++) {
    tensor_shape.add_dim(static_cast<int64_t>(shape[i]));
  }
  return tensor_shape;
}

uint64_t allocDeviceFwOutTensor(
    void* impl,
    const uint64_t* shape,
    size_t shape_size,
    int dtype) {
  return reinterpret_cast<IMediaProxy*>(impl)
      ->allocateFrameworkDeviceOutputTensor(
          toTorchShape(shape, shape_size),
          toTorchDType(static_cast<mediaPytFwProxyDtype>(dtype)));
}

uint64_t allocHostFwOutTensor(
    void* impl,
    const uint64_t* shape,
    size_t shape_size,
    int dtype) {
  return reinterpret_cast<IMediaProxy*>(impl)
      ->allocateFrameworkHostOutputTensor(
          toTorchShape(shape, shape_size),
          toTorchDType(static_cast<mediaPytFwProxyDtype>(dtype)));
}

void freeFwOutTensor(void* impl, uint64_t addr) {
  reinterpret_cast<IMediaProxy*>(impl)->freeFrameworkOutputTensor(addr);
}

uint64_t allocDeviceBuffer(void* impl, size_t size) {
  return reinterpret_cast<IMediaProxy*>(impl)->allocatePersistentBuffer(size);
}

void freeDeviceBuffer(void* impl, uint64_t addr) {
  reinterpret_cast<IMediaProxy*>(impl)->freePersistentBuffer(addr);
}

synDeviceId getSynDeviceId(void* impl) {
  return reinterpret_cast<IMediaProxy*>(impl)->getSynDeviceId();
}

synStreamHandle getComputeStream(void* impl) {
  return reinterpret_cast<IMediaProxy*>(impl)->getComputeStream();
}

torch::Tensor getFwOutputTensor(void* impl, uintptr_t addr) {
  return reinterpret_cast<IMediaProxy*>(impl)->getFrameworkOutputTensor(addr);
}

struct MediaProxyHolder {
  MediaProxyHolder(int device_id) : media_proxy_impl_(device_id) {
    mediaPytFwProxy_init(
        &media_fw_proxy_,
        &media_proxy_impl_,
        allocDeviceFwOutTensor,
        allocHostFwOutTensor,
        freeFwOutTensor,
        allocDeviceBuffer,
        freeDeviceBuffer,
        getSynDeviceId,
        getComputeStream);
  }

  PytMediaProxy media_proxy_impl_;
  mediaFwProxy media_fw_proxy_{};
};
std::unique_ptr<MediaProxyHolder> media_pyt_proxy;
} // namespace

uintptr_t CreatePytMediaProxy(int device_id) {
  if (!media_pyt_proxy)
    media_pyt_proxy = std::make_unique<MediaProxyHolder>(device_id);
  return (uintptr_t)(&media_pyt_proxy->media_fw_proxy_);
}

torch::Tensor GetOutputTensor(uintptr_t addr) {
  return getFwOutputTensor(&media_pyt_proxy->media_proxy_impl_, addr);
}
} // namespace torch_hpu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "media_pyt_bridge python binding";
  m.def(
      "CreatePytMediaProxy",
      &torch_hpu::CreatePytMediaProxy,
      "Create media pytorch bridge proxy");
  m.def(
      "get_out_tensor",
      &torch_hpu::GetOutputTensor,
      "Return pytorch tensor from proxy object");
}
