/*******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/

#include <dlfcn.h>
#include <link.h>
#include <media_pytorch_proxy.h>
#include "synapse_shim/logging.h"

#define MEDIA_API_PTR(func) decltype(::func)* func
#define MEDIA_API_INIT_PTR(func) \
  CHECK_NULL(func = (decltype(func))dlsym(lib_handle, #func))

namespace shim_media {
MEDIA_API_PTR(mediaPytFwProxy_init);

void LoadSymbols(void* lib_handle) {
  MEDIA_API_INIT_PTR(mediaPytFwProxy_init);
}

} // namespace shim_media

class LibMediaLoader {
 public:
  static void EnsureLoaded();
  static LibMediaLoader& GetInstance();
  LibMediaLoader(LibMediaLoader const&) = delete;
  void operator=(LibMediaLoader const&) = delete;
  std::string lib_path() {
    return media_lib_path_;
  }

 private:
  LibMediaLoader();
  ~LibMediaLoader();
  void* media_lib_handle_;
  std::string media_lib_path_;
};

void LibMediaLoader::EnsureLoaded() {
  LibMediaLoader::GetInstance();
}

LibMediaLoader& LibMediaLoader::GetInstance() {
  static LibMediaLoader instance;
  return instance;
}

LibMediaLoader::LibMediaLoader() {
  media_lib_handle_ = dlopen("libmedia.so", RTLD_LOCAL | RTLD_NOW);
  CHECK_NULL(media_lib_handle_);
  shim_media::LoadSymbols(media_lib_handle_);
  link_map* l_map = nullptr;
  CHECK_TRUE_DL(!dlinfo(media_lib_handle_, RTLD_DI_LINKMAP, &l_map));
  media_lib_path_ = l_map->l_name;
}

LibMediaLoader::~LibMediaLoader() {
  dlclose(media_lib_handle_);
}

void mediaPytFwProxy_init(
    mediaFwProxy* proxy,
    void* impl,
    mediaFwProxy_f_allocFwOutTensor* allocDeviceFwOutTensor,
    mediaFwProxy_f_allocHostFwOutTensor* allocHostFwOutTensor,
    mediaFwProxy_f_freeFwOutTensor* freeFwOutTensor,
    mediaFwProxy_f_allocBuffer* allocDeviceBuffer,
    mediaFwProxy_f_freeBuffer* freeDeviceBuffer,
    mediaFwProxy_f_getSynDeviceId* getSynDeviceId,
    mediaFwProxy_f_getComputeStream* getComputeStream) {
  LibMediaLoader::EnsureLoaded();
  shim_media::mediaPytFwProxy_init(
      proxy,
      impl,
      allocDeviceFwOutTensor,
      allocHostFwOutTensor,
      freeFwOutTensor,
      allocDeviceBuffer,
      freeDeviceBuffer,
      getSynDeviceId,
      getComputeStream);
}