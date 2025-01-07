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