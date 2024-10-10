/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include <absl/container/flat_hash_set.h>
#include <absl/types/variant.h>
#include <deque>
#include <memory>
#include <mutex>
#include "synapse_shim/hccl_api_shim.h"
#include "synapse_shim/logging.h"
#include "synapse_shim/synapse_api_shim.h"

/*
 * Source of "default" result for API stub parametrized with API call result
 * type.
 */
template <typename Result, typename Enable = void>
struct StubResult;

template <>
struct StubResult<synStatus> {
  synStatus operator()() {
    return synSuccess;
  }
};

template <>
struct StubResult<const char*> {
  const char* operator()() {
    return "<stub>";
  }
};

template <>
struct StubResult<hcclResult_t> {
  hcclResult_t operator()() {
    return hcclResult_t::hcclSuccess;
  }
};

template <typename IntT>
struct StubResult<
    IntT,
    typename std::enable_if<std::is_integral<IntT>::value>::type> {
  int operator()() {
    return {};
  }
};

/*
 * Factory of API call replacements that ignore arguments and return default
 * result as dictated by proper StubResult.
 */
template <typename T>
struct ApiStub;

template <typename Result, typename... Args>
struct ApiStub<std::function<Result(Args...)>> {
  using FunctionPtr = Result(Args...);
  using Function = std::function<FunctionPtr>;
  template <FunctionPtr ApiFunction>
  static Function Stub() {
    return [](Args&&...) {
      StubResult<Result> result{};
      return result();
    };
  }
};

#define INIT_STUB_LOADER_API(api, func) \
  api##_api_.func = ApiStub<api##_api_t::func##_t>::Stub<func>();
#define INIT_STUB_LOADER_SYN_API(func) INIT_STUB_LOADER_API(synapse, func)
#define INIT_STUB_LOADER_HCCL_API(func) INIT_STUB_LOADER_API(hccl, func)
/*
 * Provider of stubbed API implementation.
 */
class StubSynapseApi {
 public:
  StubSynapseApi() {
    SYN_API_SYMBOL_VISIT(INIT_STUB_LOADER_SYN_API);
    HCCL_API_SYMBOL_VISIT(INIT_STUB_LOADER_HCCL_API);
  }

  void Install() {
    syn_api = &synapse_api_;
    hccl_api = &hccl_api_;
  }

 protected:
  synapse_api_t synapse_api_;
  hccl_api_t hccl_api_;
};

void EnableSynapseApiStub() {
  static StubSynapseApi instance;
  instance.Install();
}

namespace null_hw {
template <typename SynHandle>
class Resource {};

template <>
class Resource<synTensor> {
 public:
  using synHandle = synTensor;
};
using Tensor = Resource<synTensor>;

template <>
class Resource<synSectionHandle> {
 public:
  using synHandle = synSectionHandle;
};
using Section = Resource<synSectionHandle>;

template <>
class Resource<synGraphHandle> {
 public:
  using SynHandle = synGraphHandle;
};
using Graph = Resource<synGraphHandle>;

template <>
class Resource<synRecipeHandle> {
 public:
  using SynHandle = synRecipeHandle;
  Resource<synRecipeHandle>(
      Graph& graph,
      const char* pRecipeName,
      const char* pBuildLog)
      : graph_{graph},
        recipe_name_{pRecipeName},
        build_log_{pBuildLog == nullptr ? "" : pBuildLog} {}
  Graph& graph_;
  std::string recipe_name_{};
  std::string build_log_{};
};
using Recipe = Resource<synRecipeHandle>;

template <>
class Resource<synEventHandle> {
 public:
  using SynHandle = synEventHandle;
};
using Event = Resource<synEventHandle>;

template <>
class Resource<synStreamHandle> {
 public:
  using SynHandle = synStreamHandle;
};
using Stream = Resource<synStreamHandle>;

using AnyResource =
    absl::variant<Tensor, Section, Recipe, Graph, Event, Stream>;

class SynapseApi : public StubSynapseApi {
 public:
  struct Consts {
    static constexpr std::uint64_t WORKSPACE_SIZE = 4096;
    static constexpr std::uint32_t DEVICE_ID = 0x00110000;
    static constexpr std::uint64_t NODE_ID = 0x1;
    static constexpr std::uint64_t RECIPE_SIZE = 1;
    static constexpr std::uint64_t STREAMS_TOTAL_MEM_SIZE = 1;
    static constexpr std::uint64_t TOTAL_MEMORY = 0x1000000000;
    static constexpr std::uint64_t FREE_MEMORY = 0x1000000000;
    static constexpr std::uint64_t ALLOCATION_START = 0x111D000000000ull;
    static constexpr std::uint64_t DEVICE_MALLOC_ALIGNMENT = 0x1000;
  };
  SynapseApi() : StubSynapseApi(), allocation_back_{Consts::ALLOCATION_START} {
    synapse_api_.synDeviceAcquireByDeviceType =
        [this](synDeviceId* id, const synDeviceType) {
          *id = this->AllocateDevice();
          return synSuccess;
        };

    synapse_api_.synDeviceAcquireByModuleId =
        [this](synDeviceId* id, const synModuleId) {
          *id = this->AllocateDevice();
          return synSuccess;
        };

    synapse_api_.synDeviceAcquire = [this](synDeviceId* id, const char*) {
      *id = this->AllocateDevice();
      return synSuccess;
    };

    synapse_api_.synGraphCompile = [this](
                                       synRecipeHandle* pRecipeHandle,
                                       const synGraphHandle graphHandle,
                                       const char* pRecipeName,
                                       const char* pBuildLog) {
      this->AddHandle(
          pRecipeHandle,
          this->HandleToResource(graphHandle),
          pRecipeName,
          pBuildLog);
      return synSuccess;
    };

    synapse_api_.synGraphCreate =
        [this](synGraphHandle* pGraphHandle, const synDeviceType) {
          this->AddHandle(pGraphHandle);
          return synSuccess;
        };

    synapse_api_.synTensorHandleCreate =
        [this](synTensor* tensor, synGraphHandle, synTensorType, const char*) {
          this->AddHandle(tensor);
          return synSuccess;
        };

    synapse_api_.synSectionCreate =
        [this](
            synSectionHandle* sectionHandle, uint64_t, const synGraphHandle) {
          this->AddHandle(sectionHandle);
          return synSuccess;
        };

    synapse_api_.synNodeCreateWithId = [this](
                                           const synGraphHandle,
                                           const synTensor*,
                                           const synTensor*,
                                           const uint32_t,
                                           const uint32_t,
                                           const void*,
                                           const unsigned,
                                           const char*,
                                           const char*,
                                           synNodeId* nodeUniqueId,
                                           const char**,
                                           const char**) {
      *nodeUniqueId = this->AllocateNodeId();
      return synSuccess;
    };

    synapse_api_.synStreamCreateGeneric =
        [this](
            synStreamHandle* pStreamHandle, const synDeviceId, const uint32_t) {
          this->AddHandle(pStreamHandle);
          return synSuccess;
        };

    synapse_api_.synDeviceGetNextStreamAffinity = [](const synDeviceId,
                                                     uint64_t* availAffinity) {
      *availAffinity = 0;
      return synSuccess;
    };

    synapse_api_.synEventCreate =
        [this](
            synEventHandle* pEventHandle, const synDeviceId, const uint32_t) {
          this->AddHandle(pEventHandle);
          return synSuccess;
        };

    synapse_api_.synDeviceGetCount = [](uint32_t* pCount) {
      *pCount = 1;
      return synSuccess;
    };

    synapse_api_.synDeviceGetCountByDeviceType = [](uint32_t* pCount,
                                                    const synDeviceType) {
      *pCount = 1;
      return synSuccess;
    };

    synapse_api_.synDeviceGetMemoryInfo =
        [](const synDeviceId, uint64_t* free, uint64_t* total) {
          *free = Consts::FREE_MEMORY;
          *total = Consts::TOTAL_MEMORY;
          return synSuccess;
        };

    synapse_api_.synWorkspaceGetSize = [](uint64_t* pWorkspaceSize,
                                          const synRecipeHandle) {
      *pWorkspaceSize = Consts::WORKSPACE_SIZE;
      return synSuccess;
    };

    synapse_api_.synDeviceMalloc = [this](
                                       const synDeviceId,
                                       const uint64_t size,
                                       uint64_t,
                                       const uint32_t,
                                       uint64_t* buffer) {
      *buffer = this->Malloc(size);
      return synSuccess;
    };

    synapse_api_.synDeviceGetInfo = [](const synDeviceId,
                                       synDeviceInfo* dinfo) {
      dinfo->deviceType = synDeviceType::synDeviceGaudi;
      return synSuccess;
    };

    synapse_api_.synDeviceGetInfoV2 = [](const synDeviceId,
                                         synDeviceInfoV2* dinfo) {
      dinfo->deviceType = synDeviceType::synDeviceGaudi;
      dinfo->deviceIndex = 0;
      return synSuccess;
    };

    synapse_api_.synRecipeGetAttribute =
        [](uint64_t* retVal,
           const synRecipeAttribute* recipeAttr,
           const unsigned querySize,
           const synRecipeHandle) {
          if (querySize == 1 && *recipeAttr == RECIPE_ATTRIBUTE_HOST_MEM_SIZE)
            *retVal = Consts::RECIPE_SIZE;
          return synSuccess;
        };

    synapse_api_.synDeviceGetAttribute =
        [](uint64_t* retVal,
           const synDeviceAttribute* deviceAttr,
           const unsigned querySize,
           const synDeviceId) {
          if (querySize == 1 &&
              *deviceAttr == DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE)
            *retVal = Consts::STREAMS_TOTAL_MEM_SIZE;
          return synSuccess;
        };
  }

 private:
  absl::flat_hash_set<AnyResource*> handles_;
  std::deque<AnyResource> resources_;
  std::mutex resource_mtx_;
  uint64_t allocation_back_;
  synDeviceId device_id_{Consts::DEVICE_ID};
  synNodeId node_id_{Consts::NODE_ID};

  uint64_t Malloc(uint64_t size) {
    uint64_t addr = allocation_back_;
    allocation_back_ += size * Consts::DEVICE_MALLOC_ALIGNMENT;
    return addr;
  }

  synDeviceId AllocateDevice() {
    return device_id_++;
  }
  synNodeId AllocateNodeId() {
    return node_id_++;
  }

  template <typename SynHandle, typename ResourceType = Resource<SynHandle>>
  ResourceType& HandleToResource(SynHandle handle) {
    {
      std::lock_guard<std::mutex> lock(resource_mtx_);
      CHECK_TRUE_MSG(
          handles_.contains(reinterpret_cast<AnyResource*>(handle)),
          "invalid NullHw handle " << handle);
    }
    AnyResource& resource{*reinterpret_cast<AnyResource*>(handle)};
    CHECK_TRUE_MSG(
        absl::holds_alternative<ResourceType>(resource),
        " NullHw handle " << handle << " has unexpected type");
    return absl::get<ResourceType>(resource);
  }

  template <
      typename SynHandle,
      typename ResourceType = Resource<SynHandle>,
      typename... Args>
  void AddHandle(SynHandle* out, Args... args) {
    std::lock_guard<std::mutex> lock(resource_mtx_);
    resources_.emplace_back(ResourceType{args...});
    handles_.insert(&resources_.back());
    *out = reinterpret_cast<SynHandle>(&resources_.back());
  }
};
} // namespace null_hw

void EnableNullHw();
