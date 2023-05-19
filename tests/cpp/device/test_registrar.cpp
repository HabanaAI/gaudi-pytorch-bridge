/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/device_types.h"
#include "pytorch_helpers/synapse_shim/null_hw_api.h"
#include "synapse_shim/null_hw_api.h"

using namespace habana;

class SynApiMock : public StubSynapseApi {
 public:
  SynApiMock() : StubSynapseApi() {
    synapse_api_.synDeviceAcquireByDeviceType = [this](
                                                    synDeviceId* id,
                                                    const synDeviceType type) {
      if (type != synDeviceGaudi2)
        return synFail;
      PT_TEST_DEBUG("mock allocating device via synDeviceAcquireByDeviceType");
      *id = 0;
      return synSuccess;
    };

    synapse_api_.synDeviceRelease = [this](synDeviceId id) {
      PT_TEST_DEBUG("mock releasing device ", id);
      return synSuccess;
    };

    synapse_api_.synHostMalloc = [this](
                                     const synDeviceId device,
                                     const uint64_t size,
                                     const uint32_t flags,
                                     void** buffer) {
      PT_TEST_DEBUG(
          "mock synHostMalloc(",
          device,
          ", ",
          size,
          ", ",
          flags,
          ", ",
          buffer,
          ")");
      std::vector<char> alloc(size);
      *buffer = alloc.data();
      buffers_.emplace_back(std::move(alloc));
      return synSuccess;
    };

    synapse_api_.synHostFree = [this](
                                   const synDeviceId deviceId,
                                   const void* buffer,
                                   const uint32_t flags) {
      PT_TEST_DEBUG(
          "mock synHostFree(", deviceId, ", ", buffer, ", ", flags, ")");
      return synSuccess;
    };
    synapse_api_.synDeviceMalloc = [this](
                                       const synDeviceId deviceId,
                                       const uint64_t size,
                                       uint64_t reqAddr,
                                       const uint32_t flags,
                                       uint64_t* buffer) {
      *buffer =
          0xf00b0000; // provide arbitrary pointer as result of device malloc
      PT_TEST_DEBUG(
          "mock synDeviceMalloc(",
          deviceId,
          ", ",
          size,
          ", ",
          reqAddr,
          ", ",
          flags,
          ", ",
          buffer,
          ")->",
          *buffer);
      return synSuccess;
    };
    synapse_api_.synDeviceFree = [this](
                                     const synDeviceId deviceId,
                                     const uint64_t buffer,
                                     const uint32_t flags) {
      PT_TEST_DEBUG(
          "mock synDeviceFree(", deviceId, ", ", buffer, ", ", flags, ")");
      return synSuccess;
    };
  }
  std::list<std::vector<char>> buffers_;
};

namespace habana {

/**
 * Temporarily wraps singleton hpu registrar with a tested instance allowing
 * for creation of dummy devices.
 */
class HPURegistrarTester {
 public:
  HPURegistrarTester()
      : allocator_active_device_id_{HPUDeviceAllocator::
                                        allocator_active_device_id},
        device_in_use_{synapse_helpers::device::device_in_use.lock()},
        orig_instance_{HPURegistrar::instance_.release()} {
    HPURegistrar::raw_instance_ = new HPURegistrar();
    HPURegistrar::instance_.reset(HPURegistrar::raw_instance_);
    synapse_helpers::device::device_in_use.reset();
    habana::hpu_registrar().get_or_create_device();
  }

  ~HPURegistrarTester() {
    PT_TEST_DEBUG(
        "restoring previous device ",
        device_in_use_.get(),
        " id ",
        allocator_active_device_id_);
    synapse_helpers::device::device_in_use = device_in_use_;
    device_in_use_.reset();
    HPURegistrar::instance_.swap(orig_instance_);
    HPURegistrar::raw_instance_ = HPURegistrar::instance_.get();
    HPUDeviceAllocator::allocator_active_device_id =
        allocator_active_device_id_;
    PinnedMemoryAllocator::allocator_active_device_id =
        allocator_active_device_id_;
  }

  void arm_late_action(std::function<void()>&& late_action) {
    HPURegistrar::instance_->test_inject_late_cleanup_ = std::move(late_action);
  }

  void release_registrar() {
    if (HPURegistrar::instance_ != nullptr) {
      PT_TEST_DEBUG(
          "releasing test instance of HPURegistrar ",
          HPURegistrar::instance_.get());
      HPURegistrar::instance_.reset();
    }
  }

  synDeviceId allocator_active_device_id_ = -1;
  std::shared_ptr<synapse_helpers::device> device_in_use_;

 private:
  std::unique_ptr<HPURegistrar> orig_instance_;
};

} // namespace habana

class RegistrarTest : public ::testing::Test {
  void SetUp() override {
    orig_syn_api_ = syn_api;
    orig_hccl_api_ = hccl_api;

    syn_api_mock_.Install();
  }

  void TearDown() override {
    syn_api = orig_syn_api_;
    hccl_api = orig_hccl_api_;
  }

  SynApiMock syn_api_mock_;
  synapse_api_t* orig_syn_api_;
  hccl_api_t* orig_hccl_api_;
};

TEST_F(RegistrarTest, test_late_cleanup) {
  auto registrar{std::make_unique<habana::HPURegistrarTester>()};
  ASSERT_EQ(habana::hpu_registrar().is_initialized(), true);
  torch::Tensor src = torch::empty({1024, 256});
  torch::Tensor dst = torch::empty({1024, 256}, c10::kHPU);
  registrar->arm_late_action([src = std::move(src), dst = std::move(dst)]() {
    PT_TEST_DEBUG("schedule dummy copy");
    dst.copy_(src, true);
  });

  PT_TEST_DEBUG("releasing test registrar");
  registrar->release_registrar();
  PT_TEST_DEBUG("done deleting test registrar");
}
