#pragma once

#include <synapse/include/synapse_api_types.h>
#include <synapse_helpers/device.h>
#include <array>
#include <memory>

namespace synapse_helpers {
class HPURegistrar {
  HPURegistrar() = default;
  std::array<std::unique_ptr<synapse_helpers::device>, MAX_DEVICES_PER_BOX>
      acquired_devices;
  static HPURegistrar& get_hpu_registrar() {
    static HPURegistrar instance;
    return instance;
  }

 public:
  HPURegistrar(HPURegistrar const&) = delete;
  void operator=(HPURegistrar const&) = delete;

  // This function always return initialized device
  static synapse_helpers::device& get_device(int device_id) {
    auto ret = get_hpu_registrar().acquired_devices.at(device_id).get();
    TORCH_CHECK(ret != nullptr, "Device ", device_id, "is not initialized");
    return *ret;
  }

  static synapse_helpers::device& get_device() {
    const auto& end = get_hpu_registrar().acquired_devices.end();
    auto ret = std::find_if(
        get_hpu_registrar().acquired_devices.begin(), end, [](auto& x) {
          return x.get() != nullptr;
        });

    TORCH_CHECK(ret != end, "Habana device not initialized");
    return *(ret->get());
  }

  static void insert_device(std::unique_ptr<synapse_helpers::device> device) {
    get_hpu_registrar().acquired_devices[device->id()] = std::move(device);
  }

  static bool empty() {
    for (auto& x : get_hpu_registrar().acquired_devices)
      if (x.get() != nullptr)
        return false;

    return true;
  }
};

} // namespace synapse_helpers