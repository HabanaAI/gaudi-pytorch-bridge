/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include <synapse_api_types.h>

namespace synapse_helpers {

class device;

using device_ptr = std::uint64_t;
static constexpr auto device_nullptr = device_ptr{};

/**
 * RAII wrapper for pointers allocated on the device.
 *
 * Requires that the device will outlive any pointer that it owns.
 */
/* Implementation details:
 * It uses reinterpret_casting in order to use a unique_ptr as internal storage
 * for the pointer and its deleter. It's a dirty hack, but used only internally
 * and not exposed through this classes interface.
 */
class owned_device_ptr {
 public:
  owned_device_ptr(device_ptr buffer_ptr, size_t size, device& dev)
      : ptr_{reinterpret_cast<device_ptr*>(buffer_ptr), device_ptr_deleter{dev}},
        size_(size) {}

  device_ptr get() const {
    return reinterpret_cast<device_ptr>(ptr_.get());
  }
  device_ptr release() {
    return reinterpret_cast<device_ptr>(ptr_.release());
  }
  size_t size() const {
    return size_;
  }

  explicit operator bool() {
    return bool(ptr_);
  }

 private:
  class device_ptr_deleter {
   public:
    explicit device_ptr_deleter(device& device) : device_{&device} {}

    void operator()(device_ptr* ptr);

   private:
    device* device_;
  };

  std::unique_ptr<device_ptr, device_ptr_deleter> ptr_;
  size_t size_;
};

class device_ptr_lock_interface {
 public:
  using iterator_t = device_ptr*;
  virtual ~device_ptr_lock_interface() = default;
  virtual iterator_t begin() = 0;
  virtual iterator_t end() = 0;
  virtual device_ptr at(size_t) = 0;
};

class device_ptr_lock {
 public:
  using iterator_t = device_ptr_lock_interface::iterator_t;
  device_ptr_lock(std::unique_ptr<device_ptr_lock_interface>&& impl)
      : impl_(std::move(impl)) {}
  device_ptr_lock(device_ptr_lock&& other) noexcept {
    impl_ = std::move(other.impl_);
  }
  device_ptr_lock& operator=(device_ptr_lock&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
  }

  // bool operator==(const device_ptr_lock& other) { return (impl_ ==
  // other.impl_); }
  iterator_t begin() {
    return impl_->begin();
  }
  iterator_t end() {
    return impl_->end();
  }
  device_ptr at(size_t position) {
    return impl_->at(position);
  }

 private:
  std::unique_ptr<device_ptr_lock_interface> impl_;
};

class device_allocator {
 public:
  virtual ~device_allocator() = default;
  virtual void release() = 0;
  virtual void reset() = 0;
  virtual void* alloc(std::size_t size_bytes) = 0;
  virtual void free(void* pointer) = 0;
};

using create_allocator_fnc =
    std::function<std::unique_ptr<device_allocator>(synDeviceId device_id)>;
using framework_specific_cleanup_fnc = std::function<void()>;

using device_handle = std::shared_ptr<device>;

} // namespace synapse_helpers
