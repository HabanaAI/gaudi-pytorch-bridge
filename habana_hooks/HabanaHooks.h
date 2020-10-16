#include <ATen/detail/HabanaHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>

namespace at {
namespace habana {
namespace detail {

// The real implementation of HabanaHooksInterface
struct HabanaHooks : public at::HabanaHooksInterface {
  HabanaHooks(at::HabanaHooksArgs) {}
  bool isPinnedPtr(void* data) const override;
  bool hasHabana() const override;
  int64_t current_device() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  int getNumGPUs() const override;
};

} // namespace detail
} // namespace habana
} // namespace at
