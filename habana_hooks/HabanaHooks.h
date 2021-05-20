#include <ATen/detail/HabanaHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>

namespace habana {

// The real implementation of HabanaHooksInterface
struct HabanaHooks : public at::HabanaHooksInterface {
  HabanaHooks(at::HabanaHooksArgs) {}
  bool isPinnedPtr(void* data) const override;
  bool hasHabana() const override;
  int64_t current_device() const override;
  at::Allocator* getPinnedMemoryAllocator() const override;
  int getNumGPUs() const override;
};

} // namespace habana
