#include "forked_pybind_utils.h"
#include <pybind11/pybind11.h>

namespace habana_torch::jit {

// This is a hack to remove instances deleted in C++ from the PyBind cache
// C++->Python. We need this because otherwise we may get the old Python object
// if C++ creates a new object at the memory location of the deleted object.
void clear_registered_instances(void* ptr) {
#if IS_PYBIND_2_13_PLUS
  py::detail::with_instance_map(
      ptr, [&](py::detail::instance_map& registered_instances) {
        auto range = registered_instances.equal_range(ptr);
        for (auto it = range.first; it != range.second; ++it) {
          auto vh = it->second->get_value_and_holder();
          vh.set_instance_registered(false);
        }
        registered_instances.erase(ptr);
      });
#else
  auto& registered_instances =
      pybind11::detail::get_internals().registered_instances;
  auto range = registered_instances.equal_range(ptr);
  for (auto it = range.first; it != range.second; ++it) {
    auto vh = it->second->get_value_and_holder();
    vh.set_instance_registered(false);
  }
  registered_instances.erase(ptr);
#endif
}
} // namespace habana_torch::jit
