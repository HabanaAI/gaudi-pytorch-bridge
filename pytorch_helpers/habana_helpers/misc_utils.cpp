
#include "misc_utils.h"
#include <ATen/Tensor.h>

namespace habana_lazy {

bool IsCollective(const c10::Symbol& symbol) {
  static c10::Symbol hccl_namepsace =
      c10::Symbol::fromQualString("namespaces::hccl");
  return symbol.ns() == hccl_namepsace;
}

} // namespace habana_lazy
