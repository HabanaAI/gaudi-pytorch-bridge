#include "HPUGuardImpl.h"

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(HABANA, HABANAGuardImpl);

}} // namespace at::detail
