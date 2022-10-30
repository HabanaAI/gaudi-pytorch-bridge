#include <c10/core/DeviceGuard.h>
#include <torch/csrc/api/include/torch/version.h>
#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 13))
#include <c10d/reducer_timer.hpp>
#else
#include <torch/csrc/distributed/c10d/reducer_timer.hpp>
#endif
#include "habana_helpers/logging.h"

namespace c10d {
namespace {

class HpuTimer : public Timer {
 public:
  explicit HpuTimer(c10::Device /* unused */) {}

  c10::optional<int64_t> measureDifference(Event start, Event end) override {
    int64_t start_time = getTimeRef(start);
    int64_t end_time = getTimeRef(end);
    PT_DISTRIBUTED_DEBUG("start time::", start_time, " End time::", end_time);
    if (end_time < start_time) {
      return c10::nullopt;
    }
    return end_time - start_time;
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kHPU, HpuTimer);

} // namespace
} // namespace c10d
