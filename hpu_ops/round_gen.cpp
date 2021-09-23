#include "generated/hpu_op.h"

namespace habana {
std::shared_ptr<void> FillRoundParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_RoundKernel::Params);
  constexpr unsigned short constExpectedNoOfInput = 1;
  TORCH_CHECK(
      stack.size() == constExpectedNoOfInput,
      "Expected ",
      constExpectedNoOfInput,
      " input for Round Operator"
      " but received ",
      stack.size(),
      " inputs.");
  TORCH_CHECK(stack[0].isTensor(), "Input type expected to be tensor");
  params->roundMode = RoundMode_t::ROUND_HALF_NEAREST_EVEN;
  return params;
}
} // namespace habana