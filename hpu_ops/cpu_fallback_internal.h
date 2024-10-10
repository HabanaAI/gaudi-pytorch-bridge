/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#pragma once
// clang-format off
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/IListRef.h>
#include <ATen/Operators.h>
#include <ATen/autocast_mode.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/result_type.h>
#include <ATen_ver/native/CPUFallback.h>
#include "habana_kernels/op_support_level.h"
#include "habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"
// clang-format on

namespace habana {

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

namespace detail {

/* detect that two arguments are different.
 * We are only interested in at::Tensor. For a generic case just
 * shortcut to tell that it is the same.
 */
template <typename T>
inline bool is_unchanged(const T&, const T&) {
  return true;
}

inline bool is_unchanged(const at::Tensor& t1, const at::Tensor& t2) {
  return t1.unsafeGetTensorImpl() == t2.unsafeGetTensorImpl();
}

inline bool is_unchanged(
    const c10::optional<at::Tensor>& t1,
    const c10::optional<at::Tensor>& t2) {
  if (t1.has_value())
    if (t2.has_value())
      return t1->unsafeGetTensorImpl() == t2->unsafeGetTensorImpl();
    else
      return false;
  else
    return !t2.has_value();
}

template <typename ContainerType>
inline bool is_unchanged(
    const ContainerType& t1,
    const std::vector<at::Tensor>& t2) {
  assert(t1.size() == t2.size());
  auto p2{t2.cbegin()};
  for (const auto& t : t1) {
    if (t.unsafeGetTensorImpl() != p2->unsafeGetTensorImpl())
      return false;
    ++p2;
  }
  return true;
}

template <typename T>
inline bool is_eligible_for_redispatch(const T&) {
  return true;
}

inline bool is_eligible_for_redispatch(const at::Tensor& t) {
  return t.is_floating_point();
}

inline bool is_eligible_for_redispatch(const c10::optional<at::Tensor>& t) {
  if (t.has_value())
    return t->is_floating_point();
  return true;
}

inline bool is_eligible_for_redispatch(const c10::ArrayRef<at::Tensor>& tar) {
  for (const auto& tensor : tar) {
    if (!tensor.is_floating_point())
      return false;
  }
  return true;
}

/* currently only requires_grad is the attribute which is
 * set incorrectly. so setting only requires_grad
 * here
 */
// Overload to process Tensor
inline void set_attribute(const at::Tensor& arg, at::Tensor tensor) {
  tensor.set_requires_grad(arg.requires_grad());
}

// Overload to process optional<Tensor>
inline void set_attribute(
    const c10::optional<at::Tensor>& arg,
    c10::optional<at::Tensor> tensor) {
  if (arg.has_value() && tensor.has_value()) {
    (tensor.value()).set_requires_grad((arg.value()).requires_grad());
  }
}

// Overload to process TensorLists
inline void set_attribute(
    const at::TensorList& args,
    std::vector<at::Tensor> tensors) {
  int i = 0;
  for (const auto& t : args) {
    tensors[i].set_requires_grad(t.requires_grad());
    i++;
  }
}

inline void set_attribute(
    const at::ITensorListRef& args,
    std::vector<at::Tensor> tensors) {
  int i = 0;
  for (const auto& t : args) {
    tensors[i].set_requires_grad(t.requires_grad());
    i++;
  }
}

// Template to catch non-Tensor args.
template <typename T>
void set_attribute([[maybe_unused]] T arg, [[maybe_unused]] T arg1) {}

/**
 * cast argument and detect if cast was needed.
 * Underlying cached_cast may pass an argument as-is if it is not eligible for
 * casting. If this is the case for all the arguments then dispatch op to CPU to
 * prevent endless-fallback loop.
 */
template <typename T>
inline T cast_arg(bool& arg_changed, at::ScalarType dtype, T arg) {
  PT_FALLBACK_TRACE
  auto after{at::autocast::cached_cast(dtype, arg, at::DeviceType::HPU)};
  bool did_cast = !is_unchanged(arg, after);
  arg_changed |= did_cast;
  set_attribute(arg, after);
  return after;
}

inline at::Tensor cast_arg(
    bool& arg_changed,
    at::ScalarType dtype,
    const at::Tensor& arg) {
  PT_FALLBACK_TRACE

  auto after{at::autocast::cached_cast(dtype, arg, at::DeviceType::HPU)};
  bool did_cast = !is_unchanged(arg, after);
  PT_FALLBACK_DEBUG(
      "arg_changed on entry=", arg_changed, " did_cast=", did_cast);
  arg_changed |= did_cast;
  set_attribute(arg, after);
  return after;
}

template <typename ResultType, typename ResultIndexes = void>
struct cast_result final {
  static ResultType cast(at::ScalarType dtype, ResultType result) {
    auto t = at::autocast::cached_cast(dtype, result, at::DeviceType::HPU);
    set_attribute(result, t);
    return t;
  }
};

template <typename... ResultTypes, size_t... RIs>
struct cast_result<std::tuple<ResultTypes...>, std::index_sequence<RIs...>>
    final {
  using ResultType = std::tuple<ResultTypes...>;
  static ResultType cast(at::ScalarType dtype, ResultType result) {
    return {cast_result<at::Tensor>::cast(dtype, std::get<RIs>(result))...};
  }
};

template <typename... ResultTypes>
struct cast_result<std::tuple<ResultTypes...>> final {
  using ResultType = std::tuple<ResultTypes...>;
  const size_t RetCount = sizeof...(ResultTypes);
  static ResultType cast(at::ScalarType dtype, ResultType result) {
    return cast_result<
        ResultType,
        std::make_index_sequence<sizeof...(ResultTypes)>>::cast(dtype, result);
  }
};

template <class Op, class ReturnType, class... ParameterTypes>
struct redispatch_if_any_arg_changed final {
  static ReturnType call(bool redispatch_to_hpu, ParameterTypes... args) {
    if (redispatch_to_hpu) {
      return Op::call(args...);
    }
    return at_ver::native::call_fallback_fn_symint<&cpu_fallback, Op>::call(
        args...);
  }
};

/* helper to guess result dtype in the cases that output tensor is missing.
 */
template <class... ParameterTypes>
at::ScalarType expected_result_dtype(const at::Tensor& t, ParameterTypes...) {
  return t.scalar_type();
}

template <class... ParameterTypes>
at::ScalarType expected_result_dtype(
    const at::Tensor& t,
    const at::Scalar& s,
    ParameterTypes...) {
  return at::result_type(t, s);
}

template <class... ParameterTypes>
at::ScalarType expected_result_dtype(
    const at::Tensor& t1,
    const at::Tensor& t2,
    ParameterTypes...) {
  return at::result_type(t1, t2);
}

template <class... ParameterTypes>
at::ScalarType expected_result_dtype(
    const at::Scalar& s,
    const at::Tensor& t,
    ParameterTypes...) {
  return at::result_type(s, t);
}

template <class... ParameterTypes>
at::ScalarType expected_result_dtype(const at::Scalar& s, ParameterTypes...) {
  return s.type();
}

template <class... ParameterTypes>
at::ScalarType expected_result_dtype(
    const at::Scalar& s1,
    const at::Scalar& s2,
    ParameterTypes...) {
  return at::result_type(s1, s2);
}

template <class... ParameterTypes>
at::ScalarType expected_result_dtype(
    const c10::ArrayRef<at::Tensor>& tarr,
    ParameterTypes...) {
  return tarr[0].scalar_type();
}

template <class... ParameterTypes>
at::ScalarType expected_result_dtype(
    const c10::IListRef<at::Tensor>& tarr,
    ParameterTypes...) {
  return tarr.front().scalar_type();
}
/*
 * Wrapper that executes a fallback path when op cannot be executed as is.
 * Partial specializations follow patterns of BoxedKernelWrapper (see
 * ATen/core/boxing/impl/boxing.h).
 *
 * Fallback path may be excercised because of multiple reasons, of which
 * unsupported argument dtype is slightly more nuanced. Some generations of
 * hardware may not support certain low precision floating point types. Types
 * such as fp16 or fp8 or bf16 are also not well supported by the CPU. To
 * proceed in such case arguments are cast to a higher precision fp32 and
 * redispatched. It is likely that once arguments are high precision, such an op
 * will execute on HPU. Only if it is rejected again it will run on CPU. There
 * is an extra step needed for operators that mutate an existing tensor. This
 * includes in-place operators that mutate self and out-of-place oprators that
 * change preexisting output tensor/tensors.
 */
template <class Op, class FuncType, class Enable = void>
struct _dispatch_fallback final {};

template <class Op, class ReturnType, class... ParameterTypes>
struct _dispatch_fallback<
    Op,
    ReturnType(ParameterTypes...),
    std::enable_if_t<
        c10::impl::can_unbox<ReturnType>::value &&
            !c10::impl::is_tuple_of_mutable_tensor_refs<ReturnType>::value,
        void>>
    final {
  static ReturnType call(OpSupportLevel osl, ParameterTypes... args) {
    return call(osl, expected_result_dtype(args...), args...);
  }

  static ReturnType call(
      OpSupportLevel osl,
      at::ScalarType result_dtype,
      ParameterTypes... args) {
    const std::index_sequence_for<ParameterTypes...> indices{};
    if constexpr (std::is_void_v<ReturnType>)
      call(osl, result_dtype, args..., indices);
    else
      return ::std::move(call(osl, result_dtype, args..., indices));
  }

  template <std::size_t... Indices>
  static ReturnType call(
      OpSupportLevel osl,
      [[maybe_unused]] at::ScalarType result_dtype,
      ParameterTypes... args,
      std::index_sequence<Indices...>) {
    PT_FALLBACK_TRACE
    assert(!osl); // should never be called for op that is supported
    if (osl == OpSupportLevel::Value::unsupported_dtype) {
      if ((is_eligible_for_redispatch(args) && ...)) {
        bool arg_changed = false;
        // cast_args evaluate first, each may toggle `arg_changed` that is
        // eventually passed to redispatch
        auto params = std::make_tuple(args...);
        auto partial_cast_arg = [&arg_changed](auto... argument) {
          return std::make_tuple(
              cast_arg(arg_changed, at::ScalarType::Float, argument)...);
        };
        auto casted_params = std::apply(partial_cast_arg, params);
        if constexpr (std::is_void_v<ReturnType>) {
          redispatch_if_any_arg_changed<Op, ReturnType, ParameterTypes...>::
              call(arg_changed, std::get<Indices>(casted_params)...);
          return;
        } else {
          auto result{
              redispatch_if_any_arg_changed<Op, ReturnType, ParameterTypes...>::
                  call(arg_changed, std::get<Indices>(casted_params)...)};
          if (arg_changed)
            return cast_result<ReturnType>::cast(result_dtype, result);
          else
            return result;
        }
      }
    }

    return at_ver::native::call_fallback_fn_symint<&cpu_fallback, Op>::call(
        args...);
  }
};

/*
 * Specialization of fallback dispatcher for inplace ops.
 */
template <class Op, class... ParameterTypes>
struct _dispatch_fallback<Op, at::Tensor&(at::Tensor&, ParameterTypes...)>
    final {
  static at::Tensor& call(
      OpSupportLevel osl,
      at::Tensor& t,
      ParameterTypes... args) {
    PT_FALLBACK_TRACE
    assert(!osl); // should never be called for op that is supported
    if (osl == OpSupportLevel::Value::unsupported_dtype) {
      if (is_eligible_for_redispatch(t) &&
          (is_eligible_for_redispatch(args) && ...)) {
        bool arg_changed = false;
        at::Tensor cast_input{cast_arg(arg_changed, at::ScalarType::Float, t)};
        at::Tensor& new_tensor{
            redispatch_if_any_arg_changed<
                Op,
                at::Tensor&,
                at::Tensor&,
                ParameterTypes...>::
                call(
                    arg_changed,
                    cast_input,
                    cast_arg(arg_changed, at::ScalarType::Float, args)...)};
        if (arg_changed) {
          t = at::autocast::cached_cast(
              t.scalar_type(), new_tensor, at::DeviceType::HPU);
          set_attribute(cast_input, t);
          return t;
        } else {
          t = new_tensor;
          set_attribute(cast_input, t);
          return t;
        }
      }
    }

    return at_ver::native::call_fallback_fn_symint<&cpu_fallback, Op>::call(
        t, args...);
  }
};

/*
 * Specialization of fallback dispatcher for inplace ops that use const Tensor
 * references.
 *
 */
template <class Op, class... ParameterTypes>
struct _dispatch_fallback<
    Op,
    const at::Tensor&(const at::Tensor&, ParameterTypes...)>
    final {
  static const at::Tensor& call(
      OpSupportLevel osl,
      const at::Tensor& t,
      ParameterTypes... args) {
    PT_FALLBACK_TRACE
    assert(!osl); // should never be called for op that is supported
    if (osl == OpSupportLevel::Value::unsupported_dtype) {
      if (is_eligible_for_redispatch(t) &&
          (is_eligible_for_redispatch(args) && ...)) {
        bool arg_changed = false;
        at::Tensor cast_input{cast_arg(arg_changed, at::ScalarType::Float, t)};
        const at::Tensor& new_tensor{
            redispatch_if_any_arg_changed<
                Op,
                const at::Tensor&,
                const at::Tensor&,
                ParameterTypes...>::
                call(
                    arg_changed,
                    cast_input,
                    cast_arg(arg_changed, at::ScalarType::Float, args)...)};
        if (arg_changed) {
          t.copy_(at::autocast::cached_cast(
              t.scalar_type(), new_tensor, at::DeviceType::HPU));
        } else {
          t.copy_(new_tensor);
        }
        set_attribute(cast_input, t);
        return t;
      }
    }

    return at_ver::native::call_fallback_fn_symint<&cpu_fallback, Op>::call(
        t, args...);
  }
};

void submit_result(at::Tensor& src, at::Tensor& dst);

at::Tensor& prepare_out(
    at::Tensor& from,
    at::Tensor& copy,
    at::ScalarType float_dtype);

/* out_of_place ops have one or more mutable tensors passed as final
 * arguments. dispatch_out_of_place is a helper that allows to separate and
 * cast input arguments while keeping output args intact.
 **/
template <
    class Op,
    typename ReturnType,
    typename InTypeList,
    typename OutTypeList,
    class OutIndexSequence>
struct dispatch_out_of_place {};

template <
    class Op,
    typename ReturnType,
    typename... InArgs,
    typename... OutArgs,
    size_t... OutArgsIs>
struct dispatch_out_of_place<
    Op,
    ReturnType,
    c10::guts::typelist::typelist<InArgs...>,
    c10::guts::typelist::typelist<OutArgs...>,
    std::index_sequence<OutArgsIs...>> {
  static ReturnType call(InArgs... args, OutArgs... out_args) {
    bool arg_changed{false};

    ReturnType result{out_args...};
    std::array<at::Tensor, sizeof...(OutArgs)> out_cast_copy{};
    ReturnType out_cast{prepare_out(
        std::get<OutArgsIs>(result),
        out_cast_copy[OutArgsIs],
        at::ScalarType::Float)...};
    [[maybe_unused]] const auto result_before_cast{
        redispatch_if_any_arg_changed<Op, ReturnType, InArgs..., OutArgs...>::
            call(
                arg_changed,
                cast_arg(arg_changed, at::ScalarType::Float, args)...,
                std::get<OutArgsIs>(out_cast)...)};
    (submit_result(std::get<OutArgsIs>(out_cast), std::get<OutArgsIs>(result)),
     ...);

    return result;
  }
};

/*
 * Specialization of fallback dispatcher for out-of-place op with single out
 * tensor
 */
template <class Op, class FirstArg, class... ParameterTypes>
struct _dispatch_fallback<
    Op,
    at::Tensor&(FirstArg, ParameterTypes...),
    std::enable_if_t<!c10::impl::is_mutable_tensor_ref<FirstArg>::value, void>>
    final {
  static constexpr int RetCount = 1;
  static constexpr int InArgCount =
      sizeof...(ParameterTypes) - RetCount + 1; //+1 to account for FirstArg

  using all_args = c10::guts::typelist::typelist<FirstArg, ParameterTypes...>;
  using in_args = c10::guts::typelist::take_t<all_args, InArgCount>;
  using out_args = c10::guts::typelist::drop_t<all_args, InArgCount>;
  using helper = dispatch_out_of_place<
      Op,
      std::tuple<at::Tensor&>,
      in_args,
      out_args,
      std::make_index_sequence<RetCount>>;

  static at::Tensor& call(
      OpSupportLevel osl,
      FirstArg arg,
      ParameterTypes... args) {
    PT_FALLBACK_TRACE
    assert(!osl); // should never be called for op that is supported
    if (osl == OpSupportLevel::Value::unsupported_dtype) {
      if (is_eligible_for_redispatch(arg) &&
          (is_eligible_for_redispatch(args) && ...))
        return std::get<0>(helper::call(arg, args...));
    }

    return at_ver::native::call_fallback_fn_symint<&cpu_fallback, Op>::call(
        arg, args...);
  }
};

/*
 * Specialization of fallback dispatcher for out-of-place ops returning tuple
 * of modified tensors.
 */
template <class Op, class ReturnType, class... ParameterTypes>
struct _dispatch_fallback<
    Op,
    ReturnType(ParameterTypes...),
    std::enable_if_t<
        c10::impl::is_tuple_of_mutable_tensor_refs<ReturnType>::value,
        void>>
    final {
  static constexpr int RetCount = std::tuple_size<ReturnType>();
  static constexpr int InArgCount = sizeof...(ParameterTypes) - RetCount;
  using all_args = c10::guts::typelist::typelist<ParameterTypes...>;
  using in_args = c10::guts::typelist::take_t<all_args, InArgCount>;
  using out_args = c10::guts::typelist::drop_t<all_args, InArgCount>;
  using helper = dispatch_out_of_place<
      Op,
      ReturnType,
      in_args,
      out_args,
      std::make_index_sequence<RetCount>>;

  static ReturnType call(OpSupportLevel osl, ParameterTypes... args) {
    PT_FALLBACK_TRACE
    assert(!osl); // should never be called for op that is supported
    if (osl == OpSupportLevel::Value::unsupported_dtype) {
      if ((is_eligible_for_redispatch(args) && ...))
        return helper::call(args...);
    }

    return at_ver::native::call_fallback_fn_symint<&cpu_fallback, Op>::call(
        args...);
  }
};

} // namespace detail
} // namespace habana
