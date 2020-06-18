/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

namespace synh {
/**
 * Wrapper over std::for_each.
 *
 * @param container Container to loop over.
 * @param f Function to call for each element.
 */
template <typename T, typename UnaryFunction>
UnaryFunction for_each(const T& container, UnaryFunction f) {
  return std::for_each(container.begin(), container.end(), f);
}

/**
 * Wrapper over std::count_if.
 *
 * @param container Container to loop over.
 * @param p Predicate to check when counting.
 */
template <typename T, typename Predicate>
auto count_if(const T& container, Predicate p) -> typename std::iterator_traits<
    decltype(container.begin())>::difference_type {
  return std::count_if(container.begin(), container.end(), p);
}

/**
 * Wrapper over std::none_of
 *
 * @param container Container to loop over.
 * @param p Predicate to check when counting
 */
template <typename T, typename UnaryPredicate>
bool none_of(const T& container, UnaryPredicate p) {
  return std::none_of(container.begin(), container.end(), p);
}

/**
 * Wrapper over std::find_if
 *
 * @param container Container to loop over.
 * @param p Predicate to check when counting
 */
template <typename T, typename UnaryPredicate>
auto find_if(const T& container, UnaryPredicate p)
    -> decltype(container.begin()) {
  return std::find_if(container.begin(), container.end(), p);
}
} // namespace synh