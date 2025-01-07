/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <algorithm>

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
 * @param p Predicate to check.
 */
template <typename T, typename UnaryPredicate>
bool none_of(const T& container, UnaryPredicate p) {
  return std::none_of(container.begin(), container.end(), p);
}

/**
 * Wrapper over std::any_of
 *
 * @param container Container to loop over.
 * @param p Predicate to check.
 */
template <typename T, typename UnaryPredicate>
bool any_of(const T& container, UnaryPredicate p) {
  return std::any_of(container.begin(), container.end(), p);
}

/**
 * Wrapper over std::all_of
 *
 * @param container Container to loop over.
 * @param p Predicate to check.
 */
template <typename T, typename UnaryPredicate>
bool all_of(const T& container, UnaryPredicate p) {
  return std::all_of(container.begin(), container.end(), p);
}

/**
 * Wrapper over std::find
 *
 * @param container Container to loop over.
 * @param p Value to check when searching.
 */
template <typename T, typename U>
auto find(const T& container, U value) -> decltype(container.begin()) {
  return std::find(container.begin(), container.end(), value);
}

/**
 * @brief Returns true if a collection contains an item equal to the given
 * value.
 *
 * @tparam T         Type of the container
 * @tparam U         Type of the value item
 * @param container  The collection to search in
 * @param value      The value to search for
 * @return           True if container contains value, false otherwise
 */
template <typename T, typename U>
bool contains(const T& container, U value) {
  return std::find(container.begin(), container.end(), value) !=
      container.end();
}

/**
 * Wrapper over std::find_if
 *
 * @param container Container to loop over.
 * @param p Predicate to check when searching.
 */
template <typename T, typename UnaryPredicate>
auto find_if(const T& container, UnaryPredicate p)
    -> decltype(container.begin()) {
  return std::find_if(container.begin(), container.end(), p);
}

/**
 * @brief Performs an operation on each two adjacent elements of a given range.
 * @param first The beginning of the range
 * @param last The end of the range
 * @param binary_op The operation to be performed
 */
template <typename ForwardIt, typename BinaryOperation>
void adjacent_for_each(
    ForwardIt first,
    ForwardIt last,
    BinaryOperation binary_op) {
  using value_type = decltype(*first);
  std::adjacent_find(first, last, [&](value_type& left, value_type& right) {
    binary_op(left, right);
    return false;
  });
}

/**
 * @brief Performs an operation on each two adjacent (in order of iteration)
 * elements of a given container.
 * @param container The container to perform the operation on
 * @param binary_op The operation to be performed
 */
template <typename Container, typename BinaryOperation>
void adjacent_for_each(const Container& container, BinaryOperation binary_op) {
  adjacent_for_each(std::begin(container), std::end(container), binary_op);
}

/**
 * @brief Perform a binary search for an element.
 *
 * @param first  iterator to beginning of the range to search. The range must be
 * partitioned with respect to op<.
 * @param last   iterator to the end of the range to search.
 * @param elem   the element to search for
 */
template <typename It, typename T>
It binary_find(It first, It last, const T& elem) {
  auto it = std::lower_bound(first, last, elem);
  if (it == last || elem < *it)
    return last;
  return it;
}

/**
 * @brief Perform a binary search for an element.
 *
 * @param container  container to search. Must be partitioned with respect to
 * op<.
 * @param elem       the element to search for
 */
template <typename Container, typename T>
auto binary_find(const Container& container, const T& elem)
    -> decltype(std::begin(container)) {
  return binary_find(std::begin(container), std::end(container), elem);
}

/**
 * @brief Perform a binary search for an element.
 *
 * @param first  iterator to beginning of the range to search. The range must be
 * partitioned with respect to \p comp.
 * @param last   iterator to the end of the range to search.
 * @param elem   the element to search for
 * @param comp   the comparator used to sort the range
 */
template <typename It, typename T, typename Comp>
It binary_find(It first, It last, const T& elem, Comp comp) {
  auto it = std::lower_bound(first, last, elem, comp);
  if (it == last || comp(elem, *it))
    return last;
  return it;
}

/**
 * @brief Perform a binary search for an element.
 *
 * @param container  container to search. Must be partitioned with respect to \p
 * comp.
 * @param elem       the element to search for
 * @param comp   the comparator used to sort the range
 */
template <typename Container, typename T, typename Comp>
auto binary_find(const Container& container, const T& elem, Comp comp)
    -> decltype(std::begin(container)) {
  return binary_find(std::begin(container), std::end(container), elem, comp);
}

/**
 * @brief   Checks if two ranges have a common element
 *
 * @tparam It           the type of iterator to use
 * @param first_begin   beginning of the first range to check (range not
 * required to be sorted)
 * @param first_end     end of the first range to check
 * @param second_begin  beginning of the second range to check (range not
 * required to be sorted)
 * @param second_end    end of the second range to check
 * @return              whether there is at least one equal element in both
 * ranges
 */
template <typename It>
bool have_common_element(
    It first_begin,
    It first_end,
    It second_begin,
    It second_end) {
  for (; first_begin != first_end; ++first_begin) {
    for (auto second_it = second_begin; second_it != second_end; ++second_it) {
      if (*first_begin == *second_it)
        return true;
    }
  }
  return false;
}

/**
 * @brief  Checks if two containers have a common element
 *
 * @tparam Container  the type of the containers
 * @param first       first container to check (not required to be sorted)
 * @param second      second container to check (not required to be sorted)
 * @return            whether there is at least one equal element in both
 * containers
 */
template <typename Container>
bool have_common_element(const Container& first, const Container& second) {
  return have_common_element(
      std::begin(first), std::end(first), std::begin(second), std::end(second));
}

} // namespace synh