#pragma once

#include <future>
#include <iostream>
#include <set>
#include <string>

#include <memory>

#include <threadpool.hpp>
#include <types.hpp>

#include <comparison_methods.hpp>
#include <tensor_comparator.hpp>
namespace TensorComparison_pt {
class Comparator {
 public:
  Comparator(unsigned numThreads = 0)
      : m_pool{numThreads ? numThreads : std::thread::hardware_concurrency()} {}

  template <typename ReferenceDataType, typename TensorDataType>
  static ComparisonResult _compare(
      ReferenceDataType* expected,
      TensorDataType* result,
      unsigned length,
      ComparisonMethods compareMethods) {
    ComparisonResult r;
    if (compareMethods.test(ANGLE)) {
      r[ANGLE].set(
          TestMethods<ReferenceDataType, TensorDataType>::calcCosineSimilarity(
              expected, result, length));
    }
    if (compareMethods.test(PEARSON)) {
      r[PEARSON].set(
          TestMethods<ReferenceDataType, TensorDataType>::calcPearson(
              expected, result, length));
    }
    if (compareMethods.test(L2_NORM)) {
      r[L2_NORM].set(
          TestMethods<ReferenceDataType, TensorDataType>::calcL2NormRatio(
              expected, result, length));
    }
    if (compareMethods.test(MSE)) {
      r[MSE].set(TestMethods<ReferenceDataType, TensorDataType>::calcMse(
          expected, result, length));
    }
    if (compareMethods.test(STD_DEV)) {
      r[STD_DEV].set(TestMethods<ReferenceDataType, TensorDataType>::calcStdev(
          expected, result, length));
    }
    if (compareMethods.test(ABS_ERR_MAX)) {
      r[ABS_ERR_MAX].set(
          TestMethods<ReferenceDataType, TensorDataType>::calcMaxAbsError(
              expected, result, length));
    }
    if (compareMethods.test(ABS_ERR_MIN)) {
      r[ABS_ERR_MIN].set(
          TestMethods<ReferenceDataType, TensorDataType>::calcMinAbsError(
              expected, result, length));
    }
    if (compareMethods.test(ABS_ERR_PERCENT2)) {
      float percent = 1e-2;
      r[ABS_ERR_PERCENT2].set(
          TestMethods<ReferenceDataType, TensorDataType>::
              calcAbsErrorPercentile(expected, result, length, percent));
    }
    if (compareMethods.test(ABS_ERR_PERCENT3)) {
      float percent = 1e-3;
      r[ABS_ERR_PERCENT3].set(
          TestMethods<ReferenceDataType, TensorDataType>::
              calcAbsErrorPercentile(expected, result, length, percent));
    }
    if (compareMethods.test(ABS_ERR_PERCENT4)) {
      float percent = 1e-4;
      r[ABS_ERR_PERCENT4].set(
          TestMethods<ReferenceDataType, TensorDataType>::
              calcAbsErrorPercentile(expected, result, length, percent));
    }
    if (compareMethods.test(ABS_ERR_PERCENT5)) {
      float percent = 1e-5;
      r[ABS_ERR_PERCENT5].set(
          TestMethods<ReferenceDataType, TensorDataType>::
              calcAbsErrorPercentile(expected, result, length, percent));
    }
    if (compareMethods.test(AVG_ERR)) {
      r[AVG_ERR].set(
          TestMethods<ReferenceDataType, TensorDataType>::calcAvgError(
              expected, result, length));
    }
    if (compareMethods.test(STD_DEV_VS_AVG_ERR)) {
      r[STD_DEV_VS_AVG_ERR].set(
          TestMethods<ReferenceDataType, TensorDataType>::calcAvgErrVsStdDev(
              expected, result, length));
    }
    return r;
  }

  template <typename ReferenceDataType, typename TensorDataType>
  std::future<ComparisonResult> compare(
      ReferenceDataType* expected,
      TensorDataType* result,
      unsigned length,
      ComparisonMethods compareMethods = {}) {
    auto ret = m_pool.enqueue(
        Comparator::_compare<ReferenceDataType, TensorDataType>,
        expected,
        result,
        length,
        compareMethods);
    return ret;
  }

 private:
  ThreadPool m_pool;
};
} // namespace TensorComparison_pt