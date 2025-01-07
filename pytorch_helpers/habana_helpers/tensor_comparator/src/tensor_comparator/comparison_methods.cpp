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

#include <bfloat16.h>
#include <comparison_methods.hpp>
#include <float16.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <type_traits>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wabsolute-value"
#include <Eigen/Dense>
#pragma GCC diagnostic pop

namespace Eigen {
template <>
struct NumTraits<bfloat16> : NumTraits<float> {
  typedef bfloat16 Real;
  typedef bfloat16 NonInteger;
  typedef bfloat16 Nested;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};
template <>
struct NumTraits<float16> : NumTraits<float> {
  typedef float16 Real;
  typedef float16 NonInteger;
  typedef float16 Nested;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<bfloat16, float, BinaryOp> {
  typedef bfloat16 ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<float, bfloat16, BinaryOp> {
  typedef bfloat16 ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<float16, float, BinaryOp> {
  typedef float16 ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<float, float16, BinaryOp> {
  typedef float16 ReturnType;
};
} // namespace Eigen

namespace TensorComparison_pt {
using namespace Eigen;

template <class T>
using VecMap = Map<Matrix<T, 1, Dynamic>>;
template <class T>
using MatMap = Map<Matrix<T, 2, Dynamic>>;

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcL2NormRatio(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);

  float l2NormExpected = vExpected.norm();
  float l2NormResult = vResult.norm();

  if (l2NormExpected == 0 && l2NormResult == 0) {
    return 1;
  }

  if (l2NormExpected > l2NormResult) {
    std::swap(l2NormExpected, l2NormResult);
  }

  return l2NormExpected / l2NormResult;
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcAvgError(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);
  if constexpr (std::is_unsigned<decltype(
                    (vExpected - vResult).array().mean())>::value) {
    return (vExpected.cwiseMax(vResult) - vExpected.cwiseMin(vResult))
        .array()
        .mean();
  } else {
    return abs((vExpected - vResult).array().mean());
  }
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcMaxAbsError(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);

  return (vExpected - vResult).array().abs().maxCoeff();
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcMinAbsError(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);

  return (vExpected - vResult).array().abs().minCoeff();
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcAbsErrorPercentile(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length,
    float percentile) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);

  return (((vExpected - vResult).array() < percentile).count()) / float(length);
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcStdev(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  return sqrt(calcMse(expected, result, length));
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcAvgErrVsStdDev(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  float avgErr = calcAvgError(expected, result, length);
  float stdDev = calcStdev(expected, result, length);

  return (0.5 * stdDev) - avgErr;
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcMse(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);

  return (float)(((vExpected - vResult).array() - (vExpected - vResult).mean())
                     .square()
                     .sum()) /
      (length);
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcCosineSimilarity(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);

  float l2NormExpected = vExpected.norm();
  float l2NormResult = vResult.norm();

  using T = typename std::conditional<
      std::is_floating_point<ReferenceDataType>::value or
          std::is_floating_point<TensorDataType>::value,
      float,
      ReferenceDataType>::type;

  Matrix<T, Dynamic, Dynamic> dot(0, 0);
  dot = (vExpected * vResult.transpose());

  auto val = ((float)dot.value() / (l2NormExpected * l2NormResult));
  auto clipped = std::max(-1.0f, std::min((float)val, 1.0f));
  return acos(clipped) * (180.0 / 3.1415926536);
}

template <typename ReferenceDataType, typename TensorDataType>
float TestMethods<ReferenceDataType, TensorDataType>::calcPearson(
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length) {
  VecMap<ReferenceDataType> vExpected(expected, length);
  VecMap<TensorDataType> vResult(result, length);

  float cov = ((vExpected.array() - vExpected.mean()) *
               (vResult.array() - vResult.mean()))
                  .sum();

  float stdDevResult = sqrt((vResult.array() - vResult.mean()).square().sum());
  float stdDevExpected =
      sqrt((vExpected.array() - vExpected.mean()).square().sum());

  return cov / (stdDevExpected * stdDevResult);
}

template class TestMethods<float16, float16>;
template class TestMethods<bfloat16, bfloat16>;
template class TestMethods<bfloat16, float>;
template class TestMethods<float, bfloat16>;
template class TestMethods<signed char, signed char>;
template class TestMethods<unsigned char, unsigned char>;
template class TestMethods<float, float16>;
template class TestMethods<long, long>;
template class TestMethods<int, int>;
template class TestMethods<float, float>;
template class TestMethods<short, short>;
// template class TestMethods<unsigned short, unsigned short>;
} // namespace TensorComparison_pt
