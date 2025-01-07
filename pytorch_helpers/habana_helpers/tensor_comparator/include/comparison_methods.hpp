#pragma once

namespace TensorComparison_pt {
template <typename ReferenceDataType, typename TensorDataType>
struct TestMethods {
  static float calcL2NormRatio(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcPearson(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcCosineSimilarity(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcMse(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcMaxAbsError(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcAvgError(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcStdev(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcMinAbsError(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);

  static float calcAbsErrorPercentile(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length,
      float percentile);

  static float calcAvgErrVsStdDev(
      ReferenceDataType expected[],
      TensorDataType result[],
      unsigned length);
};
} // namespace TensorComparison_pt
