#pragma once
#include <types.hpp>
#include <functional>
#include <iostream>

namespace TensorComparison_pt {
class ValidatorBase {
 public:
  virtual bool validate(ResultMap&& results) = 0;
  virtual bool update(ResultMap&& results) = 0;
  virtual void exportThresholds(const std::string&) = 0;

  virtual ~ValidatorBase() = default;

 protected:
  static const std::array<std::function<bool(float, float)>, METHODS_MAX>
      m_comparisonCriteriaOp;
};

class StaticValidator : public ValidatorBase {
 public:
  StaticValidator(const StaticThresholdMap& threshold_map);
  bool validate(ResultMap&& results) override;
  bool update(ResultMap&& /*results*/) override {
    return true;
  }
  void exportThresholds(const std::string&) override {}

 private:
  bool _validate(ComparisonResult& result);

  StaticThresholdMap m_thresholds;

  // global synapse thresholds
  static constexpr float m_pearsonMinThreshold = 0.95;
  static constexpr float m_epsilonMaxAbsoluteError = 1e-3;
  static constexpr float m_l2NormMinRatio = 1.0 - 0.2;
  static constexpr float m_monotonicRange = 1e-3;
  static constexpr float m_cosDegMaxErr = 1e-3;
  static constexpr float m_stdDevVsAvgErr = 0;

  static const std::map<enum COMPARISON_METHODS, float> globalThresholds;
};

class ThresholdFileValidator : public ValidatorBase {
 public:
  ThresholdFileValidator(const std::string& thresholdsFile);

  bool validate(ResultMap&& results) override;
  bool update(ResultMap&& results) override;
  void exportThresholds(const std::string& fileName) override;

 protected:
  static bool _validate(
      ComparisonResult r,
      std::map<std::string, float>& threshold);
  static bool _update(
      ComparisonResult r,
      std::map<std::string, float>& threshold);

  std::map<std::string, float>& fetchThreshold(const std::string& name);

  const std::string m_thresholdsFile;
  ThresholdFileMap m_thresholds;
};
} // namespace TensorComparison_pt