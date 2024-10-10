#include <serialization.hpp>
#include <types.hpp>
#include <validator.hpp>
#include <atomic>
#include <functional>
#include <stdexcept>

namespace TensorComparison_pt {

const std::array<std::function<bool(float, float)>, METHODS_MAX>
    ValidatorBase::m_comparisonCriteriaOp = {
        /* ANGLE,              */ std::greater<float>(),
        /* PEARSON,            */ std::less<float>(),
        /* L2_NORM,            */ std::less<float>(),
        /* MSE,                */ std::greater<float>(),
        /* STD_DEV,            */ std::greater<float>(),
        /* ABS_ERR_MAX,        */ std::greater<float>(),
        /* ABS_ERR_MIN,        */ std::greater<float>(),
        /* ABS_ERR_PERCENT2,   */ std::less<float>(),
        /* ABS_ERR_PERCENT3,   */ std::less<float>(),
        /* ABS_ERR_PERCENT4,   */ std::less<float>(),
        /* ABS_ERR_PERCENT5,   */ std::less<float>(),
        /* AVG_ERR,            */ std::greater<float>(),
        /* STD_DEV_VS_AVG_ERR, */ std::greater<float>(),
};

const std::map<enum COMPARISON_METHODS, float>
    StaticValidator::globalThresholds = {
        {ANGLE, m_cosDegMaxErr},
        {PEARSON, m_pearsonMinThreshold},
        {L2_NORM, m_l2NormMinRatio},
        {STD_DEV_VS_AVG_ERR, m_stdDevVsAvgErr},
        {ABS_ERR_MAX, m_epsilonMaxAbsoluteError},
};

ThresholdFileValidator::ThresholdFileValidator(
    const std::string& thresholdsFile)
    : m_thresholdsFile{thresholdsFile} {
  m_thresholds = jsn::parse(thresholdsFile);
}

void ThresholdFileValidator::exportThresholds(const std::string& fileName) {
  std::string exportFile = fileName;
  if (fileName.empty()) {
    exportFile = m_thresholdsFile;
  }

  jsn::dump(m_thresholds, exportFile);
}

bool ThresholdFileValidator::validate(ResultMap&& results) {
  for (auto& it : results) {
    auto t = fetchThreshold(it.first);
    auto r = _validate(it.second, t);
    if (!r) {
      std::cout << "Failed validating tensor: " << it.first << std::endl;
      return false;
    }
  }
  return true;
}

bool ThresholdFileValidator::update(ResultMap&& results) {
  for (auto& result : results) {
    auto it = m_thresholds.find(result.first);
    if (it == m_thresholds.end()) {
      // only update thresholds that are in the current threshold file
      continue;
    }
    _update(result.second, it->second);
  }
  return true;
}

bool ThresholdFileValidator::_update(
    ComparisonResult r,
    std::map<std::string, float>& threshold) {
  for (unsigned i{}; i < METHODS_MAX; ++i) {
    if (r[i].is_set()) {
      auto v = r[i].value();
      auto it = threshold.find(ComparisonMethodsNames[i]);
      if (it == threshold.end()) {
        threshold[ComparisonMethodsNames[i]] = v;
        continue;
      }

      auto op = m_comparisonCriteriaOp[i];
      if (op(v, it->second)) {
        std::cout << ComparisonMethodsNames[i] << " threshold " << it->second
                  << " updated with " << v << std::endl;
        threshold[ComparisonMethodsNames[i]] = v;
      }
    }
  }
  return true;
}

bool ThresholdFileValidator::_validate(
    ComparisonResult r,
    std::map<std::string, float>& threshold) {
  for (unsigned i{}; i < METHODS_MAX; ++i) {
    if (r[i].is_set()) {
      try {
        float& t = threshold.at(ComparisonMethodsNames[i]);
        auto v = r[i].value();
        auto op = m_comparisonCriteriaOp[i];
        if (op(v, t)) {
          std::cout << ComparisonMethodsNames[i] << " threshold " << t
                    << " value " << v << std::endl;
          return false;
        }
      } catch (std::out_of_range) {
        std::cout << "Missing " << ComparisonMethodsNames[i] << " threshold"
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

std::map<std::string, float>& ThresholdFileValidator::fetchThreshold(
    const std::string& name) {
  try {
    return m_thresholds.at(name);
  } catch (std::out_of_range) {
    return m_thresholds.at("DEFAULT");
  }
}

bool StaticValidator::validate(ResultMap&& results) {
  for (auto it : results) {
    auto r = _validate(it.second);
    if (!r) {
      std::cout << "Failed validating tensor: " << it.first << std::endl;
      return false;
    }
  }
  return true;
}

StaticValidator::StaticValidator(const StaticThresholdMap& threshold_map) {
  if (threshold_map.empty()) {
    // fallback to global defaults
    m_thresholds = globalThresholds;
  } else {
    m_thresholds = threshold_map;
  }
}

bool StaticValidator::_validate(ComparisonResult& r) {
  for (unsigned i{}; i < METHODS_MAX; ++i) {
    if (r[i].is_set()) {
      try {
        float& t = m_thresholds.at(static_cast<COMPARISON_METHODS>(i));
        auto v = r[i].value();
        auto op = m_comparisonCriteriaOp[i];
        if (op(v, t)) {
          std::cout << ComparisonMethodsNames[i] << " threshold " << t
                    << " value " << v << std::endl;
          return false;
        }
      } catch (std::out_of_range) {
        std::cout << "Missing " << ComparisonMethodsNames[i] << " threshold"
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}
} // namespace TensorComparison_pt