#pragma once

#include <array>
#include <bitset>
#include <cassert>
#include <map>
#include <string>
#include <vector>

namespace TensorComparison_pt {
namespace Names __attribute__((visibility("internal"))) {
  static const std::string ANGLE = "angle";
  static const std::string PEARSON = "pearson";
  static const std::string L2_NORM = "l2_norm";
  static const std::string MSE = "mse";
  static const std::string STD_DEV = "std_dev";
  static const std::string ABS_ERR_MAX = "max_abs_err";
  static const std::string ABS_ERR_MIN = "min_abs_err";
  static const std::string ABS_ERR_PERCENT2 = "abs_err_percent2";
  static const std::string ABS_ERR_PERCENT3 = "abs_err_percent3";
  static const std::string ABS_ERR_PERCENT4 = "abs_err_percent4";
  static const std::string ABS_ERR_PERCENT5 = "abs_err_percent5";
  static const std::string AVG_ERR = "avg_err";
  static const std::string STD_DEV_VS_AVG_ERR = "std_dev_vs_avg_err";
} // namespace )

enum COMPARISON_METHODS {
  ANGLE,
  PEARSON,
  L2_NORM,
  MSE,
  STD_DEV,
  ABS_ERR_MAX,
  ABS_ERR_MIN,
  ABS_ERR_PERCENT2,
  ABS_ERR_PERCENT3,
  ABS_ERR_PERCENT4,
  ABS_ERR_PERCENT5,
  AVG_ERR,
  STD_DEV_VS_AVG_ERR,

  METHODS_MAX // last
};

static const std::array<const std::string, METHODS_MAX> ComparisonMethodsNames =
    {
        Names::ANGLE,
        Names::PEARSON,
        Names::L2_NORM,
        Names::MSE,
        Names::STD_DEV,
        Names::ABS_ERR_MAX,
        Names::ABS_ERR_MIN,
        Names::ABS_ERR_PERCENT2,
        Names::ABS_ERR_PERCENT3,
        Names::ABS_ERR_PERCENT4,
        Names::ABS_ERR_PERCENT5,
        Names::AVG_ERR,
        Names::STD_DEV_VS_AVG_ERR,
};

enum ExportType { JSON, CSV };

class Optional {
  using T = float;

 public:
  bool is_set() const {
    return m_isSet;
  }
  T value() const {
    assert(m_isSet);
    return m_val;
  }
  void set(T v) {
    m_val = v;
    m_isSet = true;
  }
  Optional& operator=(const T& o) {
    set(o);
    return *this;
  }

 private:
  bool m_isSet = false;
  T m_val;
};

class ComparisonResult {
  using ComparisonResultList = std::array<Optional, METHODS_MAX>;

 public:
  ComparisonResult() = default;

  void SetComment(const std::string& comment) {
    m_comment = comment;
  }
  const std::string& GetComment() const {
    return m_comment;
  }

  ComparisonResultList::reference operator[](size_t index) {
    return m_result[index];
  }
  ComparisonResultList::const_reference operator[](size_t index) const {
    return m_result[index];
  }
  ComparisonResultList::pointer data() noexcept {
    return m_result.data();
  }
  ComparisonResultList::const_pointer data() const noexcept {
    return m_result.data();
  }
  ComparisonResultList::iterator begin() noexcept {
    return m_result.begin();
  }
  ComparisonResultList::iterator end() noexcept {
    return m_result.end();
  }
  ComparisonResultList::const_iterator begin() const noexcept {
    return m_result.begin();
  }
  ComparisonResultList::const_iterator end() const noexcept {
    return m_result.end();
  }
  ComparisonResultList::const_iterator cbegin() const noexcept {
    return m_result.cbegin();
  }
  ComparisonResultList::const_iterator cend() const noexcept {
    return m_result.cend();
  }

 private:
  ComparisonResultList m_result;
  std::string m_comment;
};

using ComparisonMethods = std::bitset<METHODS_MAX>;
using ResultKey = std::pair<int, std::string>;
using ResultMap = std::map<std::string, ComparisonResult>;
using ResultElem = std::pair<std::string, ComparisonResult>;
using ResultVec = std::vector<ResultElem>;
using ThresholdFileMap = std::map<std::string, std::map<std::string, float>>;
using StaticThresholdMap = std::map<enum COMPARISON_METHODS, float>;
} // namespace TensorComparison_pt