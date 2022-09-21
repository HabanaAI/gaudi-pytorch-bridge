#pragma once

#include <memory>
#include <string>

#include <comparator.hpp>
#include <results_warehouse.hpp>
#include <types.hpp>
#include <validator.hpp>

namespace TensorComparison_pt {
class TensorValidator {
 public:
  TensorValidator(
      const StaticThresholdMap& threshold_map = {},
      unsigned num_threads = 0);
  TensorValidator(const std::string& thresholds_file, unsigned num_threads = 0);
  template <typename ReferenceDataType, typename TensorDataType>
  bool compare(
      const std::string& name,
      ReferenceDataType* expected,
      TensorDataType* result,
      unsigned length,
      ComparisonMethods compareMethods,
      bool blocking = false);

  bool addComment(const std::string& name, const std::string& comment);

  // bool validate(bool generate = false,  const std::string &out = "");
  void makeReport(const std::string& fileName, ExportType exportType = JSON);

 protected:
  void waitResults();

 private:
  Comparator m_comparator;
  std::unique_ptr<ValidatorBase> m_validator;
  ResultsWarehouse m_resultsWarehouse;
};
} // namespace TensorComparison_pt
