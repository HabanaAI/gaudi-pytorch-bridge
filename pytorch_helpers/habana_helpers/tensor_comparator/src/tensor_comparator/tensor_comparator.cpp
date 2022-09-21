#include <tensor_comparator.hpp>
#include <future>
#include <memory>
#include <string>
#include <utility>

#include <bfloat16.h>
#include <comparator.hpp>
#include <float16.h>
#include <results_warehouse.hpp>
#include <types.hpp>
#include <validator.hpp>

namespace TensorComparison_pt {
TensorValidator::TensorValidator(
    const StaticThresholdMap& thresholdMap,
    unsigned num_threads)
    : m_comparator{num_threads},
      m_resultsWarehouse{},
      m_validator{std::make_unique<StaticValidator>(thresholdMap)} {}

TensorValidator::TensorValidator(
    const std::string& thresholdsFile,
    unsigned num_threads)
    : m_comparator{num_threads},
      m_resultsWarehouse{},
      m_validator{std::make_unique<ThresholdFileValidator>(thresholdsFile)} {}

template <typename ReferenceDataType, typename TensorDataType>
bool TensorValidator::compare(
    const std::string& name,
    ReferenceDataType expected[],
    TensorDataType result[],
    unsigned length,
    ComparisonMethods compareMethods,
    bool blocking) {
  m_resultsWarehouse.add(
      name, m_comparator.compare(expected, result, length, compareMethods));

  if (blocking) {
    waitResults();
  }
  return true;
}

bool TensorValidator::addComment(
    const std::string& name,
    const std::string& comment) {
  m_resultsWarehouse.addComment(name, comment);
  return true;
}

void TensorValidator::waitResults() {
  m_resultsWarehouse.waitResults();
}

void TensorValidator::makeReport(
    const std::string& file_name,
    ExportType exportType) {
  waitResults();
  m_resultsWarehouse.exportResults(file_name, exportType);
}

// bool TensorValidator::validate(bool generateThresholdCriteria, const
// std::string &out)
// {
//   if (generateThresholdCriteria)
//   {
//     m_validator->update(m_resultsWarehouse.getResults());
//     m_validator->exportThresholds(out);
//     return true;
//   }

//   return m_validator->validate(m_resultsWarehouse.getResults());
// }

// instantiate compare methods
template bool TensorValidator::compare(
    const std::string&,
    float*,
    float*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    float16*,
    float16*,
    unsigned,
    ComparisonMethods,
    bool);
template bool TensorValidator::compare(
    const std::string&,
    bfloat16*,
    bfloat16*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    float*,
    bfloat16*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    bfloat16*,
    float*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    long*,
    long*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    int*,
    int*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    signed char*,
    signed char*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    unsigned char*,
    unsigned char*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    float*,
    float16*,
    unsigned,
    ComparisonMethods,
    bool);

template bool TensorValidator::compare(
    const std::string&,
    short*,
    short*,
    unsigned,
    ComparisonMethods,
    bool);

// template bool TensorValidator::compare
//                                                         (const std::string&,
//                                                         unsigned short*,
//                                                         unsigned short*,
//                                                         unsigned,
//                                                         ComparisonMethods,
//                                                         bool);

} // namespace TensorComparison_pt