#pragma once

#include <algorithm>
#include <future>
#include <iostream>
#include <mutex>

#include <serialization.hpp>
#include <types.hpp>

namespace TensorComparison_pt {

using FutureResult = std::future<ComparisonResult>;

class ResultsWarehouse {
 public:
  void add(const std::string& n, FutureResult&& r) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_futureResults.find(n) != m_futureResults.end()) {
      throw "would have overwritten tensor!";
    }

    m_futureResults.emplace(n, std::move(r));
  }

  void addComment(const std::string& n, const std::string& comment) {
    waitResults();
    std::string key = n;
    auto itRes = std::find_if(
        m_results.begin(),
        m_results.end(),
        [&key](const std::pair<std::string, ComparisonResult>& element) {
          return element.first == key;
        });
    // result already exists
    if (itRes != m_results.end()) {
      itRes->second.SetComment(comment);
    } else {
      ComparisonResult res;
      res.SetComment(comment);
      ResultElem elem = {n, res};
      m_results.emplace_back(elem);
    }
  }

  ResultVec&& getResults() {
    waitResults();

    return std::move(m_results);
  }

  void waitResults() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& it : m_futureResults) {
      std::string key = it.first;
      auto itRes = std::find_if(
          m_results.begin(),
          m_results.end(),
          [&key](const std::pair<std::string, ComparisonResult>& element) {
            return element.first == key;
          });
      if (itRes != m_results.end()) {
        // throw "would have overwritten tensor!";
        itRes->second = it.second.get();
      } else {
        ResultElem elem = {it.first, it.second.get()};
        m_results.emplace_back(elem);
      }
    }
    m_futureResults.clear();
  }

  void exportResults(const std::string& fileName, ExportType exportType) {
    waitResults();

    if (exportType == JSON) {
      jsn::dump(m_results, fileName);
    } else // CSV
    {
      csv::dump(m_results, fileName);
    }
  }

 private:
  std::string m_reportFile;
  ;

  std::map<std::string, FutureResult> m_futureResults;
  ResultVec m_results;
  std::mutex m_mutex;
};
} // namespace TensorComparison_pt