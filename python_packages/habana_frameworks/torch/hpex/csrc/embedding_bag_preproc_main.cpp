#include <iostream>

#include <algorithm>
#include <functional>
#include <vector>
#include "bindings.h"

using namespace std;
typedef int T;

void gaudi_coalescing_preprocessing(
    /* in */ T* indexes,
    /* in */ T* offsets,
    int indexesCount,
    int offsetsCount,
    T maxIndexValue,
    /* out */ T* uniqueIndexes,
    /* out */ T* outputRows,
    /* out */ T* outputRowOffsets,
    /* out */ int* uniqueIdexesCount,
    /* in, modified */ std::pair<T, T>* scratch);

// torch::Tensor
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
embedding_bag_preproc(
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    int64_t embeddingTableLen) {
  auto out = at::empty(indices.sizes(), indices.options());
  torch::Tensor indices_cpu =
      indices.to(at::DeviceType::CPU); // check if really needed
  auto numIndices = static_cast<uint32_t>(indices_cpu.numel());
  torch::Tensor offsets_cpu =
      offsets.to(at::DeviceType::CPU); // check if really needed
  auto numOffsets = static_cast<uint32_t>(offsets_cpu.numel());
  auto p_indices = indices_cpu.data_ptr<int>();
  auto p_offsets = offsets_cpu.data_ptr<int>();

  vector<pair<T, T>> scratch;
  scratch.resize(numIndices * 2);

  auto uniqueIndexes = at::empty(indices_cpu.sizes(), indices_cpu.options());
  auto outputRows = at::empty(indices_cpu.sizes(), indices_cpu.options());
  auto outputRowOffsets =
      at::empty(indices_cpu.size(0) + 1, indices_cpu.options());

  int countUniqueIndexes;
  gaudi_coalescing_preprocessing(
      p_indices,
      p_offsets,
      numIndices,
      numOffsets,
      (int32_t)embeddingTableLen,
      uniqueIndexes.data_ptr<int>(),
      outputRows.data_ptr<int>(),
      outputRowOffsets.data_ptr<int>(),
      &countUniqueIndexes,
      scratch.data());

  auto countUniqueIndexesT = torch::tensor({countUniqueIndexes});
  return std::make_tuple(
      countUniqueIndexesT, uniqueIndexes, outputRows, outputRowOffsets);
}
