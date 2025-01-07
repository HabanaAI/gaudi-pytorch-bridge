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

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include <stdio.h>

#include <unistd.h>
//#include <stdlib.h>
#include <cstdlib>

#include "radix_sort.h"
using namespace std;

typedef int T;

#define COALESCING_PREPROCESSING_VERBOSE 0

class CoalescingPreprocessingThreadState {
 public:
  CoalescingPreprocessingThreadState() : numUniqueIndexes(0), startOffset(0) {}
  uint32_t numUniqueIndexes;
  uint32_t startOffset;
};

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
    /* in, modified */ std::pair<T, T>* scratch) {
  // cout << "NumIndexes= " << indexesCount << endl;
  omp_set_num_threads(1);
#pragma omp parallel for schedule(static)
  for (int sampleItr = 0; sampleItr < (offsetsCount - 1); sampleItr++) {
    int start = offsets[sampleItr];
    int end = offsets[sampleItr + 1];
    for (int indexesInSampleItr = start; indexesInSampleItr < end;
         indexesInSampleItr++) {
      scratch[indexesInSampleItr].first = indexes[indexesInSampleItr];
      scratch[indexesInSampleItr].second = sampleItr;
    }
  }

#if COALESCING_PREPROCESSING_VERBOSE
  for (int itr = 0; itr < indexesCount; itr++) {
    cout << "Idx: " << scratch[itr].first << " Sample: " << scratch[itr].second
         << endl;
  }
#endif

  auto sortedIndexWithOutputRowPair = radix_sort_parallel(
      scratch, scratch + indexesCount, indexesCount, maxIndexValue);
  int maxThreads = omp_get_max_threads();

#if COALESCING_PREPROCESSING_VERBOSE
  cout << "Sorted: " << endl;
  for (int itr = 0; itr < indexesCount; itr++) {
    cout << "Idx: " << sortedIndexWithOutputRowPair[itr].first
         << " Sample: " << sortedIndexWithOutputRowPair[itr].second << endl;
  }
#endif

  CoalescingPreprocessingThreadState preprocessingThreadState[maxThreads];

  uint32_t indexesPerThread = indexesCount / maxThreads;
  uint32_t indexesRemainederForLastThread =
      indexesCount - indexesPerThread * maxThreads;
#pragma omp parallel
  {
    // First pass - determine where each thread starts and how many unique
    // indexes it owns
    int tid = omp_get_thread_num();
    int startIdx = indexesPerThread * tid;
    int endIdx = startIdx + indexesPerThread;
    if (tid == (maxThreads - 1)) {
      endIdx += indexesRemainederForLastThread;
    }

    T prevVal =
        (tid == 0) ? (T)(-1) : sortedIndexWithOutputRowPair[startIdx - 1].first;
    while (startIdx < endIdx &&
           sortedIndexWithOutputRowPair[startIdx].first == prevVal) {
      startIdx++;
    }
    preprocessingThreadState[tid].startOffset = startIdx;

    prevVal = (T)(-1);
    for (int itr = startIdx; itr < endIdx; itr++) {
      if (sortedIndexWithOutputRowPair[itr].first != prevVal) {
        preprocessingThreadState[tid].numUniqueIndexes++;
        prevVal = sortedIndexWithOutputRowPair[itr].first;
      }
    }

#pragma omp barrier
    // Second pass - for each unique index, write its offset
    // Initially, determine where my first unique index should be written
    uint32_t nextUniqueIdxToWriteAtOffset = 0;
    for (int itr = 0; itr < tid; itr++) {
      nextUniqueIdxToWriteAtOffset +=
          preprocessingThreadState[itr].numUniqueIndexes;
    }
    endIdx = (tid == (maxThreads - 1))
        ? endIdx
        : preprocessingThreadState[tid + 1].startOffset;
#if COALESCING_PREPROCESSING_VERBOSE
#pragma omp critical
    cout << tid << " " << startIdx << " " << endIdx << " "
         << preprocessingThreadState[tid].numUniqueIndexes << " "
         << nextUniqueIdxToWriteAtOffset << endl;
#endif

    prevVal =
        (startIdx < endIdx) ? sortedIndexWithOutputRowPair[startIdx].first : 0;
    T prevOffset = startIdx;
    uint32_t idxCount = 0;
    // Each unique index should be written to
    for (int itr = startIdx; itr < endIdx; itr++) {
      outputRows[itr] = sortedIndexWithOutputRowPair[itr].second;
      if (sortedIndexWithOutputRowPair[itr].first == prevVal) {
        idxCount++;
      } else {
        uniqueIndexes[nextUniqueIdxToWriteAtOffset] = prevVal;
        outputRowOffsets[nextUniqueIdxToWriteAtOffset] = prevOffset;
        prevOffset += idxCount;
        nextUniqueIdxToWriteAtOffset++;
        prevVal = sortedIndexWithOutputRowPair[itr].first;
        idxCount = 1;
      }
    }
    if (idxCount > 0) {
      uniqueIndexes[nextUniqueIdxToWriteAtOffset] = prevVal;
      outputRowOffsets[nextUniqueIdxToWriteAtOffset] = prevOffset;
      prevOffset += idxCount;
      nextUniqueIdxToWriteAtOffset++;
    }
    if (tid == (maxThreads - 1)) {
      outputRowOffsets[nextUniqueIdxToWriteAtOffset] = prevOffset;
      *uniqueIdexesCount = nextUniqueIdxToWriteAtOffset;
    }
  }
#if COALESCING_PREPROCESSING_VERBOSE
  cout << "Postporcessing done! uniqueIdexesCount=" << *uniqueIdexesCount
       << endl;
  for (int itr = 0; itr < indexesCount; itr++) {
    cout << "Grad_out_row: " << outputRows[itr] << endl;
  }

  for (int itr = 0; itr < *uniqueIdexesCount; itr++) {
    cout << "UniqueIdx = " << uniqueIndexes[itr]
         << " Offset = " << outputRowOffsets[itr] << endl;
  }
  cout << "Last offset: " << outputRowOffsets[*uniqueIdexesCount] << endl;

#endif
#if COALESCING_PREPROCESSING_VERBOSE
  cout << indexesCount << endl;
#endif
}