/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/histc.h"
#include "generated/backend/histogram.h"

namespace habana {

std::shared_ptr<void> FillHistcParams(
    const at::Stack& stack,
    size_t& size) {

    PARAMS_STUB(ns_Histogram::ParamsV2);
    const auto bins = stack.at(1).toInt();
    const auto min = stack.at(2).toInt();
    const auto max = stack.at(3).toInt();

    params->bins = bins;
    params->density = false;
    params->has_weights = false;
    params->min = min;
    params->max = max;

    return params;
}

OutputMetaDataVector HistcMeta(const at::Stack& stack) {
    const auto dtype = stack.at(0).toTensor().scalar_type();
    const auto bins = stack.at(1).toInt();

    OutputMetaData meta{dtype, {bins}};
    return {meta};
}


void Histc::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
    size_t size = 0;
    const auto params = FillHistcParams(stack, size);
    const auto meta = HistogramBinCtMeta(stack);

    auto histogram = BuildOp(
        graph,
        guid_,
        {syn_in(0)},
        {{meta[0].shape, meta[0].dtype, 0},
        {meta[1].shape, meta[1].dtype, std::nullopt}},
        params.get(),
        size
    );
    syn_out(0) = std::move(histogram[0]);
}

std::shared_ptr<void> FillHistogramBinCtParams(
    const at::Stack& stack,
    size_t& size) {
    PARAMS_STUB(ns_Histogram::ParamsV2);
    const auto bins = stack.at(1).toScalar().toInt();
    const auto range = stack.at(2);
    const auto weights = stack.at(3);
    const auto density = stack.at(4).toScalar().toBool();

    params->bins = bins;
    params->has_weights = weights.isTensor();
    params->density = density;

    if (range.isList()) {
        const auto rangeList = range.toListRef();
        params->min = rangeList[0].toDouble();
        params->max = rangeList[1].toDouble();
    }

    return params;
}

OutputMetaDataVector HistogramBinCtMeta(const at::Stack& stack) {
    const auto dtype = stack.at(0).toTensor().scalar_type();
    const auto bins = stack.at(1);
    int64_t num_bins = 0;
    if (bins.isScalar()) {
        num_bins = bins.toScalar().toInt();
    } else {
        num_bins = bins.toTensor().sizes()[0];
    }

    OutputMetaData meta1{dtype, {num_bins}};
    OutputMetaData meta2{dtype, {num_bins + 1}};
    return {meta1, meta2};
}

std::shared_ptr<void> FillHistogramBinsParams(
    const at::Stack& stack,
    size_t& size) {
    PARAMS_STUB(ns_Histogram::ParamsV2);

    params->bins = stack.at(1).toTensor().sizes()[0] - 1;
    params->has_weights = stack.at(2).isTensor();
    params->density = stack.at(3).toScalar().toBool();

    return params;
}

OutputMetaDataVector HistogramBinsMeta(const at::Stack& stack) {
    const auto dtype = stack.at(0).toTensor().scalar_type();
    const auto num_bins = stack.at(1).toTensor().sizes()[0] - 1;

    TORCH_CHECK(num_bins == 1, "Histogram with bins tensor input is not fully supported on HPU, as there is no vectorization possible for number of bins > 1.");

    OutputMetaData meta1{dtype, {num_bins}};
    OutputMetaData meta2{dtype, {num_bins + 1}};
    return {meta1, meta2};
}

SharedMetaDataVector HistcSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
    const auto self = stack.at(0).toTensor();
    SharedMetaData histcMeta{"histogram"};
    histcMeta.inputs_data.emplace_back(self.dim(), self.scalar_type());
    histcMeta.outputs_data.emplace_back(1, self.scalar_type());
    histcMeta.outputs_data.emplace_back(1, self.scalar_type());
    return {histcMeta};
}

SharedMetaDataVector HistogramCommonSharedMeta(
    const at::Stack& stack,
    int ranges_offset) {
    const auto self = stack.at(0).toTensor();
    const auto has_ranges = stack.at(ranges_offset).isTensor();
    const auto has_weights = stack.at(ranges_offset + 1).isTensor();

    SharedMetaData histogramMeta{"histogram"};
    histogramMeta.inputs_data.emplace_back(self.dim(), self.scalar_type());
    if (has_ranges)
        histogramMeta.inputs_data.emplace_back(1, self.scalar_type());
    else
        histogramMeta.inputs_data.push_back(createOptionalNotPresentSharedMetaTensor());

    if (has_weights)
        histogramMeta.inputs_data.emplace_back(1, self.scalar_type());

    histogramMeta.outputs_data.emplace_back(1, self.scalar_type());
    histogramMeta.outputs_data.emplace_back(1, self.scalar_type());
    return {histogramMeta};
}

SharedMetaDataVector HistogramBinsSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
    return HistogramCommonSharedMeta(stack, 1);
}

SharedMetaDataVector HistogramBinCtSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
    return HistogramCommonSharedMeta(stack, 2);
}

} // namespace habana
