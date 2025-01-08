/******************************************************************************
 * Copyright (C) 2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/histogram.h"

using namespace std;
namespace habana {
std::shared_ptr<void> FillHistogramBinCtParams(
    const at::Stack& stack,
    size_t& size) {

    PARAMS_STUB(ns_Histogram::ParamsV2);
    auto bins = stack.at(1).toScalar().toInt();
    auto range = stack.at(2);
    auto weights = stack.at(3);
    auto density = stack.at(4).toScalar().toBool();

    params->bins = bins;
    params->has_weights = weights.isTensor();
    params->density = density;

    if (range.isList()) {
        auto rangeList = range.toListRef();
        params->min = rangeList[0].toDouble();
        params->max = rangeList[1].toDouble();
    }
    return params;
}

OutputMetaDataVector HistogramBinCtMeta(const at::Stack& stack) {
    const auto dtype = stack.at(0).toTensor().scalar_type();
    auto bins = stack.at(1);
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

SharedMetaDataVector HistogramCommonSharedMeta(
    const at::Stack& stack,
    int ranges_offset) {
    auto self = stack.at(0).toTensor();
    auto has_ranges = stack.at(ranges_offset).isTensor();
    auto has_weights = stack.at(ranges_offset + 1).isTensor();

    SharedMetaData histogramMeta{"histogram"};
    histogramMeta.inputs_data.emplace_back(self.dim(), self.scalar_type());
    if (has_ranges)
        histogramMeta.inputs_data.emplace_back(1, self.scalar_type());

    if (has_weights)
        histogramMeta.inputs_data.emplace_back(1, c10::ScalarType::Float);

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