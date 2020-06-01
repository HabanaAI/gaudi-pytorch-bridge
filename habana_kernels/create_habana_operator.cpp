/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/habana_operator.h"
#include "habana_kernels/unary_kernels.h"

namespace habana {
using HabanaOperatorPtr = std::shared_ptr<HabanaOperator>;
 
HabanaOperatorPtr CreateHabanaOperator(const int device_id,
                                       const std::string& node_name,
                                       c10::ScalarType node_type) {
    HabanaOperatorPtr op = nullptr;

    if ("aten::relu" == node_name) {
        op = std::make_shared<ReluOperator>(device_id, node_type);
    }
    else if ("aten::sigmoid" == node_name) {
        op = std::make_shared<SigmoidOperator>(device_id, node_type);
    }

    // Returning a null pointer for cases not added yet,
    // we can add assert once all kernels are added
    return op;
}
}
