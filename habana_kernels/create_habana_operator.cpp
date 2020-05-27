#include "habana_kernels/habana_operator.h"
#include "habana_kernels/unary_kernels.h"

namespace habana {
using HabanaOperatorPtr = std::shared_ptr<HabanaOperator>;
 
HabanaOperatorPtr CreateHabanaOperator(const int device_id,
                                       const std::string& node_name,
                                       c10::ScalarType node_type) {

    if ("aten::relu" == node_name) {
        return std::make_shared<ReluOperator>(device_id, node_type);
    }
    else if ("aten::sigmoid" == node_name) {
        return std::make_shared<SigmoidOperator>(device_id, node_type);
    }
    else {
        //Returning a null pointer for now, we can add assert once all kernels are added
        return nullptr;
    }
}
}
