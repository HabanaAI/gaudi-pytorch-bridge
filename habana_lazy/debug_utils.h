/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_lazy_tensors.h"
#include "tensor_impl.h"
// namespace habana_lazy
namespace habana_lazy {

/**
 * @brief Class to provide utilities for dumping lazy-tensor graphs
 * in text/dot format.
 */
class IrGraphDumpUtil {
 public:
  static std::string ToDot(std::vector<ir::NodePtr> nodes);

  static std::string PostOrderToDot(
      std::vector<ir::NodePtr> post_order,
      std::vector<ir::NodePtr> roots);

  static std::string ToText(std::vector<ir::NodePtr> nodes);

  static std::string PostOrderToText(
      std::vector<ir::NodePtr> post_order,
      std::vector<ir::NodePtr> roots);
};

} // namespace habana_lazy
