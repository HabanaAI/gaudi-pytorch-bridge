/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "view_utils.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/lazy_executor.h"

using namespace habana;
using namespace at;

namespace habana_lazy {
/**
 * Return true is self and other tensors are views. For storage tensors, they
 * would share the storage. For HPU storageless lazy tensors, the steps to check
 * views are -
 *   - Case 1: At least one of self and other need to be in context->view_table
 *   - Case 2: If both self and other are views, found in context->view_table
 *      - The base tensor for both entries must be the same, which covers the
 * case self = view(base) other = view(base) is_alias_of(self, other) -> returns
 * true
 *   - Case 3: If only one of self/other is view, found in context->view_table
 *      - The base tensor from view table must be same as the non-view tensor
 *        this covers the case -
 *           self = tensor(...)
 *           other = view(self)
 *           is_alias_of(self, other) -> returns true
 *        or, when self is a view of other above.
 */

bool is_aliased_view(
    HbLazyTensorImpl& self_impl,
    HbLazyTensorImpl& other_impl) {
  auto aliased = false;
  const auto& self_lazy = self_impl.tensor();
  const auto& other_lazy = other_impl.tensor();

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto self_id = self_lazy.getTensorUniqueId();
  auto other_id = other_lazy.getTensorUniqueId();

  auto find_view_base{
      [&context](int64_t tensor_id) -> std::tuple<int64_t, bool> {
        int64_t base_id = -1;
        bool is_view = false;
        auto it = context->view_table.find(tensor_id);
        if (it != context->view_table.end()) {
          const StrideParams& params = context->view_table[tensor_id];
          auto updated_tensor = get_recent_base_tensor(params.base);
          base_id = GetHbLazyTensor(updated_tensor).getTensorUniqueId();
          is_view = true;
        } else {
          auto it = context->orig_tensor_map.find(tensor_id);
          if (it != context->orig_tensor_map.end()) {
            base_id = GetHbLazyTensor((it->second)).getTensorUniqueId();
            is_view = true;
          }
        }
        return std::make_tuple(base_id, is_view);
      }};

  int64_t self_base_id = -1, other_base_id = -1;
  bool self_is_view = false, other_is_view = false;

  std::tie(self_base_id, self_is_view) = find_view_base(self_id);
  std::tie(other_base_id, other_is_view) = find_view_base(other_id);

  // Case 1: Both of them are not views, alias if self_id same as other_id
  if (!self_is_view && !other_is_view) {
    aliased = self_id == other_id;
  } else if (self_is_view && other_is_view) {
    // Case 2: If both self and other are views
    aliased = self_base_id == other_base_id;
  } else if (self_is_view) {
    // Case 3: If only one of self/other is view
    aliased = self_base_id == other_id;
  } else {
    // Case 3: If only one of self/other is view
    HABANA_ASSERT(other_is_view);
    aliased = other_base_id == self_id;
  }

  PT_VIEWTABLE_DEBUG(
      "is_aliased view ",
      aliased,
      " self_is_view ",
      self_is_view,
      " other_is_view ",
      other_is_view,
      " self_base_id ",
      self_base_id,
      " other_base_id ",
      other_base_id);
  return aliased;
}

} // namespace habana_lazy
