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

#include "generated/lazy/_adaptive_avg_pool2d.h"
#include "generated/lazy/_ctc_loss.h"
#include "generated/lazy/_upsample_bilinear2d_aa.h"
#include "generated/lazy/abs.h"
#include "generated/lazy/bmm.h"
#include "generated/lazy/channel_shuffle.h"
#include "generated/lazy/clamp.h"
#include "generated/lazy/ctc_loss_custom.h"
#include "generated/lazy/ctc_loss_custom_backward.h"
#include "generated/lazy/fused_clip_norm.h"
#include "generated/lazy/grid_sampler_2d.h"
#include "generated/lazy/grid_sampler_3d.h"
#include "generated/lazy/im2col.h"
#include "generated/lazy/in_place_interleave.h"
#include "generated/lazy/index_reduce.h"
#include "generated/lazy/kv_reorder.h"
#include "generated/lazy/linear.h"
#include "generated/lazy/masked_fill.h"
#include "generated/lazy/masked_scatter.h"
#include "generated/lazy/max_pool2d_with_indices.h"
#include "generated/lazy/max_pool3d_with_indices.h"
#include "generated/lazy/max_unpool2d.h"
#include "generated/lazy/max_unpool3d.h"
#include "generated/lazy/mm.h"
#include "generated/lazy/multi_margin_loss.h"
#include "generated/lazy/multilabel_margin_loss_forward.h"
#include "generated/lazy/nll_loss2d_forward.h"
#include "generated/lazy/nll_loss_forward.h"
#include "generated/lazy/optimizer_resource_apply_momentum.h"
#include "generated/lazy/reflection_pad1d.h"
#include "generated/lazy/reflection_pad2d.h"
#include "generated/lazy/reflection_pad3d.h"
#include "generated/lazy/replication_pad1d.h"
#include "generated/lazy/replication_pad2d.h"
#include "generated/lazy/replication_pad3d.h"
#include "generated/lazy/scaled_triangular_softmax_retain.h"
#include "generated/lazy/scatter.h"
#include "generated/lazy/scatter_add.h"
#include "generated/lazy/searchsorted.h"
#include "generated/lazy/upsample_bicubic2d.h"
#include "generated/lazy/upsample_bilinear2d.h"
#include "generated/lazy/where.h"
