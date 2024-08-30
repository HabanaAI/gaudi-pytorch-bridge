###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

from habana_frameworks.torch.utils.internal import is_lazy

if not is_lazy():
    import warnings

    warnings.warn(
        f"CustomNms, RoiAlignFunction, ScaledMaskedSoftmax, fp8_fused_sdpa from {__name__} are no yet supported in eager mode"
    )

from .CTCLoss import CTCLoss
from .CustomNms import CustomNms
from .CustomRoiAlign import RoiAlignFunction
from .CustomSoftmax import CustomSoftmax
from .Fp8FusedSDPA import fp8_fused_sdpa, fp8_sdpa_bwd_wrapper, fp8_sdpa_fwd_wrapper, gqa_output_reshape
from .FusedSDPA import FusedSDPA
from .PySDPA import PySDPA, PySDPAHinted
from .RotaryPosEmbeddingHelper import (
    RotaryPosEmbeddingHelperV1,
    RotaryPosEmbeddingHelperV2,
    RotaryPosEmbeddingHelperV3,
    RotaryPosEmbeddingMode,
    apply_rotary_pos_emb,
)
from .ScaledMaskedSoftmax import ScaledMaskedSoftmax
