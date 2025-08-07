
.. _pytorch-custom-operators:

****************************
PyTorch Custom Operators
****************************

Overview
========

This document summarizes the SynapseAI® Software PyTorch supported custom operators for
Habana® Gaudi®. Note that the operators listed below support only selected
variants and limited optional parameters for Gaudi.

The ops in the lists below are available under the torch.ops.hpu namespace.
The supported dtypes indicate the input dtypes and any given operator may support output dtypes not listed here.
For example: cast_from_fp8 supports FP8 inputs but FP32 and BF16 outputs, but the output types are not mentioned here.


Custom Fused Optimizers Support Summary
=======================================

.. rst-class:: datatable

====================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
**Custom Optimizers**                 **FP32** **BF16** **FP16** **INT64** **INT32** **INT8** **BOOL**  **FP8**  **FP4**  **INT4**
====================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
optimizer_adamw                          Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_ema                            Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_lamb_fused_norm                Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_lamb_phase1                    Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_lamb_phase2                    Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_lars                           Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_resource_apply_momentum        Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_sgd                            Yes      Yes      Yes      No        No        No      No        No        No       No
optimizer_sgd_momentum                   Yes      Yes      Yes      No        No        No      No        No        No       No
====================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========

All of the custom fused optimizers are exposed under their own wrapper functions.
For more details on their usage see :ref:`custom_operators`.
Custom optimizers are only supported in Lazy and Eager modes of execution.

Custom Operators Support Summary
=================================

.. rst-class:: datatable

====================================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
**Custom Operator**                                   **FP32** **BF16** **FP16** **INT64** **INT32** **INT8** **BOOL**  **FP8**  **FP4**  **INT4**
====================================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
_fp8_gemm_bwd                                           No       No       No       No        No        No       No       Yes       No       No
_fp8_gemm_bwd.scalar                                    No       No       No       No        No        No       No       Yes       No       No
_fp8_gemm_bwd.scalar_list                               No       No       No       No        No        No       No       Yes       No       No
accumulate_grads\_                                      Yes      Yes      Yes      Yes       Yes       Yes      Yes      Yes       No       No
block_softmax                                           Yes      Yes      Yes      No        No        No       No       No        No       No
block_softmax_adjustment                                Yes      Yes      Yes      No        No        No       No       No        No       No
calculate_scale_for_cast                                Yes      Yes      Yes      No        No        No       No       No        No       No
cast_from_fp8                                           No       No       No       No        No        No       No       Yes       No       No
cast_from_fp8.scalar                                    No       No       No       No        No        No       No       Yes       No       No
cast_from_fp8.scalar_list                               No       No       No       No        No        No       No       Yes       No       No
cast_to_fp8                                             Yes      Yes      No       No        No        No       No       No        No       No
cast_to_fp8_hybrid                                      Yes      Yes      No       No        No        No       No       No        No       No
cast_to_fp8_just_in_time                                Yes      Yes      Yes      No        No        No       No       No        No       No
cast_to_fp8_v2                                          Yes      Yes      No       No        No        No       No       No        No       No
cast_to_fp8_v2.scalar                                   Yes      Yes      No       No        No        No       No       No        No       No
cast_to_fp8_v2.scalar_list                              Yes      Yes      No       No        No        No       No       No        No       No
conv2d_fp8                                              No       No       No       No        No        No       No       Yes       No       No
conv2d_fp8.scalar                                       No       No       No       No        No        No       No       Yes       No       No
convert_from_int4                                       No       No       No       No        No        No       No       No        No       Yes
convert_from_uint4                                      No       No       No       No        No        No       No       No        No       Yes
ctc_loss_custom                                         Yes      No       No       No        No        No       No       No        No       No
ctc_loss_custom_backward                                Yes      No       No       No        No        No       No       No        No       No
custom_softmax                                          No       Yes      No       No        No        No       No       No        No       No
exp_fast_math                                           No       Yes      No       No        No        No       No       No        No       No
fake_quant_fp4                                          No       No       No       No        No        No       No       No        Yes      No
fp8_gemm                                                No       No       No       No        No        No       No       Yes       No       No
fp8_gemm_v2                                             No       No       No       No        No        No       No       Yes       No       No
fp8_gemm_v2.scalar                                      No       No       No       No        No        No       No       Yes       No       No
fp8_gemm_v2.scalar_list                                 No       No       No       No        No        No       No       Yes       No       No
fp8_sdpa_bwd                                            No       No       No       No        No        No       No       Yes       No       No
fp8_sdpa_fwd                                            No       No       No       No        No        No       No       Yes       No       No
fp8_sdpa_fwd_dropout                                    No       No       No       No        No        No       No       Yes       No       No
fp8_sdpa_fwd_non_dropout                                No       No       No       No        No        No       No       Yes       No       No
fp8_sdpa_recomp_fwd                                     No       No       No       No        No        No       No       Yes       No       No
fp8_sdpa_recomp_fwd_dropout                             No       No       No       No        No        No       No       Yes       No       No
fp8_sdpa_recomp_fwd_non_dropout                         No       No       No       No        No        No       No       Yes       No       No
fused_clip_norm                                         Yes      Yes      Yes      No        No        No       No       No        No       No
in_place_interleave                                     No       Yes      No       No        No        No       No       Yes       No       No
kv_reorder                                              Yes      Yes      Yes      Yes       Yes       Yes      Yes      Yes       No       No
masked_batch_gemm                                       Yes      Yes      No       No        No        No       No       Yes       No       No
mixture_of_experts                                      Yes      Yes      No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8                                  No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_blockwise                        No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_dynamic                          No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_fused_weights                    No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_fused_weights_blockwise          No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_fused_weights_dynamic            No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_fused_weights_scalars            No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_fused_weights_scalars_dynamic    No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_scalars                          No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fp8_scalars_dynamic                  No       No       No       No        No        No       No       Yes       No       No
mixture_of_experts.fused_weights                        Yes      Yes      No       No        No        No       No       Yes       No       No
ragged_softmax                                          Yes      Yes      Yes      No        No        No       No       No        No       No
reciprocal_fast_math                                    No       Yes      No       No        No        No       No       No        No       No
rms_norm                                                Yes      Yes      Yes      No        No        No       No       No        No       No
rms_norm_backward                                       Yes      No       No       No        No        No       No       No        No       No
rms_norm_fast                                           Yes      Yes      Yes      No        No        No       No       No        No       No
rms_norm_fast_backward                                  Yes      No       No       No        No        No       No       No        No       No
rotary_pos_embedding                                    Yes      Yes      Yes      No        No        No       No       No        No       No
rotary_pos_embedding_backward                           Yes      Yes      Yes      No        No        No       No       No        No       No
rsqrt_fast_math                                         No       Yes      No       No        No        No       No       No        No       No
scaled_masked_softmax                                   No       Yes      No       No        No        No       No       No        No       No
scaled_triangular_softmax                               Yes      Yes      Yes      No        No        No       No       No        No       No
scaled_triangular_softmax_retain                        Yes      Yes      Yes      No        No        No       No       No        No       No
sdpa_bwd                                                Yes      Yes      Yes      No        No        No       No       No        No       No
sdpa_fwd                                                Yes      Yes      Yes      No        No        No       No       No        No       No
sdpa_fwd_dropout                                        Yes      Yes      Yes      No        No        No       No       No        No       No
sdpa_fwd_non_dropout                                    Yes      Yes      Yes      No        No        No       No       No        No       No
sdpa_recomp_bwd                                         Yes      Yes      Yes      No        No        No       No       No        No       No
sdpa_recomp_fwd                                         Yes      Yes      Yes      No        No        No       No       No        No       No
sdpa_recomp_fwd_non_dropout                             Yes      Yes      Yes      No        No        No       No       No        No       No
softmax_fp8                                             No       Yes      No       No        No        No       No       Yes       No       No
softmax_fp8.Scalar                                      No       Yes      No       No        No        No       No       Yes       No       No
softmax_fp8.Scalar_scales                               No       Yes      No       No        No        No       No       Yes       No       No
sqrt_fast_math                                          No       Yes      No       No        No        No       No       No        No       No
sum_fp8                                                 No       No       No       No        No        No       No       Yes       No       No
====================================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
