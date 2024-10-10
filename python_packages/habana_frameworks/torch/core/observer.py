###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
from habana_frameworks.torch.utils.debug import Logger
from torch.ao.quantization.observer import UniformQuantizationObserverBase

"""
This module implements Habana observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

__all__ = [
    "AbsMaxObserver",
    "SimpleAbsMaxObserver",
]

logger = Logger("PT2E-QUANT CUSTOM HABANA OBSERVERS")


class AbsMaxObserver(UniformQuantizationObserverBase):
    """Habana's custom observer module for computing the quantization parameters
    based on the running min, max values. This observer can be used in case per
    tensor symmetric quantization with fp8 quantized data-type is sufficient to
    meet the accuracy need.

    Args:
        `dtype`: Quantized data-type to be used [supported: float8_e4m3fn, float8_e5m2]
        `qscheme`: Quantization scheme to be used [supported: per tensor symmetric]
        `reduce_range`: Reduces the range of the quantized data type by 1 bit [Not yet supported]
        `quant_min`: Minimum quantization value possible. Must be specified.
        `quant_max`: Maximum quantization value possible. Must be specified.
        `eps`: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.
        `backoff_margin`: Backoff margin used in scale calculation, default value is 0.
        `is_dynamic`: If QuantType is DYNAMIC or not. [supported: False]

    .. math::
        This observer implements two main tasks.
        1) Recording of per tensor min, max statistics [during calibration]: If `x` is the
           tensor under observation, the running min `x_min`, max `x_max` are recorded as:

                if `x_min` = None:
                    `x_min` = min(`x`)
                else:
                    `x_min` = min(`x_min`, min(`x`))

                if `x_max` = None:
                    `x_max` = max(`x`)
                else:
                    `x_max` = max(`x_max`, max(`x`))

        2) Computation of quantization parameters using recored min, max: Given that the
           minimum and maximum quantization values possible for the given quantized data type
           is `Q_min` and `Q_max` respectively, the backoff margin is `margin`, the epsilon
           value is `eps` and the recorded min, max values are `x_min` and `x_max` respectively,
           the quantization parameters i.e. scale `s` and zero point `z` are computed as:

                `z` = 0
                `s` = scale_to_pow2_hw(`s_val`)

            where,

                `s_val` = max(`x_absmax` * 2^`margin` / `half_Q_range`, `eps`), and
                scale_to_pow2_hw() is used to align the quantization scale as per h/w
                requirement to support `torch.float8_e4m3fn` or `torch.float8_e5m2`.

            Note: `x_absmax` and `half_Q_range` are derived as shown below.

                `x_min` = 0 if `x_min` > 0 else `x_min`,
                `x_max` = 0 if `x_max` < 0 else `x_max`,
                `x_absmax` = max(-`x_min`, `x_max`)

                `half_Q_range` = (`Q_max` - `Q_min`) / 2
    """

    min_val: torch.Tensor
    max_val: torch.Tensor
    backoff_margin: torch.float32

    def __init__(
        self,
        dtype=torch.float8_e4m3fn,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
        quant_min=torch.finfo(torch.float8_e4m3fn).min,
        quant_max=torch.finfo(torch.float8_e4m3fn).max,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        backoff_margin=0,
        is_dynamic=False,
        **kwargs,
    ) -> None:

        if dtype not in [torch.float8_e5m2, torch.float8_e4m3fn]:
            raise NotImplementedError("AbsMaxObserver: dtype only supports torch.float8_e5m2 and torch.float8_e4m3fn.")
        if qscheme != torch.per_tensor_symmetric:
            raise NotImplementedError("AbsMaxObserver: qscheme only supports torch.per_tensor_symmetric.")
        if reduce_range != False:
            raise NotImplementedError("AbsMaxObserver: reduce_range is not supported.")
        if is_dynamic != False:
            raise NotImplementedError("AbsMaxObserver: is_dynamic is not supported.")

        assert quant_min == -quant_max

        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.backoff_margin = backoff_margin

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""

        def scale_to_pow2_hw(scale, quant_dtype):
            import habana_frameworks.torch.utils.experimental as htexp

            GAUDI2 = htexp.synDeviceType.synDeviceGaudi2
            GAUDI3 = htexp.synDeviceType.synDeviceGaudi3

            EXP_BIAS_SETS = {
                (GAUDI2, torch.float8_e4m3fn): [3, 7, 11, 15],
                (GAUDI2, torch.float8_e5m2): [15],
                (GAUDI3, torch.float8_e4m3fn): range(0, 63),
                (GAUDI3, torch.float8_e5m2): range(0, 63),
            }

            EXP_WIDTH = {torch.float8_e4m3fn: 4, torch.float8_e5m2: 5}

            def get_default_exp_bias(dtype):
                exp_width = EXP_WIDTH[dtype]
                return 2 ** (exp_width - 1) - 1

            MAX_RANGE = {
                torch.float8_e4m3fn: 2 ** ((2**4 - 2 - get_default_exp_bias(torch.float8_e4m3fn)))
                * (2 - 2 ** -(8 - 1 - 4)),
                torch.float8_e5m2: 2 ** ((2**5 - 2 - get_default_exp_bias(torch.float8_e5m2)))
                * (2 - 2 ** -(8 - 1 - 5)),
            }

            def get_fullscale(dtype, exp_bias=None):
                default_exp_bias = get_default_exp_bias(dtype)
                fullscale = MAX_RANGE[dtype]
                exp_bias = default_exp_bias if exp_bias == None else exp_bias
                fullscale = fullscale * (2 ** (default_exp_bias - exp_bias))
                return fullscale

            def get_fullscales_by_expbias_set(dtype, expbias_set):
                return [get_fullscale(dtype, exp_bias=eb) for eb in expbias_set]

            def get_fp8_hw_alligned_scales(dtype, device):
                exp_bias_set = EXP_BIAS_SETS.get((device, dtype), None)
                return (
                    None
                    if exp_bias_set == None
                    else [x / MAX_RANGE[dtype] for x in get_fullscales_by_expbias_set(dtype, exp_bias_set)]
                )

            DEVICES_SCALE_FACTORS = {GAUDI2: 4, GAUDI3: 1}
            FP8_143_SCALES = {
                device: get_fp8_hw_alligned_scales(quant_dtype, device) for device in DEVICES_SCALE_FACTORS.keys()
            }
            FP8_143_SCALES_TRAITS = {
                device: (min(FP8_143_SCALES[device]), max(FP8_143_SCALES[device]), DEVICES_SCALE_FACTORS[device])
                for device in DEVICES_SCALE_FACTORS.keys()
            }

            def scale_to_pow2(scale):
                scale_pow2 = 2 ** torch.ceil(torch.log2(scale))
                return scale_pow2

            scale_pow2 = scale_to_pow2(scale)
            min_scale, max_scale, scale_factor = FP8_143_SCALES_TRAITS[GAUDI2]
            scale_pow2_hw = torch.minimum(
                torch.maximum(
                    2 ** (torch.ceil(torch.log2(scale_pow2) / scale_factor) * scale_factor),
                    torch.tensor(min_scale, dtype=scale.dtype, device=scale.device),
                ),
                torch.tensor(max_scale, dtype=scale.dtype, device=scale.device),
            )

            return scale_pow2_hw

        def calc_maxabs_scale(self):
            min_val_neg = torch.min(self.min_val, torch.zeros_like(self.min_val))
            max_val_pos = torch.max(self.max_val, torch.zeros_like(self.max_val))
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            fullscale = float(self.quant_max - self.quant_min) / 2
            scale = torch.ones(max_val_pos.size(), dtype=torch.float32, device=self.max_val.device.type)
            scale = max_val_pos / fullscale
            scale_adjusted = scale * (2**self.backoff_margin)
            scale_adjusted = torch.max(scale_adjusted, self.eps)
            return scale, scale_adjusted

        scale, scale_adjusted = calc_maxabs_scale(self)
        if (self.dtype == torch.float8_e4m3fn) or (self.dtype == torch.float8_e5m2):
            scale_adjusted = scale_to_pow2_hw(scale_adjusted, self.dtype)
        logger.debug(f"old_scale = {scale.item()}, new_scale = {scale_adjusted.item()}")
        scale = scale_adjusted

        zero_point = torch.zeros(self.max_val.size(), dtype=torch.int64, device=self.max_val.device.type)

        return scale, zero_point

    @torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    @torch.jit.export
    def reset_observer_state(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))


class SimpleAbsMaxObserver(UniformQuantizationObserverBase):
    """Habana's custom observer module for computing the quantization parameters
    based on the running absolute max values. This observer can be used in case per
    tensor symmetric quantization with fp8 quantized data-type is sufficient to
    meet the accuracy need.

    Args:
        `dtype`: Quantized data-type to be used [supported: float8_e4m3fn, float8_e5m2]
        `qscheme`: Quantization scheme to be used [supported: per tensor symmetric]
        `reduce_range`: Reduces the range of the quantized data type by 1 bit [Not yet supported]
        `quant_min`: Minimum quantization value possible. Must be specified.
        `quant_max`: Maximum quantization value possible. Must be specified.
        `eps`: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.
        `backoff_margin`: Backoff margin used in scale calculation, default value is 0.
        `is_dynamic`: If QuantType is DYNAMIC or not. [supported: False]

    .. math::
        This observer implements two main tasks.
        1) Recording of per tensor absolute max statistics [during calibration]: If `x` is the
           tensor under observation, the running absolute maximum `x_absmax` is recorded as:

                if `x_absmax` = None:
                    `x_absmax` = max(abs(`x`))
                else:
                    `x_absmax` = max(`x_absmax`, max(abs(`x`)))

        2) Computation of quantization parameters using recored absolute max: Given that the
           maximum quantization value possible for the given quantized data type is `Q_max`,
           the backoff margin is `margin`, the epsilon value is `eps` and the recorded absolute
           max is `x_absmax`, the quantization parameters i.e. scale `s` and zero point `z` are
           computed as:

                `z` = 0
                `s` = scale_to_pow2_hw(`s_val`)

            where,

                `s_val` = max(`x_absmax` * 2^`margin` / `Q_max`, `eps`), and
                scale_to_pow2_hw() is used to align the quantization scale as per h/w
                requirement to support `torch.float8_e4m3fn` or `torch.float8_e5m2`.
    """

    abs_max_val: torch.Tensor
    backoff_margin: torch.float32

    def __init__(
        self,
        dtype=torch.float8_e4m3fn,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
        quant_min=torch.finfo(torch.float8_e4m3fn).min,
        quant_max=torch.finfo(torch.float8_e4m3fn).max,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        backoff_margin=0,
        is_dynamic=False,
        **kwargs,
    ) -> None:

        if dtype not in [torch.float8_e5m2, torch.float8_e4m3fn]:
            raise NotImplementedError(
                "SimpleAbsMaxObserver: dtype only supports torch.float8_e5m2 and torch.float8_e4m3fn."
            )
        if qscheme != torch.per_tensor_symmetric:
            raise NotImplementedError("SimpleAbsMaxObserver: qscheme only supports torch.per_tensor_symmetric.")
        if reduce_range != False:
            raise NotImplementedError("SimpleAbsMaxObserver: reduce_range is not supported.")
        if is_dynamic != False:
            raise NotImplementedError("SimpleAbsMaxObserver: is_dynamic is not supported.")

        assert quant_min == -quant_max

        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("abs_max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.backoff_margin = backoff_margin

    def forward(self, x_orig):
        r"""Records the running absolute maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.abs_max_val.dtype)
        abs_max_val_cur = torch.amax(torch.abs(x))
        abs_max_val = torch.max(abs_max_val_cur, self.abs_max_val)
        self.abs_max_val.copy_(abs_max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""

        def scale_to_pow2_hw(scale, quant_dtype):
            import habana_frameworks.torch.utils.experimental as htexp

            GAUDI2 = htexp.synDeviceType.synDeviceGaudi2
            GAUDI3 = htexp.synDeviceType.synDeviceGaudi3

            EXP_BIAS_SETS = {
                (GAUDI2, torch.float8_e4m3fn): [3, 7, 11, 15],
                (GAUDI2, torch.float8_e5m2): [15],
                (GAUDI3, torch.float8_e4m3fn): range(0, 63),
                (GAUDI3, torch.float8_e5m2): range(0, 63),
            }

            EXP_WIDTH = {torch.float8_e4m3fn: 4, torch.float8_e5m2: 5}

            def get_default_exp_bias(dtype):
                exp_width = EXP_WIDTH[dtype]
                return 2 ** (exp_width - 1) - 1

            MAX_RANGE = {
                torch.float8_e4m3fn: 2 ** ((2**4 - 2 - get_default_exp_bias(torch.float8_e4m3fn)))
                * (2 - 2 ** -(8 - 1 - 4)),
                torch.float8_e5m2: 2 ** ((2**5 - 2 - get_default_exp_bias(torch.float8_e5m2)))
                * (2 - 2 ** -(8 - 1 - 5)),
            }

            def get_fullscale(dtype, exp_bias=None):
                default_exp_bias = get_default_exp_bias(dtype)
                fullscale = MAX_RANGE[dtype]
                exp_bias = default_exp_bias if exp_bias == None else exp_bias
                fullscale = fullscale * (2 ** (default_exp_bias - exp_bias))
                return fullscale

            def get_fullscales_by_expbias_set(dtype, expbias_set):
                return [get_fullscale(dtype, exp_bias=eb) for eb in expbias_set]

            def get_fp8_hw_alligned_scales(dtype, device):
                exp_bias_set = EXP_BIAS_SETS.get((device, dtype), None)
                return (
                    None
                    if exp_bias_set == None
                    else [x / MAX_RANGE[dtype] for x in get_fullscales_by_expbias_set(dtype, exp_bias_set)]
                )

            DEVICES_SCALE_FACTORS = {GAUDI2: 4, GAUDI3: 1}
            FP8_143_SCALES = {
                device: get_fp8_hw_alligned_scales(quant_dtype, device) for device in DEVICES_SCALE_FACTORS.keys()
            }
            FP8_143_SCALES_TRAITS = {
                device: (min(FP8_143_SCALES[device]), max(FP8_143_SCALES[device]), DEVICES_SCALE_FACTORS[device])
                for device in DEVICES_SCALE_FACTORS.keys()
            }

            def scale_to_pow2(scale):
                scale_pow2 = 2 ** torch.ceil(torch.log2(scale))
                return scale_pow2

            scale_pow2 = scale_to_pow2(scale)
            min_scale, max_scale, scale_factor = FP8_143_SCALES_TRAITS[GAUDI2]
            scale_pow2_hw = torch.minimum(
                torch.maximum(
                    2 ** (torch.ceil(torch.log2(scale_pow2) / scale_factor) * scale_factor),
                    torch.tensor(min_scale, dtype=scale.dtype, device=scale.device),
                ),
                torch.tensor(max_scale, dtype=scale.dtype, device=scale.device),
            )

            return scale_pow2_hw

        def calc_maxabs_scale(self):
            fullscale = float(self.quant_max)
            scale = torch.ones(self.abs_max_val.size(), dtype=torch.float32, device=self.abs_max_val.device.type)
            scale = self.abs_max_val / fullscale
            scale_adjusted = scale * (2**self.backoff_margin)
            scale_adjusted = torch.max(scale_adjusted, self.eps)
            return scale, scale_adjusted

        scale, scale_adjusted = calc_maxabs_scale(self)
        if (self.dtype == torch.float8_e4m3fn) or (self.dtype == torch.float8_e5m2):
            scale_adjusted = scale_to_pow2_hw(scale_adjusted, self.dtype)
        logger.debug(f"old_scale = {scale.item()}, new_scale = {scale_adjusted.item()}")
        scale = scale_adjusted

        zero_point = torch.zeros(self.abs_max_val.size(), dtype=torch.int64, device=self.abs_max_val.device.type)

        return scale, zero_point

    @torch.jit.export
    def extra_repr(self):
        return f"abs_max_val={self.abs_max_val}"

    @torch.jit.export
    def reset_observer_state(self):
        """Resets the absolute max values."""
        self.abs_max_val.copy_(torch.tensor(float("-inf")))
