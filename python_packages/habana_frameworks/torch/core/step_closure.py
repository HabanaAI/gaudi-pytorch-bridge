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

import threading

import habana_frameworks.torch._core_C as htcore
import torch
from habana_frameworks.torch.utils.internal import is_lazy, lazy_only

_DEVICE_CONTEXTS = dict()
_DEVICE_CONTEXTS_LOCK = threading.Lock()


class _DeviceContext(object):
    def __init__(self, device):
        self.device = device


def _get_device_context(device=None):
    if device is None:
        device = htcore._hb_get_default_device()

    with _DEVICE_CONTEXTS_LOCK:
        devctx = _DEVICE_CONTEXTS.get(device, None)
        if devctx is None:
            devctx = _DeviceContext(device)
            _DEVICE_CONTEXTS[device] = devctx
        return devctx


@lazy_only
def add_step_closure(closure, args=()):
    devctx = _get_device_context()
    step_closures = getattr(devctx, "step_closures", None)
    if step_closures is None:
        step_closures = []
        devctx.step_closures = step_closures
    step_closures.append(lambda a=args: closure(*a))


def _run_step_closures():
    devctx = _get_device_context()
    step_closures = getattr(devctx, "step_closures", None)
    if step_closures is not None:
        devctx.step_closures = []
        for closure in step_closures:
            closure()


def _mark_step_if_lazy(device_str=""):
    if is_lazy():
        mark_step(device_str)


@lazy_only
def mark_step(device_str="", sync=False):
    htcore._mark_step(device_str, sync)
    _run_step_closures()


@lazy_only
def iter_mark_step(device_str=""):
    htcore._iter_mark_step()
    _run_step_closures()
