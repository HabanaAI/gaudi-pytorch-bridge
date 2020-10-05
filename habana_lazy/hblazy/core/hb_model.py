import threading
import _hblazy

_DEVICE_CONTEXTS = dict()
_DEVICE_CONTEXTS_LOCK = threading.Lock()


class DeviceContext(object):
    def __init__(self, device):
        self.device = device


def _get_device_context(device=None):
    if device is None:
        device = _hblazy._hb_get_default_device()

    with _DEVICE_CONTEXTS_LOCK:
        devctx = _DEVICE_CONTEXTS.get(device, None)
        if devctx is None:
            devctx = DeviceContext(device)
            _DEVICE_CONTEXTS[device] = devctx
        return devctx


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
    return devctx


def mark_step():
    _hblazy._hb_step_marker(_hblazy._hb_get_default_device(), [])
    devctx = _run_step_closures()
    devctx.all_reduce_token = None


def optimizer_step(optimizer, barrier=False, optimizer_args={}, groups=None):
    # reduce_gradients(optimizer, groups=groups)
    loss = optimizer.step(**optimizer_args)
    if barrier:
        mark_step()
    return loss
