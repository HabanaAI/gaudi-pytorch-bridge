from typing import Any, Union

import torch


def wrapped__new__(cls, data=None, requires_grad=True):
    if type(data) is HabanaParameterWrapper:
        return torch.Tensor._make_subclass(cls, data, requires_grad)
    else:
        return wrapped__new__.original__new__(cls, data, requires_grad)


class HabanaParameterWrapper(torch.nn.Parameter):
    db = {}

    def __getattribute__(self, name: str) -> Any:
        try:
            self_ = HabanaParameterWrapper.db[id(self)]
        except KeyError:
            HabanaParameterWrapper.db[id(self)] = self
            self_ = self
        if name == "__torch_function__":
            return HabanaParameterWrapper.__torch_function__
        return object.__getattribute__(self_, name)

    def __setattr__(self, name: str, value) -> None:
        try:
            self_ = HabanaParameterWrapper.db[id(self)]
        except KeyError:
            HabanaParameterWrapper.db[id(self)] = self
            self_ = self
        return object.__setattr__(self_, name, value)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        else:
            for k, v in kwargs.items():
                if type(v) == HabanaParameterWrapper:
                    kwargs[k] = HabanaParameterWrapper.db[id(v)]
        new_args = [None] * len(args)
        for i in range(len(args)):
            arg = args[i]
            if type(arg) is list:
                new_args[i] = [
                    HabanaParameterWrapper.db[id(inner_arg)] if type(inner_arg) == HabanaParameterWrapper else inner_arg
                    for inner_arg in arg
                ]
            else:
                new_args[i] = HabanaParameterWrapper.db[id(arg)] if type(arg) == HabanaParameterWrapper else arg
        if func.__name__ == "__set__":
            if hasattr(new_args[0], "device") and hasattr(new_args[1], "device"):
                if new_args[0].device != new_args[1].device:
                    new_args[0].change_device_placement(new_args[1].device)
                    return
        return super().__torch_function__(func, types, new_args, kwargs)

    def __del__(self):
        try:
            del HabanaParameterWrapper.db[id(self)]
        except:
            pass


def update_habana_parameter(result):
    if type(result) == torch.nn.Parameter:
        result.__class__ = HabanaParameterWrapper
        HabanaParameterWrapper.db[id(result)] = result


def wrapped__getattr__(self, name: str) -> Union[torch.Tensor, torch.nn.Module]:
    result = self.original__get_attr__(name)
    try:
        if not name in self.checked_parameters:
            update_habana_parameter(result)
            self.checked_parameters.add(name)
    except:
        self.checked_parameters = set(["name"])
        update_habana_parameter(result)
    return result


class DevicePlacementChanger:
    def __init__(self, module, name, param) -> None:
        self.module = module
        self.name = name
        self.param = param

    def __call__(self, new_device):
        # changing parameters data tensors in HPU is impossible
        # to W/A this we create copy of original parameter
        copied = self.module._parameters[self.name].clone().to(new_device)
        # assigning data tensor to empty tensor - allows to free memory on device
        self.module._parameters[self.name].data = torch.empty([], device=self.param.device)
        # reassign previously copied parameter
        del self.module._parameters[self.name]
        self.module._parameters[self.name] = copied
        # convert to habana parameter
        self.module._parameters[self.name].__class__ = HabanaParameterWrapper
        self.module._parameters[self.name].change_device_placement = self
        HabanaParameterWrapper.db[id(self.param)] = copied


def wrapped_to(self, *args, **kwargs):
    def for_all_parameters_in_submodules(fn):
        cnt = 0

        def walk(module, fn):
            nonlocal cnt
            for child in module.children():
                walk(child, fn)
            for name, param in module._parameters.items():
                fn(module, name, param, cnt)
                cnt += 1

        walk(self, fn)

    def collect_shared_parameters(module, name, param, cnt):
        nonlocal shared_parameters
        if isinstance(param, HabanaParameterWrapper):
            if id(param) not in HabanaParameterWrapper.db:
                raise weight_sharing_exception
            param_id = id(HabanaParameterWrapper.db[id(param)])
        else:
            param_id = id(param)

        if param_id in shared_parameters:
            shared_parameters[param_id].append(cnt)
        else:
            shared_parameters[param_id] = [cnt]

    def collect_parameters(module, name, param, cnt):
        nonlocal collected_parameters
        collected_parameters.append(param)

    def convert_to_habana_parameters(module, name, param, cnt):
        if not isinstance(param, HabanaParameterWrapper) and param is not None:
            module._parameters[name].__class__ = HabanaParameterWrapper
            HabanaParameterWrapper.db[id(param)] = param

    def add_device_placement_workaround(module, name, param, cnt):
        if param is None:
            return

        nonlocal self
        param.change_device_placement = DevicePlacementChanger(module, name, param)

    def share_parameters(module, name, param, cnt):
        nonlocal shared_parameters
        nonlocal collected_parameters
        if cnt in shared_parameters:
            if len(collected_parameters) <= shared_parameters[cnt]:
                raise weight_sharing_exception
            module._parameters[name] = collected_parameters[shared_parameters[cnt]]

    def rearrange_shared_parameters(shared_parameters):
        return {shared_param: params[0] for params in shared_parameters.values() for shared_param in params[1:]}

    shared_parameters = {}
    collected_parameters = []
    weight_sharing_exception = Exception(
        "Weight sharing unsuccessful. " "You can disable weight sharing by setting: PT_HPU_WEIGHT_SHARING=0"
    )

    # Convert all parameters to habana parameters
    for_all_parameters_in_submodules(convert_to_habana_parameters)

    # Collect all parameters
    for_all_parameters_in_submodules(collect_parameters)
    collected_parameters_before = collected_parameters.copy()
    collected_parameters = []

    # Collect shared parameters
    for_all_parameters_in_submodules(collect_shared_parameters)
    shared_parameters = rearrange_shared_parameters(shared_parameters)

    # Call original model.to
    result = self.original_to(*args, **kwargs)

    # Collect all new parameters
    for_all_parameters_in_submodules(collect_parameters)
    collected_parameters_after = collected_parameters

    # Recreate shared parameters
    for_all_parameters_in_submodules(share_parameters)

    # Validate shared parameters
    shared_parameters_before = shared_parameters.copy()
    shared_parameters = {}
    for_all_parameters_in_submodules(collect_shared_parameters)
    shared_parameters_after = rearrange_shared_parameters(shared_parameters)
    if len(shared_parameters_before) != len(shared_parameters_after):
        raise weight_sharing_exception
    for key_before, value_before in shared_parameters_before.items():
        if key_before not in shared_parameters_after:
            raise weight_sharing_exception
        if value_before != shared_parameters_after[key_before]:
            raise weight_sharing_exception

    # Reattach remote parameters
    if len(collected_parameters_before) != len(collected_parameters_after):
        raise weight_sharing_exception
    for i in range(len(collected_parameters_before)):
        if id(collected_parameters_before[i]) in HabanaParameterWrapper.db and id(collected_parameters_before[i]) != id(
            collected_parameters_after[i]
        ):
            HabanaParameterWrapper.db[id(collected_parameters_before[i])] = collected_parameters_after[i]

    for key, value in HabanaParameterWrapper.db.items():

        def get_value(value):
            if id(value) in HabanaParameterWrapper.db and id(HabanaParameterWrapper.db[id(value)]) != id(value):
                return get_value(HabanaParameterWrapper.db[id(value)])
            else:
                return value

        HabanaParameterWrapper.db[key] = get_value(value)

    # Recreate shared parameters
    for_all_parameters_in_submodules(add_device_placement_workaround)
    return result


def enable_weight_sharing():
    torch.nn.modules.Module.original__get_attr__ = torch.nn.modules.Module.__getattr__
    torch.nn.modules.Module.original_to = torch.nn.modules.Module.to
    torch.nn.modules.Module.__getattr__ = wrapped__getattr__
    torch.nn.modules.Module.to = wrapped_to
    wrapped__new__.original__new__ = torch.nn.Parameter.__new__
    torch.nn.Parameter.__new__ = wrapped__new__
