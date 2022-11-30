import torch
from typing import Union, Any

class HabanaParameterWrapper(torch.nn.Parameter):
    db = {}

    def __init__(self, wrapped):
        HabanaParameterWrapper.db[id(self)] = wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(HabanaParameterWrapper.db[id(self)], name)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        else:
            for k, v in kwargs.items():
                if type(v) == HabanaParameterWrapper:
                    kwargs[k] = HabanaParameterWrapper.db[id(v)]
        args = [HabanaParameterWrapper.db[id(arg)] if type(arg) == HabanaParameterWrapper else arg for arg in args]
        if func.__name__ == "__set__":
            if hasattr(args[0], "device") and hasattr(args[1], "device"):
                if args[0].device != args[1].device:
                    args[0] = args[0].to(args[1].device)
        return super().__torch_function__(func, types, args, kwargs)

def get_habana_parameter(self, result, name):
    if type(result) == torch.nn.Parameter:
        result = HabanaParameterWrapper(result)
        self._parameters[name] = result
    return result

def wrapped__getattr__(self, name: str) -> Union[torch.Tensor, torch.nn.Module]:
    result = self.original__get_attr__(name)
    try:
        if not name in self.checked_parameters:
            result = get_habana_parameter(self, result, name)
            self.checked_parameters.add(name)
    except:
        self.checked_parameters = set(['name'])
        result = get_habana_parameter(self, result, name)
    return result

def wrapped_to(self, *args, **kwargs):
    def for_all_parameters_in_submodules(fn):
        cnt = 0
        def walk(module, fn):
            nonlocal cnt
            for child in module.children():
                walk(child, fn)
            for name,param in module._parameters.items():
                fn(module, name, param, cnt)
                cnt += 1
        walk(self,fn)

    def collect_shared_parameters(module, name, param, cnt):
        nonlocal shared_parameters
        if isinstance(param, HabanaParameterWrapper):
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

    def share_parameters(module, name, param, cnt):
        nonlocal shared_parameters
        nonlocal collected_parameters
        if cnt in shared_parameters:
            for cnt_other in shared_parameters[cnt]:
                module._parameters[name] = collected_parameters[cnt_other]

    def rearrange_shared_parameters(shared_parameters):
        return {shared_parameters[k][0] : shared_parameters[k][1:] for k,v in shared_parameters.items() if len(v) > 1}

    shared_parameters = {}
    collected_parameters = []

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
    weight_sharing_exception =  Exception("Weight sharing unsuccessful."
    "You can proceed without weight sharing by removing:\n"
    "\timport habana_frameworks.torch.core as ht\n"
    "\tht.enable_experimental_weight_sharing()")
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

    #Reattach remote parameters
    if len(collected_parameters_before) != len(collected_parameters_after):
        raise weight_sharing_exception
    for i in range(len(collected_parameters_before)):
        if id(collected_parameters_before[i]) in HabanaParameterWrapper.db:
            HabanaParameterWrapper.db[id(collected_parameters_before[i])] = collected_parameters_after[i]
    return result


def enable_weight_sharing():
    torch.nn.modules.Module.original__get_attr__ = torch.nn.modules.Module.__getattr__
    torch.nn.modules.Module.original_to = torch.nn.modules.Module.to
    torch.nn.modules.Module.__getattr__ = wrapped__getattr__
    torch.nn.modules.Module.to = wrapped_to