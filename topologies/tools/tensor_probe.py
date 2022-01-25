#TensorProbe tool can be used to dump relevant tensors for, say,
#convergence analysis/Debug.
#
#This tool has two parts:
#
#1.ModelParamsDump: Dumps model level tensors like:
#model input, target, outout, loss, trainable params, gradients etc.
#i.e. tensors accessible from outside the model
#
#2.Hooks
#Using the pytorch hooks framework, this tool can dump intermediate
#i/o tensors and gradients b/w pytorch nn modules of the model
#
#Use the following env variables to configure the tool:
#
#TP_HOOKS_ENABLE :
#1 - Enable; 0 - Disable
#
#TP_HOOKS_ITER_INDICES_TO_DUMP:
#iteration indices for which the intermediate tensors are to be dumped.
#'ALL' -dump for all iterations
#Or,
#list of comma separated indices - specific iters to dump.
#Eg:  0,1,6,9
#
#TP_MODEL_PARAM_DUMP_ENABLE :
#1 - Enable; 0 - Disable
#
#TP_MODEL_PARAM_DUMP_ITER_INDICES_TO_DUMP:
#Similar def as above, but for model level tensors.
#
#TP_MODEL_PARAM_DUMP_TENSOR_GROUP
#list of comma separated values representing tensor groups as mentioned below.
#If this env variable is not defined, all tensor groups are dumped.
#If this env variable is defined, only the tensors belonging to the specified
#tensor group are dumped.
#
# value : Tensor group
#  tg   : target
#  ip   : model input
#  op   : model output
#  pb   : params before update
#  pa   : params after update
#  bi   : buffers at input
#  bo   : buffers at output
#  ls   : loss
#  gd   : Gradients
#
#e.g: export TP_MODEL_PARAM_DUMP_TENSOR_GROUP=tg,ip,ls,gd dumps target, input, loss and gradient tensors only
#
#TP_DATA_DUMP_PATH:
#path of directory to dump the tensor data into


from __future__ import print_function
import datetime
import os
import time
import sys
import torch

from .model_info import *

#Hooks TODO:
# 1. Check if we need to enable hooks during eval mode. currentlyhooks are
#    enabled only for Train mode
# 2. Chck if individual hook enabling needs to be done( instead of registering
#    hooks for all modules of the model
# 3. forward pre hook  if there is a need.

tp_config = {
        'TP_HOOKS_ENABLE' : 0,
        'TP_HOOKS_ITER_INDICES_TO_DUMP' : [], # no iteration will be dumped
	'TP_MODEL_PARAM_DUMP_ENABLE': 0,
	'TP_MODEL_PARAM_DUMP_ITER_INDICES_TO_DUMP': [], # no iteration will be dumped
	'TP_MODEL_PARAM_DUMP_TENSOR_GROUP': [],
        'TP_DATA_DUMP_PATH': None
        }

def tp_set_config_from_env():

    def get_flag(env_var, default_val):
        flag = default_val
        flag_str = os.environ.get(env_var)
        if flag_str is not None:
            k = int(flag_str)
            flag = 1 if k > 0 else default_val
        return flag

    def get_csv_to_val(env_var, default_val):
        val = default_val
        val_str = os.environ.get(env_var)
        if val_str is not None:
            if val_str == 'ALL':
                val = 'ALL'
            else:
                val = [int(i) for i in val_str.split(',') if i.isdigit()]
        return val

    def get_str(env_var, default_val):
        val = default_val
        val_str = os.environ.get(env_var)
        if val_str is not None:
            val = val_str
        return val

    global tp_config
    tp_config['TP_HOOKS_ENABLE'] = get_flag('TP_HOOKS_ENABLE', 0)
    tp_config['TP_HOOKS_ITER_INDICES_TO_DUMP'] = get_csv_to_val('TP_HOOKS_ITER_INDICES_TO_DUMP', None)
    tp_config['TP_DATA_DUMP_PATH'] = get_str('TP_DATA_DUMP_PATH', None)
    tp_config['TP_MODEL_PARAM_DUMP_ENABLE'] = get_flag('TP_MODEL_PARAM_DUMP_ENABLE', 0)
    tp_config['TP_MODEL_PARAM_DUMP_ITER_INDICES_TO_DUMP'] = get_csv_to_val('TP_MODEL_PARAM_DUMP_ITER_INDICES_TO_DUMP', None)
    tp_config['TP_MODEL_PARAM_DUMP_TENSOR_GROUP'] = get_str('TP_MODEL_PARAM_DUMP_TENSOR_GROUP', None)
    print("TensorProbe Config ", tp_config)


def tp_model_params_check_tensor_group(group):
    #By default dump all groups of tensors
    if tp_config['TP_MODEL_PARAM_DUMP_TENSOR_GROUP'] == None:
        return True
    if group in tp_config['TP_MODEL_PARAM_DUMP_TENSOR_GROUP']:
        return True
    else:
        return False

class ModelParamsDump(object):
    def __init__(self):
        tp_set_config_from_env()
        self.curr_iter_idx = 0
        self.current_epoch = 0

    def get_data_dump_path(self):
        return tp_config['TP_DATA_DUMP_PATH']

    def to_dump_data(self):
        dump = False
        if tp_config['TP_MODEL_PARAM_DUMP_ENABLE']:
            if tp_config['TP_MODEL_PARAM_DUMP_ITER_INDICES_TO_DUMP'] == 'ALL':
                dump = True
            elif self.curr_iter_idx in tp_config['TP_MODEL_PARAM_DUMP_ITER_INDICES_TO_DUMP']:
                dump = True
        return dump

    def set_current_epoch_no(self, epoch):
        #possibly one epoch completed and moving to the next epoch or starting
        #from a checkpoint. So reset the iteration counter
        if epoch != self.current_epoch:
            self.curr_iter_idx = 0

        self.current_epoch = epoch

    def increment_curr_iter_idx(self):
        self.curr_iter_idx = self.curr_iter_idx + 1

    def dump_params_info(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, " :requires_gard = True",flush=True)
            else:
                print (name, " :requires_gard = False",flush=True)

    def dump_grads(self, device, model, path_modifier=None):
        if self.to_dump_data() is False:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    tensor_name = name + '.grad'
                    self.save_tensor(device, param.grad, tensor_name, path_modifier)

    def dump_buffers_data(self, device, model, path_modifier=None):
        if self.to_dump_data() is False:
            return
        for name, buf in model.named_buffers():
            if 'tracked' not in name: # avoid dumping num_batches_tracked of BN
                tensor_name = name
                self.save_tensor(device, buf, tensor_name, path_modifier)

    def dump_params_data(self, device, model, path_modifier=None):
        if self.to_dump_data() is False:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                #print (name, param.data, flush=True)
                tensor_name = name
                self.save_tensor(device, param.data, tensor_name, path_modifier)

    def save_tensor(self, device, torch_tensor, tensor_name, path_modifier=None, force_dump = False):
        if force_dump == False:
            if self.to_dump_data() is False:
                return
        path = self.get_data_dump_path()
        if (device == torch.device("cpu")):
            path = os.path.join(path, 'cpu')
        elif (device == torch.device("cuda")):
            path = os.path.join(path, 'cuda')
        elif (device == torch.device("hpu") or device == torch.device('hpu:0')):
            path = os.path.join(path, 'hpu')

        if path_modifier is not None:
                path = os.path.join(path, path_modifier)

        path = os.path.join(path,'e'+str(self.current_epoch))
        path = os.path.join(path,'i'+str(self.curr_iter_idx))
        #print(path)

        os.makedirs(path, exist_ok=True)

        torch_tensor_cpu = torch_tensor.to('cpu').detach()

        pyt_t_file_name = os.path.join(path, tensor_name + '.pt')
        torch.save(torch_tensor_cpu, pyt_t_file_name)


class Hook():

    def __init__(self, module, name, device, usr_hook_fn = None, is_forward = True):
        if usr_hook_fn is None:
            if is_forward is True:
                hook_fn = self.fwd_hook_fn
            else:
                hook_fn = self.bwd_hook_fn
        else:
            hook_fn = usr_hook_fn

        self.is_forward = is_forward
        self.name = name
        self.device = device
        self.curr_iter_idx = 0
        self.current_epoch = 0

        self.hook = self.register_hook(module, hook_fn, is_forward)

    def get_data_dump_path(self):
        return tp_config['TP_DATA_DUMP_PATH']

    def to_dump_data(self):
        dump = False
        if tp_config['TP_HOOKS_ENABLE']:
            if tp_config['TP_HOOKS_ITER_INDICES_TO_DUMP'] == 'ALL':
                dump = True
            elif self.curr_iter_idx in tp_config['TP_HOOKS_ITER_INDICES_TO_DUMP']:
                dump = True
        return dump

    def print_io_info(self, module, inputs, outputs):
        print("iter = ", self.curr_iter_idx, " module =", module)

        def arg_info(arg, tag):
            print("Num ",tag, " = ", len(arg))
            for i in range(len(arg)):
                    if arg[i] is not None:
                        print("shape of ", tag," at index ", i,  "is :", arg[i].shape)
                    else:
                        print(tag," at index ", i,  "is None")
        if self.is_forward is True:
            print("FWD: IO info for :", self.name)
        else:
            print("BWD: IO info for :", self.name)

        arg_info(inputs, 'inputs')
        arg_info(outputs, 'outputs')

    def fwd_hook_fn(self, module, inputs, outputs):
        if self.to_dump_data() is False:
            return
        if module.training == False: # For now, use hooks only during training, not during eval
            return
        #self.print_io_info(module, inputs, outputs)
        self.save_tensors(inputs, outputs)

    def bwd_hook_fn(self, module, inputs, outputs):
        if self.to_dump_data() is False:
            return
        #self.print_io_info(module, inputs, outputs)
        self.save_tensors(inputs, outputs)

    def register_hook(self, module, hook_fn, is_forward = True):
        if is_forward is True:
            self.hook = module.register_forward_hook(hook_fn)
        else:
            self.hook = module.register_backward_hook(hook_fn)

    def increment_iteration_idx(self):
        self.curr_iter_idx = self.curr_iter_idx + 1

    def save_tensors(self, inputs, outputs):
        if self.is_forward is True:
            path_modifier = 'frwd'
            input_tag = '_input_'
            output_tag = '_output_'
        else:
            path_modifier = 'bkwd'
            input_tag = '_grad_input_'
            output_tag = '_grad_output_'

        if(isinstance(inputs, torch.Tensor)):
            inputs = [inputs]
        if(isinstance(outputs, torch.Tensor)):
            outputs = [outputs]

        for i in range(len(inputs)):
                if inputs[i] is not None and torch.is_tensor(inputs[i]):
                    tensor_name = self.name + input_tag + str(i)
                    #print(" tensor_name: ", tensor_name, " path_modifier: ", path_modifier)
                    self.save_tensor(inputs[i], tensor_name, path_modifier=path_modifier)

        if isinstance(outputs, torch.Tensor):
            tensor_name = self.name + output_tag + "0"
            self.save_tensor(outputs, tensor_name, path_modifier=path_modifier)
        else:
            for i in range(len(outputs)):
                if outputs[i] is not None and torch.is_tensor(outputs[i]):
                    tensor_name = self.name + output_tag + str(i)
                    #print(" tensor_name: ", tensor_name, " path_modifier: ", path_modifier)
                    self.save_tensor(outputs[i], tensor_name, path_modifier=path_modifier)

    def save_tensor(self, torch_tensor, tensor_name, path_modifier=None, force_dump = False):
        path = self.get_data_dump_path()
        device = self.device

        if (device == torch.device("cpu")):
            path = os.path.join(path, 'cpu')
        elif (device == torch.device("cuda")):
            path = os.path.join(path, 'cuda')
        elif (device == torch.device("hpu") or device == torch.device('hpu:0')):
            path = os.path.join(path, 'hpu')

        if path_modifier is not None:
                path = os.path.join(path, path_modifier)
        path = os.path.join(path,'e'+str(self.current_epoch))
        path = os.path.join(path,'i'+str(self.curr_iter_idx))
        #print(path)

        os.makedirs(path, exist_ok=True)

        torch_tensor_cpu = torch_tensor.to('cpu').detach()

        pyt_t_file_name = os.path.join(path, tensor_name + '.pt')
        torch.save(torch_tensor_cpu, pyt_t_file_name)


#=======Hook Class Ends========================


def tp_hooks_register(model, device):

    tp_set_config_from_env()

    if tp_config['TP_HOOKS_ENABLE'] == 0:
        return None

    fwd_hooks = []
    bwd_hooks = []
    for name, module in model.named_modules():
        # Modules may be organized in a hierarchical manner. i.e. a module may have submodules.
        # So register hooks on the leaf module.
        nm = model_find_num_submodules(module)
        if nm == 0:
            fwd_hooks.append(Hook(module, name, device, None, True))
            bwd_hooks.append(Hook(module, name, device, None, False))
    return [fwd_hooks, bwd_hooks]

def tp_hooks_increment_iteration_idx(hooks):
    if hooks == None:
        return
    if tp_config['TP_HOOKS_ENABLE'] == 0:
        return

    for hk in hooks[0]:
        hk.increment_iteration_idx()
    for hk in hooks[1]:
        hk.increment_iteration_idx()

def tp_hooks_set_current_epoch_no(hooks, epoch):
    if hooks == None:
        return
    if tp_config['TP_HOOKS_ENABLE'] == 0:
        return
    for hk in hooks[0]:
        #possibly one epoch completed and moving to the next epoch or starting
        #from a checkpoint. So reset the iteration counter
        if hk.current_epoch != epoch:
            hk.curr_iter_idx = 0
        hk.current_epoch = epoch
    for hk in hooks[1]:
        if hk.current_epoch != epoch:
            hk.curr_iter_idx = 0
        hk.current_epoch = epoch

# some utility functions to dump important tensors at the beginning and end of a iteration.
def tp_probe_tensors_iteration_start(model, device, target, inp, ParamsDump, force_dump, rank=0):
    if rank == 0:
        if ParamsDump.to_dump_data is False:
            return
        if tp_model_params_check_tensor_group('tg'):
            if isinstance(target, torch.Tensor):
                ParamsDump.save_tensor(device, target, 'target', force_dump=force_dump)
            elif isinstance(target, dict):
                for k, v in target.items():
                    if isinstance(v, torch.Tensor):
                        target_key = 'target_' + k
                        ParamsDump.save_tensor(device, target[k], target_key, force_dump=force_dump)

        if tp_model_params_check_tensor_group('ip'):
            if isinstance(inp, torch.Tensor):
                ParamsDump.save_tensor(device, inp, 'input', force_dump=force_dump)
            elif isinstance(inp, dict):
                for k, v in inp.items():
                    if isinstance(v, torch.Tensor):
                        inp_key = 'input_' + k
                        ParamsDump.save_tensor(device, inp[k], inp_key, force_dump=force_dump)

        if tp_model_params_check_tensor_group('pb'):
            ParamsDump.dump_params_data(device, model, 'params_before_update')
        if tp_model_params_check_tensor_group('bi'):
            ParamsDump.dump_buffers_data(device, model, 'buffers_at_input')

def tp_probe_tensors_iteration_end(model, device, output, loss, ParamsDump, force_dump, rank=0):
    if rank == 0:
        if ParamsDump.to_dump_data is False:
            return
        if tp_model_params_check_tensor_group('op'):
            ParamsDump.save_tensor(device, output, 'output', force_dump=force_dump)
        # Caller can pass a scalar value for loss.
        if tp_model_params_check_tensor_group('ls'):
            if torch.is_tensor(loss):
                ParamsDump.save_tensor(device, loss, 'loss', force_dump=force_dump)
            else:
                ParamsDump.save_tensor(device, torch.tensor(loss), 'loss', force_dump=force_dump)

        if tp_model_params_check_tensor_group('bo'):
            ParamsDump.dump_buffers_data(device, model, 'buffers_at_output')
        if tp_model_params_check_tensor_group('pa'):
            ParamsDump.dump_params_data(device, model, 'params_after_update')
        if tp_model_params_check_tensor_group('gd'):
            ParamsDump.dump_grads(device, model, 'grads')
