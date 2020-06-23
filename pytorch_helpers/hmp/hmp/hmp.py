import torch
import os
import builtins
from .ops_dict import ops_dict
from .utils import *
from . import config


def convert(opt_level='O1', bf16_file_path='', fp32_file_path='', isVerbose=False):
    '''
    Wraps torch functions specified in bf16 and fp32 list to cast their inputs
    Inputs:
    bf16_file_path - User provided file containing list of torch ops that needs to operate on bfloat16 inputs.
    If null string, then default list ops_bf16.txt would be used
    fp32_file_path - User provided file containing list of torch ops that needs to operate on float32 inputs.
    If null string, then default list ops_fp32.txt would be used
    isVerbose - Enable/disable verbose mode
    opt_level:
        O1 - Two files for torch operators are taken as input namely for bfloat16 and float32 dtypes.
        Dtypes for rest of the operators would be decided based on their input tensors
        O2 - GEMM and Conv kernels would operate on bfloat16. Rest of the operators in float32
    '''
    check_input(opt_level, bf16_file_path, fp32_file_path)

    config.verbose_mode = isVerbose
    config.opt_level = opt_level

    print("hmp:verbose_mode ", config.verbose_mode)
    print("hmp:opt_level", config.opt_level)

    # Use default files if user didn't provide
    if (opt_level == 'O1'):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        if (bf16_file_path == ''):
            bf16_file_path = os.path.join(dir_path, 'ops_bf16.txt')

        if (fp32_file_path == ''):
            fp32_file_path = os.path.join(dir_path, 'ops_fp32.txt')

        any_file_path = os.path.join(dir_path, 'ops_any.txt')

        ops_bf16_list = get_list_from_file(bf16_file_path)

        ops_fp32_list = get_list_from_file(fp32_file_path)

        ops_any_list = get_list_from_file(any_file_path)

        # cast ops in the bf16 list
        cast_ops_list(ops_bf16_list, ops_dict, to_bf16)

        # cast ops in the fp32 list
        cast_ops_list(ops_fp32_list, ops_dict, to_fp32)

        # Handle rest of the multi input ops
        # single input op will follow the previous node
        cast_ops_list(ops_any_list, ops_dict)

    elif (opt_level == 'O2'):
        ops_bf16_list = ['conv1d', 'conv2d', 'conv3d', 'linear', 'bmm', 'mm', 'matmul', 'mv', 'dot']
        ops_fp32_list = []

        for key in ops_dict:
            if key not in ops_bf16_list:
                ops_fp32_list.append(key)

        # cast ops in the bf16 list
        cast_ops_list(ops_bf16_list, ops_dict, to_bf16)

        # cast ops in the fp32 list
        cast_ops_list(ops_fp32_list, ops_dict, to_fp32)
