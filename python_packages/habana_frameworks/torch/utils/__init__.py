###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import functools

import torch


# Split Index is list which contains which input to split, index starts with 1 if decorater is used and 0 if simple function is called
def split_tensor_batch(_func=None, *, num_splits=1, split_index_list=[]):
    def decorator_split_tensor_batch(func):
        @functools.wraps(func)
        def wrapper_split_tensor_batch(*args, **kwargs):
            call_orig_func = (num_splits == 1) or torch.is_grad_enabled() or (len(split_index_list) == 0)
            if not call_orig_func:
                args_len = len(args)
                split_index_list.sort()
                for i in split_index_list:
                    if (i >= args_len) or (not torch.is_tensor(args[i])):
                        call_orig_func = True
                        break
            if call_orig_func:
                return func(*args, **kwargs)
            else:
                split_tensor = []
                list_of_args = list(args)
                for i in split_index_list:
                    split_tensor.append(torch.tensor_split(args[i], num_splits))

                # Verify if all tensors to be split gave same number of splits
                actual_splits = len(split_tensor[0])
                for i in range(len(split_index_list)):
                    if len(split_tensor[i]) != actual_splits:
                        # If not all input splits are equal, fallback to original function.
                        return func(*args, **kwargs)

                out_tensor = []
                for n in range(actual_splits):
                    for k, i in enumerate(split_index_list):
                        list_of_args[i] = split_tensor[k][n]
                    out_tensor.append(func(*list_of_args, **kwargs))

                return torch.cat(out_tensor)

        return wrapper_split_tensor_batch

    if _func is None:
        return decorator_split_tensor_batch
    else:
        return decorator_split_tensor_batch(_func)
