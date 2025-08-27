###############################################################################
# Copyright (c) 2021-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import functools
from collections.abc import Callable
from typing import Optional

import torch


def split_tensor_batch(
    _func: Callable | None = None, *, num_splits: int = 1, split_index_list=[], cat_out_index_list=[]
):
    """
    A decorator to automatically split specified tensor arguments along the batch dimension,
    apply the wrapped function to each split independently, and then concatenate selected outputs.

    Args:
        _func (Callable, optional): The function to wrap (used internally by decorator).
        num_splits (int): Number of splits to divide the input tensors into. Default is 1 (no split).
        split_index_list (List[int]): List of argument indices (0-based) indicating which inputs to split.
                                      Inputs can be either tensors or lists/tuples of tensors.
        cat_out_index_list (List[int] | None): Specifies which outputs (0-based indices) should be concatenated.
            - If None: returns all outputs as lists without concatenation.
            - If empty list: concatenates all outputs.
            - If specific list: only those outputs are concatenated.

    Returns:
        Decorated function that automatically splits input tensors and merges selected outputs.

    Raises:
        IndexError: If `cat_out_index_list`/'split_index_list` contains an index out of bounds.
        TypeError: If non-tensor inputs are given for split or non-tensor outputs are given to torch.cat.
    """

    def decorator_split_tensor_batch(func: Callable):
        @functools.wraps(func)
        def wrapper_split_tensor_batch(*args, **kwargs):
            # Decide whether to skip splitting logic
            call_orig_func = (num_splits <= 1) or torch.is_grad_enabled() or (len(split_index_list) == 0)
            # Fallback: Call original function if splitting isn't applicable
            if call_orig_func:
                return func(*args, **kwargs)

            args_len = len(args)
            split_index_list_sorted = sorted(split_index_list)
            split_index_list_sorted = [i + 1 if _func is None else i for i in split_index_list_sorted]

            # Validate split indices
            for i in split_index_list_sorted:
                if i >= args_len:
                    raise IndexError(
                        f"split_index_list contains index {i}, but only {args_len} positional arguments were provided."
                    )
                arg_i = args[i]
                if torch.is_tensor(arg_i):
                    continue
                if len(arg_i) > 0 and torch.is_tensor(arg_i[0]):
                    continue
                raise TypeError(
                    f"Argument at index {i} is neither a tensor nor a non-empty list/tuple of tensors. Got type: {type(arg_i)}"
                )

            # Split specified tensor arguments
            split_tensor = []
            list_of_args = list(args)
            for i in split_index_list_sorted:
                arg_i = args[i]
                if torch.is_tensor(arg_i):
                    split_tensor.append(torch.tensor_split(arg_i, num_splits))
                else:
                    split_tensor.append(arg_i)

            # Ensure all splits have equal number of elements
            actual_splits = len(split_tensor[0])
            if any(len(t) != actual_splits for t in split_tensor):
                raise ValueError(
                    f"All split tensors must have the same number of splits. Found: {[len(t) for t in split_tensor]}"
                )

            all_outputs = []
            # Call function on each split
            for n in range(actual_splits):
                for k, i in enumerate(split_index_list_sorted):
                    list_of_args[i] = split_tensor[k][n]
                all_outputs.append(func(*list_of_args, **kwargs))

            # Normalize outputs
            output_type_list = isinstance(all_outputs[0], list)
            output_type_tuple = isinstance(all_outputs[0], tuple)
            if not (output_type_list or output_type_tuple):
                all_outputs = [(out,) for out in all_outputs]

            num_outputs = len(all_outputs[0])
            cat_out_idx_list = cat_out_index_list

            # Validate output indices for concatenation
            if cat_out_index_list is not None and max(cat_out_idx_list, default=-1) > num_outputs:
                raise IndexError(f"cat_out_index_list has index >= number of outputs ({num_outputs})")

            # Default: cat all outputs if no specific list is given
            if cat_out_index_list is not None and len(cat_out_idx_list) == 0:
                cat_out_idx_list = list(range(num_outputs))

            merged_outputs = []
            for i in range(num_outputs):
                output_chunks = [out[i] for out in all_outputs]

                if cat_out_index_list is None or i not in cat_out_idx_list:
                    merged_outputs.append(output_chunks)
                    continue

                if all(x is None for x in output_chunks):
                    merged_outputs.append(None)
                elif all(isinstance(x, list) for x in output_chunks):
                    merged_outputs.append([torch.cat(tensors, dim=0) for tensors in zip(*output_chunks, strict=False)])
                elif all(torch.is_tensor(x) for x in output_chunks):
                    merged_outputs.append(torch.cat(output_chunks, dim=0))
                else:
                    merged_outputs.append(output_chunks)

            # Return tuple if multiple outputs, else unwrap
            return tuple(merged_outputs) if len(merged_outputs) > 1 else merged_outputs[0]

        return wrapper_split_tensor_batch

    return decorator_split_tensor_batch if _func is None else decorator_split_tensor_batch(_func)
