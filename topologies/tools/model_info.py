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

def model_print(model):
    print("name, param in model.state_dict().items()")
    for name, param in model.state_dict().items():
        print("name: ", name)

def model_print_modules(model):
    print("name, module in model._modules.items()")
    for name, module in model._modules.items():
        print("name: ", name, "module = ", module)

def model_print_named_modules(model):
    print("name, module in model.named_modules")
    for name, module in model.named_modules():
        print("name: ", name, "module = ", module)

def model_find_num_submodules(model):
    k = 0
    for x in model.named_children():
        k= k+1
    return k

def model_print_named_buffers(model):
    for name, buf in model.named_buffers():
        print("buffer name = ", name)

def model_print_unique_named_modules(model):
    print("name, module in unique model.named_modules")
    for name, module in model.named_modules():
        nm = model_find_num_submodules(module)
        if nm == 0:
            print("name: ", name, "module = ", module)

def model_print_named_parameters(model):
    for name, param in model.named_parameters():
        print (name, ', size = ', param.data.size(), flush=True)
