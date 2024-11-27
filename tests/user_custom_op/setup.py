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


import pybind11
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
from setuptools import setup
from torch.utils import cpp_extension

torch_include_dir = get_include_dir()
torch_lib_dir = get_lib_dir()
habana_modules_directory = "/usr/include/habanalabs"
pybind_include_path = pybind11.get_include()

modes = {
    "eager": ("habana_pytorch2_plugin", ["hpu_custom_topk.cpp", "hpu_custom_add.cpp"]),
    "lazy": ("habana_pytorch_plugin", ["hpu_custom_topk.cpp", "hpu_custom_add.cpp"]),
    "lazy_legacy": ("habana_pytorch_plugin", ["hpu_legacy_custom_topk.cpp"]),
}

for mode, (plugin, files) in modes.items():
    setup(
        name=f"hpu_custom_op_{mode}",
        ext_modules=[
            cpp_extension.CppExtension(f"hpu_custom_op_{mode}", files, libraries=[plugin], library_dirs=[torch_lib_dir])
        ],
        include_dirs=[
            torch_include_dir,
            habana_modules_directory,
            pybind_include_path,
        ],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )
