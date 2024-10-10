# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

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
