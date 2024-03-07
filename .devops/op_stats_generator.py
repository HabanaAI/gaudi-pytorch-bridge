#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import importlib.util
import os
import shutil
import sys
from types import ModuleType

OUTPUT_FILES = [
    "consolidate_ops_list.csv",
    "unique_ops_list.csv",
    "unique_ops_list2.csv",
    "summary.csv",
    "consolidate_ops_list.json",
    "unique_ops_list.json",
    "unique_ops_list2.json",
    "summary.json",
]
PT_MODULES_DIR = os.getenv("PYTORCH_MODULES_ROOT_PATH")


def load_op_stats_module() -> ModuleType:
    op_stats_spec = importlib.util.spec_from_file_location(
        "op_stats", os.path.join(PT_MODULES_DIR, "scripts", "op_stats.py")
    )
    module = importlib.util.module_from_spec(op_stats_spec)
    sys.modules["op_stats"] = module
    op_stats_spec.loader.exec_module(module)
    return module


op_stats = load_op_stats_module()


def generate_stats(torch_install_dir: str, build_dir: str, output_dir: str) -> None:
    op_declaration_path = os.path.join(torch_install_dir, "include", "ATen", "RegistrationDeclarations.h")
    op_stats.parse_args_and_run_main(
        [
            f"--ops_decl={op_declaration_path}",
            f"--pt_integ_path={PT_MODULES_DIR}",
            f"--gen_files_path={os.path.join(build_dir, 'generated')}",
        ]
    )

    for result in OUTPUT_FILES:
        shutil.copy(result, build_dir)

        output_file = os.path.join(output_dir, result)
        if os.path.exists(output_file):
            os.remove(output_file)
        shutil.move(result, output_file)
