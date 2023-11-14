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
from habana_frameworks.torch.utils import _debug_C


def format_args(args):
    if args and isinstance(args[0], str):
        format_string = args[0]
        format_args = args[1:]

        if "{}" in format_string:
            # Using {} style format
            formatted_args = ", ".join(map(str, format_args))
            return format_string.format(formatted_args)
        elif "%" in format_string:
            # Using % style format
            return format_string % tuple(format_args)

    return ", ".join(map(str, args))


class Logger:
    def __init__(self, type):
        self.type = type

    def log(self, level, args):
        if _debug_C.is_log_python_enabled(level):
            formatted_msg = f"[{self.type}] {format_args(args)}"
            _debug_C.log_python(level, formatted_msg)

    def trace(self, *args):
        self.log(_debug_C.log_level.trace, args)

    def debug(self, *args):
        self.log(_debug_C.log_level.debug, args)

    def info(self, *args):
        self.log(_debug_C.log_level.info, args)

    def warn(self, *args):
        self.log(_debug_C.log_level.warn, args)

    def error(self, *args):
        self.log(_debug_C.log_level.error, args)
