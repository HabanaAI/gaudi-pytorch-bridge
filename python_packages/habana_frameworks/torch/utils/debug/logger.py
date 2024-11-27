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
from habana_frameworks.torch.utils import _debug_C


def format_args(args):
    if args and isinstance(args[0], str):
        format_string = args[0]
        format_args = args[1:]

        if "{}" in format_string:
            # Using {} style format
            if not format_args:
                return ""
            return format_string.format(*map(str, format_args))
        elif "%" in format_string:
            # Using % style format
            return format_string % tuple(format_args)

    return ", ".join(map(str, args))


class Logger:
    def __init__(self, type):
        self.type = type
        self.store_data = False
        self.data = []

    def log(self, level, args):
        logs_enabled = _debug_C.is_log_python_enabled(level)
        if logs_enabled or self.store_data:
            formatted_msg = f"[{self.type}] {format_args(args)}"
            if logs_enabled:
                _debug_C.log_python(level, formatted_msg)
            if self.store_data:
                self.data.append(formatted_msg)

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

    def critical(self, *args):
        self.log(_debug_C.log_level.critical, args)

    def set_store_data(self, enable):
        self.store_data = enable
        self.data = []


def get_log_level(logger_level):
    log_level = None
    if logger_level == "critical":
        log_level = _debug_C.log_level.critical
    elif logger_level == "error":
        log_level = _debug_C.log_level.error
    elif logger_level == "warn":
        log_level = _debug_C.log_level.warn
    elif logger_level == "info":
        log_level = _debug_C.log_level.info
    elif logger_level == "debug":
        log_level = _debug_C.log_level.debug
    elif logger_level == "trace":
        log_level = _debug_C.log_level.trace
    else:
        assert False, f"unsupported logger_level = {logger_level}"
    return log_level


def enable_logging(logger_name, logger_level):
    log_level = get_log_level(logger_level)
    _debug_C.enable_logging(logger_name, log_level)
