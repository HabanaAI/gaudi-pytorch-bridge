###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
import sys
from enum import Enum

from habana_frameworks.torch.utils import _debug_C


class LogLevel(Enum):
    TRACE = _debug_C.log_level.trace
    DEBUG = _debug_C.log_level.debug
    INFO = _debug_C.log_level.info
    WARN = _debug_C.log_level.warn
    ERROR = _debug_C.log_level.error
    CRITICAL = _debug_C.log_level.critical


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

    def log(self, level: LogLevel, args):
        logs_enabled = self.is_enabled_for(level)
        if logs_enabled or self.store_data:
            formatted_msg = f"[{self.type}] {format_args(args)}"
            if logs_enabled:
                _debug_C.log_python(level.value, formatted_msg)
            if self.store_data:
                self.data.append(formatted_msg)

    def trace(self, *args):
        self.log(LogLevel.TRACE, args)

    def debug(self, *args):
        self.log(LogLevel.DEBUG, args)

    def info(self, *args):
        self.log(LogLevel.INFO, args)

    def warn(self, *args):
        self.log(LogLevel.WARN, args)

    def error(self, *args):
        print("[ERROR]", *args, file=sys.stderr)
        self.log(LogLevel.ERROR, args)

    def critical(self, *args):
        print("[CRITICAL]", *args, file=sys.stderr)
        self.log(LogLevel.CRITICAL, args)

    def set_store_data(self, enable):
        self.store_data = enable
        self.data = []

    @staticmethod
    def is_enabled_for(level: LogLevel) -> bool:
        return is_log_python_enabled(level)


def get_log_level(logger_level):
    if logger_level == "critical":
        return LogLevel.CRITICAL
    if logger_level == "error":
        return LogLevel.ERROR
    if logger_level == "warn":
        return LogLevel.WARN
    if logger_level == "info":
        return LogLevel.INFO
    if logger_level == "debug":
        return LogLevel.DEBUG
    if logger_level == "trace":
        return LogLevel.TRACE

    assert False, f"unsupported logger_level = {logger_level}"


def enable_logging(logger_name, logger_level):
    log_level = get_log_level(logger_level)
    _debug_C.enable_logging(logger_name, log_level.value)


def refresh_logging_folder_path():
    """
    Helper function to reinitialize logging directory. Directory is read from HABANA_LOGS env var.
    Useful in a case, when logging dir was changed after habana-torch package initialization.
    """
    _debug_C.refresh_hllog_output_dir_from_env()


def is_log_python_enabled(log_level):
    return _debug_C.is_log_python_enabled(log_level.value)
