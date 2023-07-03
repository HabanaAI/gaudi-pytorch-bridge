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

import logging
from .config import configuration_flags

DEFAULT_BACKEND_LOGGER = "aot_hpu_backend"
LOG_FILE_NAME = f"{DEFAULT_BACKEND_LOGGER}_log.txt"


def get_compile_backend_logger():
    init_compile_backend_logger()
    return logging.getLogger(DEFAULT_BACKEND_LOGGER)


def _create_console_handled(verbose=False):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    return console_handler


def _get_log_path():
    import os
    log_path_str = os.environ.get("HABANA_LOGS", ".")
    log_path = os.path.join(log_path_str)
    os.makedirs(log_path_str, exist_ok=True)
    # Support multinode
    worker_id = os.environ.get("ID", None)
    if worker_id is not None:
        log_path = os.path.join(log_path, str(worker_id))
    log_file_path = os.path.join(log_path, LOG_FILE_NAME)
    return log_file_path


def _create_file_handler(verbose=False):
    file_handler = logging.FileHandler(_get_log_path())
    file_handler.setLevel(logging.DEBUG)
    return file_handler


def init_compile_backend_logger(verbose=False):
    if not init_compile_backend_logger._logger_ready:
        logging.basicConfig()
        verbose = configuration_flags["verbose"]
        logger = logging.getLogger(DEFAULT_BACKEND_LOGGER)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_create_console_handled(verbose))
        logger.addHandler(_create_file_handler(verbose))
        init_compile_backend_logger._logger_ready = True


init_compile_backend_logger._logger_ready = False
