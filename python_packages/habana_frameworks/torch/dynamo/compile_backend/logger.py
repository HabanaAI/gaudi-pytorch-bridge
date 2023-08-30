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
FX_GRAPHS_LOGGER = "fx_graphs"


def get_compile_backend_logger():
    init_compile_backend_logger()
    return logging.getLogger(DEFAULT_BACKEND_LOGGER)


def get_fx_graph_logger():
    init_fx_graph_logger()
    return logging.getLogger(FX_GRAPHS_LOGGER)


def dump_fx_graph(fx_module, recipe_id):
    logger = get_fx_graph_logger()
    logger.debug("# # # graph_recipe_%d # # #", recipe_id)
    logger.debug(fx_module.print_readable(False))
    logger.debug("IR:\n%s\n\n", fx_module.graph)


def _create_console_handler(verbose=False):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    return console_handler


def _get_log_path(log_file_name=DEFAULT_BACKEND_LOGGER):
    import os
    log_path_str = os.environ.get("HABANA_LOGS", ".")
    log_path = os.path.join(log_path_str)
    # Support multinode
    worker_id = os.environ.get("ID", None)
    if worker_id is not None:
        log_path = os.path.join(log_path, str(worker_id))
    os.makedirs(log_path, exist_ok=True)
    log_file_path = os.path.join(log_path, f"{log_file_name}.log")
    return log_file_path


def _create_file_handler(log_file_name=DEFAULT_BACKEND_LOGGER):
    file_handler = logging.FileHandler(_get_log_path(log_file_name))
    file_handler.setLevel(logging.DEBUG)
    return file_handler


def init_compile_backend_logger(verbose=False):
    if not init_compile_backend_logger._logger_ready:
        logging.basicConfig()
        verbose = configuration_flags["verbose"]
        logger = logging.getLogger(DEFAULT_BACKEND_LOGGER)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_create_console_handler(verbose))
        logger.addHandler(_create_file_handler())
        init_compile_backend_logger._logger_ready = True


init_compile_backend_logger._logger_ready = False


def init_fx_graph_logger():
    if not init_fx_graph_logger._logger_ready:
        logging.basicConfig()
        logger = logging.getLogger(FX_GRAPHS_LOGGER)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_create_file_handler(log_file_name=FX_GRAPHS_LOGGER))
        init_fx_graph_logger._logger_ready = True


init_fx_graph_logger._logger_ready = False
