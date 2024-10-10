###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
import struct
from itertools import product

import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch


def worker_fn(_, filestore_file, cache_pg=True):
    import habana_frameworks.torch.distributed.hccl
    import torch.distributed as dist

    store = dist.FileStore(filestore_file, -1)
    dist.init_process_group(backend="hpu:hccl", rank=0, world_size=1, store=store)

    new_pg = dist.new_group(ranks=[0], backend="hccl")
    new_pg2 = dist.new_group(ranks=[0], backend="hccl")

    if cache_pg:
        # cache PG in module, so they don't be destructed by Python GC
        dist.distributed_c10d.test_pg_cache = [new_pg, new_pg2]


def parse_file_store(file_store_path):
    """Parses FileStore file content and returns each record as separate tuple
    in list.
    """
    records = []

    with open(file_store_path, "rb") as f:
        buff = f.read()
        processed_bytes = 0
        while processed_bytes < len(buff):
            (name_len,) = struct.unpack("i", buff[processed_bytes : (processed_bytes + 4)])
            processed_bytes += 4

            name = buff[processed_bytes : (processed_bytes + name_len)]
            processed_bytes += name_len

            (payload_len,) = struct.unpack("i", buff[processed_bytes : (processed_bytes + 4)])
            processed_bytes += 4

            records.append((name, buff[processed_bytes : processed_bytes + payload_len]))
            processed_bytes += payload_len  # skip payload

    return records


@pytest.mark.parametrize("cache_pg_objects", [True, False])
def test_process_group_destroy_order(tmp_path, cache_pg_objects):
    """Test spawns separate process that creates 3 process groups: default
    process group and 2 sub-groups (via dist.new_group). It verifies if all process
    groups have been destroyed and synced during process tear-down.

    The destruction and order of destructions of process groups is verified via
    reading FileStore (FileStore is used to communicate between workers belonging
    to given process group).

    :param cache_pg_objects: if set to True, process groups objects are assigned
                             to module object, what suppress their destruction by
                             Python GC. Note: PG destroy has to be called anyway.
    """

    filestore_file = f"{tmp_path}/filestore"
    try:
        os.remove(filestore_file)
    except FileNotFoundError:
        pass

    torch.multiprocessing.spawn(
        worker_fn, args=(filestore_file, cache_pg_objects), nprocs=1, join=True, daemon=False, start_method="spawn"
    )

    records = parse_file_store(filestore_file)
    records_keys = [r[0].decode("utf-8") for r in records]

    second_pg_prefix = "/0//2//hpu//"
    first_pg_prefix = "/0//1//hpu//"
    default_pg_prefix = "/0//hpu//"

    host_barrier_key_name = "0HOST_BARRIER:1"
    destroy_key_name = "ProcessGroup::destroy"

    expected_keys_in_store = None

    # In comm cache scenario only default pg exists in records
    if bc.get_pt_enable_comm_group_cache():
        expected_keys_in_store = product([default_pg_prefix], [host_barrier_key_name, destroy_key_name])
    else:
        expected_keys_in_store = product(
            [second_pg_prefix, first_pg_prefix, default_pg_prefix], [host_barrier_key_name, destroy_key_name]
        )
    expected_keys_in_store = ["".join(key_tuple) for key_tuple in expected_keys_in_store]

    records_order_in_store = [records_keys.index(key) for key in expected_keys_in_store]

    assert records_order_in_store == sorted(records_order_in_store)
