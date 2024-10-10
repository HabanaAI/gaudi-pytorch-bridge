import os
import shutil

import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch


def graphs_visualization_cleanup(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)


def verify_graph_visualization_created(path, directory):
    full_path = os.path.join(path, directory)
    if os.path.exists(full_path):
        for file in os.listdir(full_path):
            if file.endswith(".pbtxt"):
                return True

    return False


def test_multinode_graph_visualization(rank="37"):
    if pytest.mode == "lazy":
        # For lazy there's no graph compiled, do nothing and leave
        return

    rank_env_var_value = os.environ.get("RANK", None)
    if rank_env_var_value is None:
        os.environ["RANK"] = rank  # Mock env variable set in multi-node by PyTorch

    with bc.env_setting("PT_HPU_GRAPH_DUMP_MODE", "all"):

        def fn(x, y):
            return torch.bmm(x, y)

        graph_dumps_directory = os.environ.get("PT_HPU_GRAPH_DUMP_PREFIX", ".graph_dumps")
        graph_dumps_full_path = os.path.join(f"{os.getcwd()}/{graph_dumps_directory}")

        graphs_visualization_cleanup(graph_dumps_full_path)

        # Register devices
        device_cpu = torch.device("cpu")
        device_hpu = torch.device("hpu")

        # Create tensor
        mat1 = torch.randn((8, 3, 4), dtype=torch.bfloat16)
        mat2 = torch.randn((8, 4, 5), dtype=torch.bfloat16)

        # Transfer to HPU
        mat1_in_hpu = mat1.to(device_hpu)
        mat2_in_hpu = mat2.to(device_hpu)

        # Process
        if pytest.mode == "compile":
            fn = torch.compile(fn, backend="hpu_backend")
        res = fn(mat1_in_hpu, mat2_in_hpu)

        # Get result back to CPU
        res.to(device_cpu)

        result = verify_graph_visualization_created(graph_dumps_full_path, f"rank{rank}")
        graphs_visualization_cleanup(graph_dumps_full_path)

        assert result

    if rank_env_var_value is None:
        del os.environ["RANK"]
