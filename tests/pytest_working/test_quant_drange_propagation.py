import os
import sys
import json
import torch
import pytest
import torch.nn as nn
from habana_frameworks.torch import hpu
from habana_frameworks.torch.utils.library_loader import load_habana_module


def is_lazy():
    return int(os.environ.get("PT_HPU_LAZY_MODE", 1)) == 1


embedding_dim = 5


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.gelu_impl = nn.GELU(approximate="tanh")
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, y):
        x = torch.bmm(x, y)
        x = self.gelu_impl(x)
        x = self.layer_norm(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, x):
        x = torch.permute(x, (2, 0, 1))
        x = torch.reshape(x, (-1,))
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net1 = Net1()
        self.net2 = Net2()

    def forward(self, x, y):
        x = self.net1(x, y)
        x = self.net2(x)
        return x


def populate_drange(model=None) -> None:
    drange = {"net1.bmm.0": 2.0, "net1.gelu_impl.0": 2.1, "net1.layer_norm.0": 2.2}
    if hpu.is_available():
        model._buffers["ranges"] = dict({"inputs": dict(), "outputs": dict(), "weights": dict()})
        for name, max_value in drange.items():
            nmin = name + ".min_val"
            model._buffers["ranges"]["outputs"][nmin] = torch.tensor(0)
            model._non_persistent_buffers_set.discard(nmin)
            nmax = name + ".max_val"
            model._buffers["ranges"]["outputs"][nmax] = torch.tensor(max_value)
            model._non_persistent_buffers_set.discard(nmax)


@pytest.mark.skipif(not is_lazy(), reason="Lazy only test")
def test_dranges_passed_from_bridge_to_synapse():
    os.environ["GRAPH_VISUALIZATION"] = "1"
    hpu.enable_inference_mode()

    device_cpu = torch.device("cpu")
    device_hpu = torch.device("hpu")

    mat1 = torch.randn((8, 3, 4), dtype=torch.bfloat16)
    mat2 = torch.randn((8, 4, 5), dtype=torch.bfloat16)
    model = Net().eval()

    mat1_in_hpu = mat1.to(device_hpu)
    mat2_in_hpu = mat2.to(device_hpu)
    model = model.to(device_hpu)
    populate_drange(model)

    import habana_frameworks.torch.core as htcore

    htcore.hpu_initialize(model)

    test_out_hpu = model(mat1_in_hpu, mat2_in_hpu)
    htcore.mark_step()
    test_out_cpu = test_out_hpu.to(device_cpu)
    # print('Output:')
    # print(test_out_cpu)

    count = 0
    directory = os.getcwd()
    filename = ".graph_dumps/HabanaFusedOpLazy_0_0-PreGraph-symbol.pbtxt"
    pregraph_file = os.path.join(directory, filename)
    print("pregraph_file: ", pregraph_file)
    if os.path.exists(pregraph_file):
        with open(pregraph_file, "r") as infile:
            data = infile.read()
            count = data.count("drange")
        # print('Count of Tensors with drange:', count)

        results = {}
        results["count"] = count
        results["status"] = "PASS" if count == 20 else "FAIL"
        json_object = json.dumps(results, indent=4)
        print(json_object)

        output_file = "drange_count.log"
        print(f"output file name: {output_file}")
        with open(output_file, "w") as outfile:
            outfile.write("\nResult:\n")
            outfile.write(json_object + "\n")


# if __name__ == '__main__':
#     test_dranges_passed_from_bridge_to_synapse()
