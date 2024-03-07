import habana_frameworks.torch.core as htcore
import pytest
import torch
from jit_topologies_test_utils import BasicBlock, MnistNet, Policy, ResNet, mul_op

mnist_test_case_list = [
    # N, C, H, W, trace_file
    (5, 1, 28, 28, None),
]

resnet_test_case_list = [
    # N, C, H, W, trace_file
    (1, 3, 224, 224, None),
]

reinf_test_case_list = [
    # W, H trace_file
    (1, 4, None)
]

mul_test_case_list = [((torch.FloatTensor(2, 3, 4)), (torch.FloatTensor(2, 3, 4)), "./cpu_trace.pt")]


hpu = torch.device("hpu")
cpu = torch.device("cpu")


@pytest.mark.skip(reason="Resnet tracing erroring out due to PT framework converting trace inputs to double")
@pytest.mark.parametrize("N, C, H, W, trace_file", resnet_test_case_list)
def test_resnet18_jit_hpu(N, C, H, W, trace_file):
    input_tensor = torch.FloatTensor(N, C, H, W).to(hpu)
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    try:
        model_trace = get_model_trace_from_device(resnet18, input_tensor, None, hpu, True, False, None)
        print("-----------------Start ResNet--------------------------------\n")
        print("-------------HPU Graph------------------------")
        print(model_trace.graph_for(input_tensor))
    except Exception:
        print("Exiting after printing Fused Graph fusion pass")
        print("-----------------End ResNet--------------------------------\n")
        return


@pytest.mark.skip(reason="Test only for debug purpose to check IR graph")
@pytest.mark.parametrize("tensor_a, tensor_b, trace_file", mul_test_case_list)
def test_mul_jit_hpu(tensor_a, tensor_b, trace_file):
    try:
        tensor_a.to(hpu)
        tensor_b.to(hpu)
        model_trace = get_model_trace_from_device(mul_op, tensor_a, tensor_b, hpu, True, True, trace_file)
        print("-------------------Start Mul-----------------------------\n")
        print("-------------HPU Graph------------------------")
        print(model_trace.graph_for(tensor_a, tensor_b))
    except Exception:
        print("Exiting after printing Fused Graph fusion pass")
        print("-------------------End Mul------------------------------\n")
        return


@pytest.mark.skip(reason="Test only for debug purpose to check IR graph")
@pytest.mark.parametrize("N, C, H, W, trace_file", mnist_test_case_list)
def test_mnist_jit_hpu(N, C, H, W, trace_file):
    input_tensor = torch.FloatTensor(N, C, H, W).to(hpu)
    try:
        model_trace = get_model_trace_from_device(MnistNet(), input_tensor, None, hpu, True, False, None)
        print("-----------------Start MNIST---------------------------\n")
        print("-------------HPU Graph------------------------")
        print(model_trace.graph_for(input_tensor))
    except Exception:
        print("Exiting after printing Fused Graph fusion pass")
        print("------------------End MNIST-------------------------\n")
        return


@pytest.mark.skip(reason="Test only for debug purpose to check IR graph")
@pytest.mark.parametrize("H, W, trace_file", reinf_test_case_list)
def test_reinf_jit_hpu(H, W, trace_file):
    input_tensor = torch.FloatTensor(H, W).to(hpu)
    try:
        model_trace = get_model_trace_from_device(Policy(), input_tensor, None, hpu, True, False, None)
        print("----------------Start Reinf Learning---------------------------\n")
        print("-------------HPU Graph------------------------")
        print(model_trace.graph_for(input_tensor))
        print("-------------------------------------------------\n")
    except Exception:
        print("Exiting after printing Fused Graph fusion pass")
        print("-----------------End Reinf Learning--------------------------\n")
        return


def get_model_trace_from_device(
    model_name,
    input_tensor,
    weight_tensor=None,
    device_name=hpu,
    enable_fusion=True,
    load_trace=False,
    trace_file_path=None,
):
    model_trace = ""
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    if enable_fusion is True:
        htcore.enable()
    else:
        htcore.disable()

    if load_trace is False:
        if weight_tensor is not None:
            device_model = model_name.to(device_name)
            model_trace = torch.jit.trace(device_model, (input_tensor, weight_tensor))
        else:
            device_model = model_name.to(device_name)
            model_trace = torch.jit.trace(device_model, (input_tensor))
    else:
        model_trace = torch.jit.load(trace_file_path, map_location=device_name)

    return model_trace


if __name__ == "__main__":
    test_mnist_jit_hpu(*mnist_test_case_list[0])
    test_reinf_jit_hpu(*reinf_test_case_list[0])
    test_mul_jit_hpu(*mul_test_case_list[0])
