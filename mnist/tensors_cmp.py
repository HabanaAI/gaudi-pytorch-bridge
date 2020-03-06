import argparse
import os
import torch

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='compare tensors')
    parser.add_argument('--tensor', type=int, default=1,
                        help='tensor number')
    parser.add_argument('--iteration', type=int, default=1,
                        help='iteration number')
    args = parser.parse_args()

    tensor_files = os.listdir("tensors/{}".format(args.iteration))
    tensor_numbers = list(set(map(lambda x: int(x.split('-')[0]), tensor_files)))
    tensor_numbers.sort()

    for no in tensor_numbers:
        tensor_cpu = torch.load("tensors/{}/{}-cpu.pt".format(args.iteration, no))
        tensor_hpu = torch.load("tensors/{}/{}-hpu.pt".format(args.iteration, no))

        isclose = torch.isclose(tensor_cpu, tensor_hpu)

        torch.set_printoptions(precision=20)
        diff_cpu = tensor_cpu[isclose.logical_not()]
        diff_hpu = tensor_hpu[isclose.logical_not()]
        print(no,
              "  Number of different/all elements: {}/{}".format(
                  diff_cpu.size()[0], tensor_cpu.flatten().size()[0]),
              "  Maximum difference between values: {}".format(
                  torch.max(torch.abs(diff_cpu - diff_hpu))),
              "  Avg difference between values: {}".format(
                  torch.mean(torch.abs(diff_cpu - diff_hpu))),
              sep="\n")


if __name__ == '__main__':
    main()
