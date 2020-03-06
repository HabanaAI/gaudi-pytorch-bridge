import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


tensors = {'cpu': [], 'habana': []}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(3 * 3 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        tensors[x.device.type].append(x)
        x = self.conv1(x)
        tensors[x.device.type].append(x)
        x = F.relu(x)
        tensors[x.device.type].append(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        tensors[x.device.type].append(x)
        x = self.conv2(x)
        tensors[x.device.type].append(x)
        x = F.relu(x)
        tensors[x.device.type].append(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        tensors[x.device.type].append(x)
        x = x.view(-1, 3 * 3 * 50)
        tensors[x.device.type].append(x)
        x = self.fc1(x)
        tensors[x.device.type].append(x)
        x = F.relu(x)
        tensors[x.device.type].append(x)
        x = self.fc2(x)
        tensors[x.device.type].append(x)
        x = F.log_softmax(x, dim=1)
        tensors[x.device.type].append(x)

        return x



def cmp_iter(model_cpu, optimizer_cpu, model_hpu, optimizer_hpu, train_loader):
    model_cpu.train()
    model_hpu.train()
    iteration = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to("cpu"), target.to("cpu")
        optimizer_cpu.zero_grad()
        output = model_cpu(data)
        loss = F.nll_loss(output, target)
        tensors['cpu'].append(loss)
        loss.backward()
        tensors['cpu'].extend([
            model_cpu.conv1.weight.grad,
            model_cpu.conv2.weight.grad,
            model_cpu.fc1.weight.grad,
            model_cpu.fc1.weight.grad
            ])

        optimizer_cpu.step()
        tensors['cpu'].extend([
            model_cpu.conv1.weight,
            model_cpu.conv2.weight,
            model_cpu.fc1.weight,
            model_cpu.fc1.weight
            ])


        data, target = data.to("habana"), target.to("habana")
        optimizer_hpu.zero_grad()
        output = model_hpu(data)
        loss = F.nll_loss(output, target)
        tensors['habana'].append(loss)
        loss.backward()
        tensors['habana'].extend([
            model_hpu.conv1.weight.grad,
            model_hpu.conv2.weight.grad,
            model_hpu.fc1.weight.grad,
            model_hpu.fc1.weight.grad
            ])

        optimizer_hpu.step()
        tensors['habana'].extend([
            model_hpu.conv1.weight,
            model_hpu.conv2.weight,
            model_hpu.fc1.weight,
            model_hpu.fc1.weight
            ])

        os.makedirs("tensors/{}/".format(iteration))
        for i in range(len(tensors['cpu'])):
            tensor_cpu = tensors['cpu'][i]
            tensor_hpu = tensors['habana'][i].to('cpu')

            if not torch.allclose(tensor_cpu, tensor_hpu):
                torch.save(tensor_cpu, "tensors/{}/{}-cpu.pt".format(iteration, i))
                torch.save(tensor_hpu, "tensors/{}/{}-hpu.pt".format(iteration, i))
            #print(i, torch.allclose(tensor_cpu, tensor_hpu))
        tensors['cpu'] = []
        tensors['habana'] = []

        iteration += 1

        if iteration == 9:
            break


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-habana', action='store_true', default=False,
                        help='disables habana training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.ops.load_library("../build/libhabana_pytorch_plugin.so")

    torch.manual_seed(args.seed)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_habana else {}
    kwargs = {}  # TODO: do we need any kwargs?
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    torch.manual_seed(0)

    model_cpu = Net().to("cpu")
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=args.lr,
                          momentum=args.momentum)

    torch.manual_seed(0)

    model_hpu = Net().to("habana")
    optimizer_hpu = optim.SGD(model_hpu.parameters(), lr=args.lr,
                          momentum=args.momentum)

    cmp_iter(model_cpu, optimizer_cpu, model_hpu, optimizer_hpu, train_loader)

    print(len(tensors['cpu']), len(tensors['habana']))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
