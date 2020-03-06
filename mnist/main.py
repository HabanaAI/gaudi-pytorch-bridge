from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

generic_path = './dumped_tensors/'
iteration = 0


def save_tensor(torch_tensor, tensor_name, path):
    import os
    os.makedirs(path, exist_ok=True)

    with open(path + tensor_name + '_metadata', 'w') as file:  # reset file
        file.write(str(torch_tensor.size()) + '\n' + str(torch_tensor.stride()))

    np_tensor = torch_tensor.to('cpu').detach().numpy()
    np.save(path + tensor_name + '_data', np_tensor)


def save_gradients(model):
    path = generic_path + 'iter_' + str(iteration) + '/'
    save_tensor(model.conv1.weight.grad, 'conv1.grad', path)
    save_tensor(model.conv2.weight.grad, 'conv2.grad', path)
    save_tensor(model.fc1.weight.grad, 'fc1.grad', path)
    save_tensor(model.fc2.weight.grad, 'fc2.grad', path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(3 * 3 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # import sys
        # import numpy as np
        # sys.path.append('/home/jgrzybek/development/pytorch-integration/tests')
        # from test_utils import compare_tensors

        # import pudb
        # pudb.set_trace()
        path = generic_path + 'iter_' + str(iteration) + '/'
        tensor_nr = 0

        save_tensor(x, str(tensor_nr) + '_input', path)
        tensor_nr += 1
        save_tensor(self.conv1.weight, str(tensor_nr) + '_conv1_weights', path)
        tensor_nr += 1
        x = self.conv1(x)
        save_tensor(x, str(tensor_nr) + '_conv1_out', path)
        tensor_nr += 1
        x = F.relu(x)
        save_tensor(x, str(tensor_nr) + '_relu1_out', path)
        tensor_nr += 1
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        save_tensor(x, str(tensor_nr) + '_maxpool1_out', path)
        tensor_nr += 1
        save_tensor(self.conv2.weight, str(tensor_nr) + '_conv2_weights', path)
        tensor_nr += 1
        x = self.conv2(x)
        save_tensor(x, str(tensor_nr) + '_conv2_out', path)
        tensor_nr += 1
        x = F.relu(x)
        save_tensor(x, str(tensor_nr) + '_relu2_out', path)
        tensor_nr += 1
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        save_tensor(x, str(tensor_nr) + '_maxpool2_out', path)
        tensor_nr += 1
        x = x.view(-1, 3 * 3 * 50)
        save_tensor(x, str(tensor_nr) + '_view_out', path)
        tensor_nr += 1
        save_tensor(self.fc1.weight, str(tensor_nr) + '_fc1_weights', path)
        tensor_nr += 1
        x = self.fc1(x)
        save_tensor(x, str(tensor_nr) + '_fc1_out', path)
        tensor_nr += 1
        x = F.relu(x)
        save_tensor(x, str(tensor_nr) + '_relu3_out', path)
        tensor_nr += 1
        save_tensor(self.fc2.weight, str(tensor_nr) + '_fc2_weights', path)
        tensor_nr += 1
        x = self.fc2(x)
        save_tensor(x, str(tensor_nr) + '_fc2_out', path)
        tensor_nr += 1
        x = F.log_softmax(x, dim=1)
        save_tensor(x, str(tensor_nr) + '_logsoftmax_out', path)
        tensor_nr += 1

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    global iteration
    with open('mnistpy.log', 'w') as file:  # reset file
        file.write('')

    for batch_idx, (data, target) in enumerate(train_loader):
        import pudb
        pudb.set_trace()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        save_gradients(model)
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        with open('mnistpy.log', 'a') as file:
            file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        iteration = iteration + 1
        if iteration == 2:
            raise 'FINISH'


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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

    use_habana = not args.no_habana
    if use_habana:
        torch.ops.load_library("libhabana_pytorch_plugin.so")

    torch.manual_seed(args.seed)

    device = torch.device("habana" if use_habana else "cpu")
    global generic_path
    generic_path = generic_path + ("hpu/" if use_habana else "cpu/")

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

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
