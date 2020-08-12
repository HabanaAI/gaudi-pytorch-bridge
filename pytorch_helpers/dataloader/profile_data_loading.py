import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pathlib
import os
import time
import matplotlib
import matplotlib.pyplot as plt

X = []
Y = []

def save_points(x, y):
    X.append(x)
    Y.append(y)
    print(x, y)

def profile_pytorch_data_loader_for_resnet():
    matplotlib.use( 'tkagg' )
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    train_dir = pathlib.Path('/software/data/pytorch/imagenet/ILSVRC2012/')
    train = datasets.ImageFolder(train_dir, transform)

    dataloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=32)

    t_sum = 0
    last_time = time.time()

    for i, data in enumerate(dataloader):
        t = time.time()
        t_diff = t - last_time
        t_sum += t_diff
        ##save_points(t_diff, t_sum)
        save_points(i, t_diff)
        last_time = t

        if i >= 100:
            break

    print("Total time take = ", t_sum)

if __name__=='__main__':
    profile_pytorch_data_loader_for_resnet()
    #plt.plot(X,Y)
    #plt.show()
