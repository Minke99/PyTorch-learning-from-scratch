import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("./dataset", train = False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)


class MyNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
KNN = MyNN()
for data in dataloader:
    imgs, targets = data
    outputs = KNN(imgs)
    # print(outputs)
    # print(targets)
    loss_result = loss(outputs, targets)
    print(loss_result)
    # loss_result.backward()
