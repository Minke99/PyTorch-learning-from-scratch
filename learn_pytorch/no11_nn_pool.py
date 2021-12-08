import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root = "./pool_dataset", train = False, transform=dataset_transform, download=True)

dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))

class MyNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


KNN = MyNN()
output = KNN(input)
print(output)

step = 0
writer = SummaryWriter("maxpool_log")
for data in dataloader:
    imgs, targets = data
    writer.add_images("imput", imgs, step)
    output = KNN(imgs)
    writer.add_images("output", output, step)
    step+=1

writer.close()