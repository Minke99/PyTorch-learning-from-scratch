import torch
import torch.nn.functional as F
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

dataset_transform = torchvision.transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(root = "./conv_dataset", train = False, transform=dataset_transform, download=True)

dataloader = DataLoader(dataset, batch_size=64)

class MyNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

KNN = MyNN()
print(KNN)
step = 0
writer = SummaryWriter("conv_log")
for data in dataloader:
    imgs, targets = data
    output = KNN(imgs)
    print(imgs.shape)
    # torch.Size([16, 3, 32, 32])
    print(output.shape)
    # torch.Size([16, 6, 30, 30])

    writer.add_images("input", imgs, step)
    # Do not know the first input, set it to -1, so that pc will calculate it for us
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step+=1

writer.close()