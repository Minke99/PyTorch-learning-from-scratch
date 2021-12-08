import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./linear_dataset", train = False, transform=torchvision.transforms.ToTensor(), download=True)

# If not drop last, final set will not match the ideal matrix size and cause error
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)

class MyNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


KNN = MyNN()

for data in dataloader:
    imgs, targets = data
    # reshape has the same functionality to:
    # output = torch.flatten(imgs)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    output = KNN(output)
    print(output.shape)