import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

dataset_transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root = "./pool_dataset", utrain = False, transform=dataset_transform, download=True)

dataloader = DataLoader(dataset, batch_size=64)

class MyNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


KNN = MyNN()
output = KNN(input)
print(output)

step = 0
writer = SummaryWriter("ReLU_log")
for data in dataloader:
    imgs, targets = data
    writer.add_images("imput", imgs, step)
    output = KNN(imgs)
    writer.add_images("output", output, step)
    step+=1

writer.close()