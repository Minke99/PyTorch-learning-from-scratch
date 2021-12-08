import torch
from torch import nn


class MyNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input+1

KNN = MyNN()
x = torch.tensor(1.0)
output = KNN(x)
print(output)