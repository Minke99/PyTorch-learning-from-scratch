import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Test set
test_data = torchvision.datasets.CIFAR10(root ="./dataset", train = False, transform=dataset_transform, download=True)

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

print(test_data[0])

writer = SummaryWriter("dataloader_log")

step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step += 1

writer.close()