import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from no20_NN_model_file import *

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(train_data_size, test_data_size)

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# Init NN
KNN = MyNN()
if torch.cuda.is_available():
    KNN = KNN.cuda()

# Loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# Optimizer
optimizer = torch.optim.SGD(KNN.parameters(), lr=0.01)

# Train NN parameters
total_train_step = 0
total_test_step = 0

epoch = 10

# Tensorboard
writer = SummaryWriter("./train_log")

start_time = time.time()

# This code can be deleted
KNN.train()

for i in range(epoch):
    print("------Round {}------".format(i+1))
    for data in train_data_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = KNN(imgs)
        loss = loss_fn(outputs, targets)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        # Print every 100 times
        if total_train_step % 100 == 0:
            print("Time used: ",(time.time()-start_time))
            print("Training {}, Loss = {}".format(total_train_step, loss.item()))
            writer.add_scalar("Train_loss", loss.item(), total_train_step)

    # Test step start:
    # Running on the test data set to see whether the training is ideal enough

    # This code can be deleted
    KNN.eval()

    total_test_loss = 0
    total_accuracy = 0
    # Do not change the gradient
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = KNN(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("Total test set loss: {}".format(total_test_loss))
    print("Total test set accuracy: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("Test_loss", total_test_loss, total_test_step)
    writer.add_scalar("Test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # Save each model
    torch.save(KNN, "MyNN_{}.pth".format(i))
    # torch.save(KNN.state_dict(), "MyNN_{}.pth".format(i))

writer.close()