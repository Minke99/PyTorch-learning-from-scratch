import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


# Ref structure
# https://www.google.com/search?q=cifar10+model+structure&sxsrf=AOaemvJaBqK6O4tbT7v82TRlQvoj-nmq-g:1638868357540&tbm=isch&source=iu&ictx=1&fir=biAd2DhtszD5pM%252C_M9rdNwyXaUXLM%252C_%253BdMiH09BVZcdHiM%252C_M9rdNwyXaUXLM%252C_%253BNSvCrenO5Yf6zM%252CYOIYm2TTIa88BM%252C_%253BIe-WNk3Hl9xswM%252CGUirWLAoIqAhmM%252C_%253BScR6gYpvEyurAM%252CePiLEvJFwsfCaM%252C_%253BmEgCgjw7UN33LM%252CGw0MgB6L6RzhGM%252C_%253Bn6zmNVAbCyYjpM%252Cci1Fdf2ki6FAXM%252C_%253BeXa5Kp26jIfQYM%252C8WfGas7f2XEhoM%252C_%253BYFGWhwmkvydBVM%252CrK72KbZMsF1dlM%252C_%253Bx6ZoFu1L_-y8FM%252CRatrC3uEXdAJ9M%252C_%253Bq83Uz7-2Gr_bCM%252CnGPJF9FFZODh0M%252C_%253BaSE6j3GvRFpnWM%252C_M9rdNwyXaUXLM%252C_%253BlhR8kVpIDsO_DM%252CreJu_8rIIeEpNM%252C_%253BmZrjglQiIJwPpM%252CreJu_8rIIeEpNM%252C_&vet=1&usg=AI4_-kTY6H9g_q9Etd-PweMR-HX4uU7a8g&sa=X&sqi=2&ved=2ahUKEwjC_6KyrNH0AhVnFTQIHa0RCwQQ9QF6BAgCEAE#imgrc=biAd2DhtszD5pM


class MyNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64,10)

        # Same as above
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
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Create NN
KNN = MyNN()
print(KNN)

# Test correction
input = torch.ones( 64, 3, 32, 32)
output = KNN(input)
print(output.shape)

writer = SummaryWriter("seq_log")
writer.add_graph(KNN, input)
writer.close()
