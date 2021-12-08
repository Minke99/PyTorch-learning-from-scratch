import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# Save method 1
# Saves NN and Parameters in NN
# If using this method for own NN, may cause error when loading in another script
# Need to prototype NN class again in the loading script
torch.save(vgg16, "vgg16_method1.pth")

# Save method 2
# Recommended (smaller)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")