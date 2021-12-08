from torchvision import transforms
from PIL import Image

# Absolute path
# Relative path
img_path = #Relative path
Image.open(img_path)
# See img info
print(img)

# Create own specific tool
tensor_trans_tool = transforms.ToTensor()
tensor_img = tensor_trans_tool(img)
# See img info
print(tensor_img)

