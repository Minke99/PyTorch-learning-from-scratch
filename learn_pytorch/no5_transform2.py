from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

path = "data/img/73451046_p0.png"
writer = SummaryWriter("logs")
img = Image.open(path).convert("RGB")
print(img)

# How to use ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
# input = (input - mean) / std
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 1)

# Resize
print(img.size)
trans_resize = transforms.Resize([512,512])
img_resize = trans_resize(img)
print(img_resize)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 1)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# Combine 2 trans, order matters, first output should be second input
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 2)

# Random crop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()