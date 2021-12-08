from torch.utils.data import Dataset
from PIL import Image
import os
import cv2

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):

        # global var
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)

        # get list of the img path so that for later usage
        # Only contains the name of img, no path
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path_list)



root_dir = "dataset/train"
img_label_dir = "ants_folder"
ants_dataset = MyData(root_dir, img_label_dir)

img, label = ants_dataset[0]
img.show()
len(ants_dataset)

# combine 2 datasets, train_dataset = dataset1 + dataset2