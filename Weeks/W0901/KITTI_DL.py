#COS 470/570: KITTI dataloader example
#Xin Zhang

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

# The KITTI testing folder does not include labels.
# Therefore, a common practice is to divide the training folder into separate training and testing sets.

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_transform=None, subset="train"):
        self.transform = transform
        self.label_transform = label_transform
        self.root_dir = os.path.join(root_dir, "training")
        if subset == "train":
            self.left_images = sorted(os.listdir(os.path.join(self.root_dir, "image_2")))[:160]
            self.right_images = sorted(os.listdir(os.path.join(self.root_dir, "image_3")))[:160]
            #label, Disparity maps
            self.disp_images = sorted(os.listdir(os.path.join(self.root_dir, "disp_noc_0")))[:160]
        else:
            self.left_images = sorted(os.listdir(os.path.join(self.root_dir, "image_2")))[160:]
            self.right_images = sorted(os.listdir(os.path.join(self.root_dir, "image_3")))[160:]
            #label, Disparity maps
            self.disp_images = sorted(os.listdir(os.path.join(self.root_dir, "disp_noc_0")))[160:]

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        img_left_path = os.path.join(self.root_dir, "image_2", self.left_images[idx])
        img_right_path = os.path.join(self.root_dir, "image_3", self.right_images[idx])
        disp_path = os.path.join(self.root_dir, "disp_noc_0", self.disp_images[idx])  

        image_left = Image.open(img_left_path).convert('RGB')
        image_right = Image.open(img_right_path).convert('RGB')
        # Load as grayscale
        disparity = Image.open(disp_path).convert('L')  

        if self.transform:
            image_left = self.transform(image_left)
            image_right = self.transform(image_right)
        if self.label_transform:
            disparity = self.label_transform(disparity)

        return image_left, image_right, disparity


transform = transforms.Compose([
    transforms.Resize((1240, 375)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_transform = transforms.Compose([
    transforms.Resize((1240, 375)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = KITTIDataset(root_dir='data_scene_flow', transform=transform, label_transform=label_transform, subset="train")
test_dataset = KITTIDataset(root_dir='data_scene_flow', transform=transform, label_transform=label_transform, subset="test")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

for batch_idx, (image_left, image_right, disparity) in enumerate(train_loader):
    print(f"Batch index: {batch_idx}")
    print("Shape of left images:", image_left.shape)
    print("Shape of right images:", image_right.shape)
    print("Shape of disparity maps:", disparity.shape)
    break 