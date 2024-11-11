#COS 470/570: LFW dataloader example
#Xin Zhang

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class LFWDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        pairs_file = 'pairsDevTrain.txt' if subset == 'train' else 'pairsDevTest.txt'
        self.pairs = self.parse_pairs(os.path.join(root_dir, pairs_file))

    def parse_pairs(self, file_path):
        pairs = []
        with open(file_path, 'r') as file:
            # discard header
            lines = file.readlines()[1:]  
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    # are of the same person
                    person, img1, img2 = parts
                    # 1=true
                    pairs.append((f'{person}/{person}_{int(img1):04d}.jpg', f'{person}/{person}_{int(img2):04d}.jpg', 1))
                elif len(parts) == 4:
                    # different persons
                    person1, img1, person2, img2 = parts
                    #0=false
                    pairs.append((f'{person1}/{person1}_{int(img1):04d}.jpg', f'{person2}/{person2}_{int(img2):04d}.jpg', 0))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name1, img_name2, label = self.pairs[idx]
        img_path1 = os.path.join(self.root_dir, 'lfw-funneled', 'lfw_funneled', img_name1)
        img_path2 = os.path.join(self.root_dir, 'lfw-funneled', 'lfw_funneled', img_name2)
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(label, dtype=torch.int64)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

root_dir = 'LFW'
train_dataset = LFWDataset(root_dir=root_dir, subset='train', transform=transform)
test_dataset = LFWDataset(root_dir=root_dir, subset='test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

for image_1, image_2, label in train_loader:
    print(image_1.shape)
    print(image_2.shape)
    print(label.shape)
    break
    