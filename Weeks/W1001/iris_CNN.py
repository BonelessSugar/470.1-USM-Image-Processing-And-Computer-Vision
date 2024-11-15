#COS 470/570: IRIS classification
#Xin Zhang

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

# Dataset class
class IrisDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset="train"):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['iris-setosa', 'iris-versicolour', 'iris-virginica']
        self.files = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(self.root_dir, cls)
            class_files = [os.path.join(cls_folder, img) for img in os.listdir(cls_folder)]
            random.shuffle(class_files)  # Shuffle the files within this class

            # Calculate split indices for the current class
            if cls == "iris-versicolour":
                total_items = 80
            else:
                total_items = len(class_files)
            train_end = int(0.6 * total_items)
            val_end = int(0.8 * total_items)

            # Split the files based on the subset parameter
            if subset == "train":
                files_subset = class_files[:train_end]
            elif subset == "val":
                files_subset = class_files[train_end:val_end]
            else:  # 'test' or other unspecified values default to test set
                files_subset = class_files[val_end:total_items]

            # Append current class files to the main list
            self.files.extend(files_subset)
            self.labels.extend([idx] * len(files_subset))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# CNN model
class IrisCNN(nn.Module):
    def __init__(self):
        super(IrisCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 64 * 64, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # 3 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
       
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_root = 'iris' 
train_dataset = IrisDataset(dataset_root, transform=transform, subset="train")
val_dataset = IrisDataset(dataset_root, transform=transform, subset="val")
test_dataset = IrisDataset(dataset_root, transform=transform, subset="test")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model, Loss, and Optimizer
model = IrisCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training and Validation
for epoch in range(200):  
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)  
        _, predicted = torch.max(outputs, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item() 
    epoch_loss = running_loss / total  
    epoch_accuracy = 100 * correct / total  

    print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')


    if epoch % 3 == 0:
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}, Validation Accuracy: {100 * correct / total}%, Avg Loss: {val_loss / total}')

# Testing
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total}%, Avg Loss: {test_loss / total}')



