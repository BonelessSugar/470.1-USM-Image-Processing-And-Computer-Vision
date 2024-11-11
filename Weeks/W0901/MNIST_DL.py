#COS 470/570: MNIST dataloader example
#Xin Zhang

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((32,32)),
    # Converts image to PyTorch format
    transforms.ToTensor(),  
    # Normalization
    transforms.Normalize((0.1307,), (0.3081,))  
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

for batch_id, (data, label) in enumerate(train_loader):
    print("Batch ID:" + str(batch_id))
    print("Data Shape:")
    print(data.shape)
    print("Label Shape:")
    print(label.shape)
    break
