import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    #google the normalization for the dataset
        #dataset mean value and standardization
    transforms.Normalize((0.1307,), (0.3081,))
])

#if dont know dataset name, google it
#specifcy parameters
    #if used pytorch to download dataset, use following
    #dont want to download dataset oursevles, so pytorch will do it automatically (download = True)
train_dataset = datasets.MNIST(root='/data',train=True,download=True,transform=transform)

#implement DataLoader
    #first parameter is dataset class
    #next is batch size, up to you (hyperparameter)
    #shuffle (disorder images)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

#use DataLoader to train or do something
#printing shape of each batch, so not doing forward or backwards
    #first is image list, 2nd is labels list
for images, labels in train_dataloader:
    print(images.shape)
    etc...

#data shape:
16 = batch size
    16 images for each batch
1 = channels
    grayscale image
32, 32
    32x32 image resize

#label shape:
16 = label size
    16 labels for each batch
