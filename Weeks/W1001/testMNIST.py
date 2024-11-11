#COS 470/570: MNIST + Lenet test phase
#Xin Zhang

import torch
import torch.nn as nn
import torch.optim as optim
# Import the LeNet-5 model
from LeNet5 import LeNet5
# Import data loaders
from MNIST_DL import train_loader, test_loader

# Model initialization
model = LeNet5()

# Path to the saved model
model_path = './model_epoch_0.pth' 

# Load the saved model
model.load_state_dict(torch.load(model_path))
print(f'Model loaded from {model_path}')

#switch the model to an evaluation model
model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0
correct = 0
total = 0
#set no_grad: because for the evaluation, we do not need to calculate the gradient
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_loss /= total
accuracy = 100 * correct / total
print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')


