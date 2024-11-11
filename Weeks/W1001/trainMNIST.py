#COS 470/570: MNIST + Lenet training phase
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

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# switch to train mode
model.train()
for epoch in range(10):  # run for 10 epochs
    epoch_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        #clean gradient for every batch
        optimizer.zero_grad()
        #forward to genreate the prediction
        output = model(data)
        #calculate the loss
        loss = criterion(output, target)
        #update the parameters: backpropagation
        loss.backward()
        optimizer.step()
        
        #sum the loss and the number of correct prediction
        epoch_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss /= total
    accuracy = 100 * correct / total
    print(f'Epoch {epoch}: Average Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # No separate validation set is provided; thus, the test set will be used for validation purposes. 
    # This strategy is flexible and can be adjusted based on available data.
    if epoch % 3 == 0:
        #val the model every 3 epoches
        model.eval()
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
        if accuracy > 0.9:
            #save the model
            torch.save(model.state_dict(), f'./model_epoch_{epoch}.pth')
            print(f'Model saved to ./model_epoch_{epoch}.pth')


