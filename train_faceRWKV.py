import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data

import tensorboardX

from models.FaceRWKV import FaceRWKV, RWKVConfig
from Dataset import CAERSRDataset

# get the data set
data_dir = 'data/CAER-S/train'
dataset = CAERSRDataset(data_dir)
trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# get the model
config = RWKVConfig()
config.num_classes = dataset.get_classes()
model = FaceRWKV(config)

# get the loss function
criterion = nn.CrossEntropyLoss()

# get the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#tensorboard setup
writer = tensorboardX.SummaryWriter()

# train the model
num_epochs = 30
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
            #tensorboard logging
            writer.add_scalar('loss', loss.item(), epoch*len(trainloader)+i)
            writer.add_scalar('accuracy', 0.0, epoch*len(trainloader)+i)
    if epoch % 5 == 4:
        torch.save(model.state_dict(), 'models/CAER-S/epoch' + str(epoch+1) + '.pth')


print('Finished Training')
#save last model
torch.save(model.state_dict(), 'models/CAER-S/epoch' + str(num_epochs) + '.pth')