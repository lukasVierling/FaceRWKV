import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data

import tensorboardX

from models.FaceRWKV import FaceRWKV, RWKVConfig
from Dataset import CAERSRDataset

def main():
    print("cuda avail:", torch.cuda.is_available())
    batch_size = 1
    log_n = 500
    # get the data set
    train_data_dir = 'data/CAER-S/train'
    val_data_dir = 'data/CAER-S/validation'

    train_dataset = CAERSRDataset(train_data_dir)
    val_dataset = CAERSRDataset(val_data_dir)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # get the model
    config = RWKVConfig()
    config.num_classes = train_dataset.get_classes()
    config.ctx_len = int(600*400 / (config.patch_size**2)) #should reduce unnecessary padding
    model = FaceRWKV(config)
    CUDA = False
    if CUDA:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    # get the loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

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
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_n == 0: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/log_n))
                running_loss = 0.0
                #tensorboard logging
                writer.add_scalar('loss', loss.item(), epoch*len(trainloader)+i)
        #print and log val acc
        val_acc = validate(model, valloader, device)
        print('val acc:', val_acc)
        writer.add_scalar('val_acc', val_acc, epoch)
        #save model every 5 epochs
        if epoch % 5 == 4:
            torch.save(model.state_dict(), 'models/CAER-S/epoch' + str(epoch+1) + '.pth')


    print('Finished Training')
    #save last model
    torch.save(model.state_dict(), 'models/CAER-S/epoch' + str(num_epochs) + '.pth')

def validate(model, valloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == "__main__":
    main()