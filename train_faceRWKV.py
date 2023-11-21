import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
from tqdm import tqdm

import tensorboardX

from models.FaceRWKV import FaceRWKV, RWKVConfig
from Dataset import CAERSRDataset

def main():
    print("cuda avail:", torch.cuda.is_available())
    batch_size = 16
    # get the data set
    data_dir = 'data/CAER-S/train'
    dataset = CAERSRDataset(data_dir)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # get the model
    config = RWKVConfig()
    config.num_classes = dataset.get_classes()
    model = FaceRWKV(config)
    CUDA = True
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

    # Assuming you have already defined your trainloader, model, criterion, optimizer, and writer

    num_epochs = 30

    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(enumerate(trainloader, 0), total=len(trainloader)) as pbar:
            for i, data in pbar:
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
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    pbar.set_postfix({'Loss': running_loss / 2000})
                    running_loss = 0.0
                    # tensorboard logging
                    writer.add_scalar('loss', loss.item(), epoch * len(trainloader) + i)
                    writer.add_scalar('accuracy', 0.0, epoch * len(trainloader) + i)

            if epoch % 5 == 4:
                torch.save(model.state_dict(), 'models/CAER-S/epoch' + str(epoch + 1) + '.pth')

    # Save the last model
    torch.save(model.state_dict(), 'models/CAER-S/epoch' + str(num_epochs) + '.pth')

    print('Finished Training')

if __name__ == "__main__":
    main()