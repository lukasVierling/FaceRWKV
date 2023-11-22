import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
from tqdm import tqdm
import socket
import time
import os
import argparse

import tensorboardX

from models.FaceRWKV import FaceRWKV, RWKVConfig
from models.CNNmodel import CNNClassifier
from Dataset import CAERSRDataset
from utils import WarmupCosineAnnealingLR

import yaml

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main(args=None):
    #save path
    save_dir = 'checkpoint'
    hostname = socket.gethostname()
    current_time = str(time.time())

    print("cuda avail:", torch.cuda.is_available())
    # get the data set

    #load the config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    #load parameters relevant for training
    train_data_dir = config_dict['training']['train_data_dir']
    val_data_dir = config_dict['training']['val_data_dir']
    batch_size = config_dict['training']['batch_size']
    log_n = config_dict['training']['log_n']
    num_epochs = config_dict['training']['num_epochs']
    warmup_epochs = config_dict['training']['warmup_epochs']
    max_epochs = config_dict['training']['max_epochs']
    warmup_factor = config_dict['training']['warmup_factor']

    # get the model
    config = RWKVConfig()
    if args is not None:
        config.from_yaml(args.config)
    # Safety first
    config.ctx_len = int(config.resolution[0]*config.resolution[1] / (config.patch_size**2)) #should reduce unnecessary padding
     #length of the sequence, necessary to determine positional encoding in model

    train_dataset = CAERSRDataset(train_data_dir, config.resolution)
    val_dataset = CAERSRDataset(val_data_dir, config.resolution)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    #get classes
    config.num_classes = train_dataset.get_classes()
    #model = FaceRWKV(config)
    model = CNNClassifier(train_dataset.get_classes())
    CUDA = torch.cuda.is_available()
    if CUDA:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.to(device)

    # get the loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # get the optimizer
    optimizer = optim.Adam(model.parameters())
    # Learning rate warmup
    warmup_epochs = 1
    max_epochs = 5
    warmup_factor = 0.1
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs, warmup_factor)


    #tensorboard setup
    writer = tensorboardX.SummaryWriter()
    
    #save the config in checkpoint_dir = f'checkpoint/run_{socket.gethostname()}_{time.time()}'
    # Create the directory if it doesn't exist
    yaml_dir = f'checkpoint/run_{socket.gethostname()}_{time.time()}'
    os.makedirs(yaml_dir, exist_ok=True)

    with open(os.path.join(yaml_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)


    # train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Use tqdm to create a progress bar for the training batches
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for i, data in enumerate(pbar):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            # Gradient clipping
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % log_n == log_n-1:
                print('Epoch: %d -------- Step: %5d -------- Loss: %.3f' % (epoch+1, i+1, running_loss/log_n))
                running_loss = 0.0
                #tensorboard logging
                writer.add_scalar('loss', loss.item(), epoch*len(trainloader)+i)
            pbar.set_postfix({'loss': loss.item()})
        # print and log val acc
        val_acc = validate(model, valloader, device)
        print('val acc:', val_acc)
        writer.add_scalar('val_acc', val_acc, epoch)
        # save model every 5 epochs
        if epoch % 5 == 4:
            #torch.save(model.state_dict(), f'checkpoint/run_{socket.gethostname()}_{time.time()}/epoch' + str(epoch+1) + '.pth')
                    # Create the directory if it doesn't exist
            checkpoint_dir = f'checkpoint/run_{socket.gethostname()}_{time.time()}'
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save the model's state dictionary
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

    print('Finished Training')
    # save last model
    #torch.save(model.state_dict(), f'checkpoint/run_{socket.gethostname()}_{time.time()}/epoch' + str(num_epochs) + '.pth')
    checkpoint_dir = f'checkpoint/run_{socket.gethostname()}_{time.time()}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the model's state dictionary
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch{num_epochs}.pth')
    torch.save(model.state_dict(), checkpoint_path)

    def validate(valloader, device):
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
    #add arg parse for a directory path to yaml
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    args = parser.parse_args()

    main(args)