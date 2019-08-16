import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse

parser = argparse.ArgumentParser(description='Training a model by inputting parameters into train.py')

parser.add_argument('data_dir', type = str, default='flowers',
                    help='Set data directory (default = flowers)')
parser.add_argument('--save_dir', type = str, default='/home/workspace/ImageClassifier/checkpoint2.pth',
                    help='Set directory to save checkpoints (default = /home/workspace/ImageClassifier/checkpoint2.pth)')
parser.add_argument('--arch', type=str, default = "vgg16_bn",
                    help='Choose architecture, (default = vgg16_bn), or vgg13_bn')
parser.add_argument('--learning_rate', type=float, default=0.003,
                    help='Set the learning rate (default = 0.003)')
parser.add_argument('--hidden_units', type=int, default=512,
                    help='Set the number of hidden units (default = 512)')
parser.add_argument('--epochs', type=int, default=3,
                    help='Set the number of epochs (default = 3)')
parser.add_argument('--gpu', type=str, default='cpu',
                    help='Use gpu for training, (default = cpu), or gpu: choose based on what is enabled')


args = parser.parse_args()


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

data_transforms_test = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets_train =  datasets.ImageFolder(train_dir, transform=data_transforms_train)

image_datasets_valid =  datasets.ImageFolder(valid_dir, transform=data_transforms_test)

image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)

dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)

dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=32)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


device = torch.device("cuda:0" if args.gpu == 'gpu' else "cpu")

if args.arch == 'vgg16_bn':

    model = models.vgg16_bn(pretrained=True)

elif args.arch == 'vgg13_bn':
    model = models.vgg13_bn(pretrained=True)

else:
    print('Only vgg16_bn or vgg13_bn are available architectures')

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 60
for epoch in range(epochs):
    for inputs, labels in dataloaders_train:
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders_valid:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {test_loss/len(dataloaders_valid):.3f}.. "
                  f"Valid accuracy: {accuracy/len(dataloaders_valid):.3f}")
            running_loss = 0
            model.train()

# TODO: Do validation on the test set
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders_test:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        test_loss += batch_loss.item()
                    
                    
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
print(f"Test loss: {test_loss/len(dataloaders_test):.3f}.. "
        f"Test accuracy: {accuracy/len(dataloaders_test):.3f}")
          
model.train()            

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets_train.class_to_idx
model.cpu()
checkpoint = {'arch': args.arch,
              'state_dict': model.state_dict(),
             'class_to_idx': model.class_to_idx,
             'epochs': epochs, 
              'classifier': model.classifier,
              'learning_rate': args.learning_rate,
             'optimizer_state_dict': optimizer.state_dict}
torch.save(checkpoint, args.save_dir)


