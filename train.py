# Imports here
import numpy as np
import pandas as pd
import seaborn as sb
import time
import datetime
import argparse
from time import time, sleep
from os import listdir
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import json

def main():
    start_time = time()
    in_arg = get_input_args()
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

    save_dir = in_arg.save_dir    
    model = classifier(in_arg)
    
    gpu = (in_arg.gpu == 'True')
    cuda = torch.cuda.is_available()
    if gpu and cuda:        
        model.to('cuda')
        print("GPU")
    else:        
        model.to('cpu')
        print("CPU")
    
    criterion = nn.NLLLoss()
    lr = in_arg.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    epochs = in_arg.epochs
    print_every = 40
    steps = 0
    data_transforms = create_transforms()
    datasets = create_datasets(data_transforms, in_arg)
    dataloaders = create_dataloaders(datasets)
    train(model, dataloaders, criterion, optimizer, epochs, steps, print_every, gpu, cuda)
    validation(model,dataloaders, gpu, cuda)
    save_checkpoint(model, epochs, optimizer, in_arg, datasets, save_dir)

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='path to the image folder')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='save directory')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='torchvision.models model; choices: vgg16, densenet; default: vgg16')
    parser.add_argument('--learning_rate', type=float, default='0.001',
                        help='learning rate for the optimizer; default 0.001')
    parser.add_argument('--hidden_units', type=int, default='512',
                        help='hidden units; default 512')
    parser.add_argument('--epochs', type=int, default='4',
                        help='number of epochs; default: 4')    
    parser.add_argument('--gpu', type=str, default='True',
                        help='gpu; choices: '
                             'True or False; default: True')

    return parser.parse_args()




def classifier(input_arg):
    if input_arg.arch == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier = vgg_classifier(input_arg)
    else:
        model = models.densenet121(pretrained=True)
        model.classifier = densenet_classifier(input_arg)

    return model

def vgg_classifier(input_arg):
    classifier = nn.Sequential(OrderedDict([
                ('fc0', nn.Linear(25088, 1024)),
                ('relu', nn.ReLU()),
                ('fc1', nn.Linear(1024, 500)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(500, 102)),
                ('output',nn.LogSoftmax(dim=1))
                          ]))
    return classifier

def densenet_classifier(input_arg):
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

def train(model, dataloaders, criterion, optimizer, epochs, steps, print_every, gpu, cuda):
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['trainloader']):
            steps += 1
            if gpu and cuda:
                device = 'cuda'
            else:
                device = 'cpu'
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0

def validation(model, dataloaders, gpu, cuda):
    if gpu and cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    correct = 0
    total = 0
    #model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['validationloader']:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))





def create_transforms():
    from torchvision import datasets, transforms, models
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    training_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.RandomRotation(30),
                                              transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                        ])
    testing_transforms = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224), transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224), transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    data_transforms = {'training_transforms': training_transforms,
                    'testing_transforms': testing_transforms,
                    'validation_transforms': validation_transforms
                    }
    return data_transforms


def create_datasets(data_transforms, in_arg):
    from torchvision import datasets, transforms, models
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Load the datasets with ImageFolder
    training_set = datasets.ImageFolder(train_dir, transform = data_transforms['training_transforms'])
    testing_set  = datasets.ImageFolder(test_dir, transform = data_transforms['testing_transforms'])
    validation_set = datasets.ImageFolder(valid_dir, transform = data_transforms['validation_transforms'])

    datasets = {'training_set': training_set,
                'testing_set': testing_set,
                'validation_set': validation_set
                }

    return datasets


def create_dataloaders(datasets):
    trainloader = torch.utils.data.DataLoader(datasets['training_set'], batch_size=64, shuffle=True)
    testloader =   torch.utils.data.DataLoader(datasets['testing_set'], batch_size=32)
    validationloader = torch.utils.data.DataLoader(datasets['validation_set'], batch_size=32)
    dataloaders = {'trainloader': trainloader,
                    'testloader': testloader,
                    'validationloader': validationloader
                    }

    return dataloaders

def save_checkpoint(model, epochs, optimizer, in_arg, datasets,save_dir):       
    if in_arg.save_dir is not None:
        save_directory = save_dir
    model.class_to_idx = datasets['training_set'].class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'epochs': 6,
                  'model' : models.vgg16(pretrained=True),
                  'model_index' : model.class_to_idx,
                  'optimizer' : optimizer.state_dict(),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, save_directory)
    print('Save was successful')

# Call to main function to run the program
if __name__ == "__main__":
    main()
