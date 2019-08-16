
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.image as mpimg
import json
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Training a model by inputting parameters into train.py')

parser.add_argument('input', type = str, default='/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg',
                    help='/path/to/image')
parser.add_argument('checkpoint', type = str, default='/home/workspace/ImageClassifier/checkpoint2.pth',
                    help='Checkpoint file (default = /home/workspace/ImageClassifier/checkpoint2.pth)')
parser.add_argument('--category_names', type=str, default = "cat_to_name.json",
                    help='Mapping of categories to real names (default = cat_to_name.json)')
parser.add_argument('--top_k', type=int, default=3,
                    help='Set the top K most likely classes (default = 3)')
parser.add_argument('--gpu', type=str, default='cpu',
                    help='Use gpu for inference, (default = cpu), or gpu: choose based on what is enabled')


args = parser.parse_args()



with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
 


def load_checkpoint(filepath):
    if args.gpu == 'cpu':
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        
    elif args.gpu == 'gpu':
        checkpoint = torch.load(filepath)
        
    if checkpoint['arch'] == 'vgg16_bn':

        model = models.vgg16_bn(pretrained=True)

    elif checkpoint['arch'] == 'vgg13_bn':
        
        model = models.vgg13_bn(pretrained=True) 
    #model = models.vgg16_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])
    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    #model.optimizer = checkpoint['optimizer_state_dict']
    model.optimizer = checkpoint['optimizer_state_dict']
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.epochs = checkpoint['epochs']
    device = torch.device("cuda:0" if args.gpu == 'gpu' else "cpu")
    model.to(device)
    model.eval()
    

    return model  

model   = load_checkpoint(args.checkpoint)


 

def process_image(image):
 
 

    image = Image.open(image)
    if image.height > image.width:
        size = (256,10000)
    else:
        size = (10000,256)
    image.thumbnail(size)
    width, height = image.size

    pil_image = image.crop(((width-224)/2,(height-224)/2,width - (width-224)/2 ,height - (height-224)/2))
 

    image_np = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_np - mean)/std
    image_trans = image_norm.transpose((2,0,1))
    image = image_trans
    
    image = torch.tensor(image, dtype=torch.float)
    
    return image

def imshow(image, ax=None, title=None):
  
    if ax is None:
        fig, ax = plt.subplots()
    
   
    image = image.numpy().transpose((1, 2, 0))
    
  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):

    image = process_image(image_path)
    device = torch.device("cuda:0" if args.gpu == 'gpu' else "cpu")
    image = image.to(device)
    image.unsqueeze_(0)
    logps = model.forward(image)
    ps = torch.exp(logps)
 

    top_ps, top_classes = ps.topk(topk, dim=1)
    probs = top_ps[0].cpu().detach().numpy().tolist()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[int(i)] for i in top_classes[0]]
    
    return probs, classes

probs_test, classes_test =predict(args.input, model, args.top_k)

names = [cat_to_name[i] for i in classes_test]


print('The most likely flower class is' ,names[0], 'with a probability of ', probs_test[0],'.')
print('Less likely probabilities include: ')
for i, j in zip(names[1:],probs_test[1:]):
    print(i, 'with a probability of ', j)
    




