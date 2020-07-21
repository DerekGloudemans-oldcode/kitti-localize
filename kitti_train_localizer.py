# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:45:48 2020

@author: derek
"""
import os
import numpy as np
import random 
import math
import time
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F
import matplotlib.pyplot  as plt

from kitti_localization_dataset import Localization_Dataset
from detrac_localization_dataset import Localize_Dataset


import warnings
warnings.filterwarnings(action='once')



def iou(a,b):
    """
    Description
    -----------
    Calculates intersection over union for all sets of boxes in a and b

    Parameters
    ----------
    a : a torch of size [batch_size,4] of bounding boxes.
    b : a torch of size [batch_size,4] of bounding boxes.

    Returns
    -------
    mean_iou - float between [0,1] with average iou for a and b
    """
    
    area_a = a[:,2] * a[:,2] * a[:,3]
    area_b = b[:,2] * b[:,2] * b[:,3]
    
    minx = torch.max(a[:,0]-a[:,2]/2, b[:,0]-b[:,2]/2)
    maxx = torch.min(a[:,0]+a[:,2]/2, b[:,0]+b[:,2]/2)
    miny = torch.max(a[:,1]-a[:,2]*a[:,3]/2, b[:,1]-b[:,2]*b[:,3]/2)
    maxy = torch.min(a[:,1]+a[:,2]*a[:,3]/2, b[:,1]+b[:,2]*b[:,3]/2)
    zeros = torch.zeros(minx.shape,dtype = float)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection
    iou = torch.div(intersection,union)
    mean_iou = torch.mean(iou)
    
    return mean_iou

# define ResNet18 based network structure
class ResNet_Localizer(nn.Module):
    
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate some nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_Localizer, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        #self.feat = models.resnet18(pretrained=True)
        self.feat = models.resnet34(pretrained = True)
        # get size of some layers
        start_num = self.feat.fc.out_features
        mid_num = int(np.sqrt(start_num))
        
        cls_out_num = 8 # 
        reg_out_num = 4 # bounding box coords
        embed_out_num = 128
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,cls_out_num,bias = True)
                          #nn.Softmax(dim = 1)
                          )
        
        # define regressor
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,reg_out_num,bias = True),
                          nn.ReLU()
                          )
        
        self.embedder = nn.Sequential(
                  nn.Linear(start_num,start_num // 3,bias=True),
                  nn.ReLU(),
                  nn.Linear(start_num // 3,embed_out_num,bias = True),
                  nn.ReLU()
                  )
        
        for layer in self.classifier:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        for layer in self.regressor:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        for layer in self.embedder:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
            
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.feat(x)
        cls_out = self.classifier(features)
        reg_out = self.regressor(features)
        #embed_out = self.embedder(features)
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out

def train_model(model, optimizer, scheduler,losses,
                    dataloaders,device, patience= 10, start_epoch = 0,
                    all_metrics = None):
        """
        Alternates between a training step and a validation step at each epoch. 
        Validation results are reported but don't impact model weights
        """
        max_epochs = 500
        
        # for storing all metrics
        if all_metrics == None:
          all_metrics = {
                  'train_loss':[],
                  'val_loss':[],
                  "train_acc":[],
                  "val_acc":[]
                  }
        
        # for early stopping
        best_loss = np.inf
        epochs_since_improvement = 0

        for epoch in range(start_epoch,max_epochs):
            for phase in ["train","val"]:
                if phase == 'train':
                    
                    if len(all_metrics["val_loss"]) > 0:
                        scheduler.step(all_metrics["val_loss"][-1]) # adjust learning rate after so many epochs
                        
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                # Iterate over data.
                count = 0
                total_loss = 0
                total_acc = 0
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        cls_out,reg_out = model(inputs)
                        
                        loss = 0
                        each_loss = []
                        # apply each reg loss function
                        # normalize targets
                        imsize = 224
                        wer = 3
                        reg_targets = (targets[:,:4]+imsize*(wer-1)/2)/(imsize*wer)
                        
                        try:
                            for loss_fn in losses['reg']:
                                loss_comp = loss_fn(reg_out.float(),reg_targets.float()) 
                                if phase == 'train':
                                    loss_comp.backward(retain_graph = True)
                                each_loss.append(round(loss_comp.item()*100000)/100000.0)
                                
                            # apply each cls loss function
                            cls_targets = targets[:,4]
                            for loss_fn in losses['cls']:
                                loss_comp = loss_fn(cls_out.float(),cls_targets.long()) /10.0
                                if phase == 'train':
                                    loss_comp.backward()
                                each_loss.append(round(loss_comp.item()*100000)/100000.0)
                            acc = 0
                            
                            # backpropogate loss and adjust model weights
                            if phase == 'train':
                                #loss.backward()
                                optimizer.step()
            
                        except RuntimeError:
                            print("Some sort of autograd error")
                            
                    # verbose update
                    count += 1
                    total_acc += acc
                    total_loss += sum(each_loss) #loss.item()
                    if count % 100 == 0:
                        print("{} epoch {} batch {} -- Loss so far: {:03f} -- {}".format(phase,epoch,count,total_loss/count,[item for item in each_loss]))
                    if count % 5000 == 0:
                        plot_batch(model,next(iter(dataloaders['train'])),class_dict)
                    
                    # periodically save best checkpoint
                    if count % 5000 == 0:
                        avg_loss = total_loss/count
                        if avg_loss < best_loss:
                            # save a checkpoint
                            PATH = "/home/worklab/Desktop/checkpoints/kitti_localizer_34/resnet34_epoch{}_batch{}.pt".format(epoch,count)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                "metrics": all_metrics
                                }, PATH)
                            
            # report and record metrics at end of epoch
            avg_acc = total_acc/count
            avg_loss = total_loss/count
            if epoch % 1 == 0:
                
                plot_batch(model,next(iter(dataloaders['train'])),class_dict)
                
                print("Epoch {} avg {} loss: {:05f}  acc: {}".format(epoch, phase,avg_loss,avg_acc))
                all_metrics["{}_loss".format(phase)].append(total_loss)
                all_metrics["{}_acc".format(phase)].append(avg_acc)

                if avg_loss < best_loss:
                    # save a checkpoint
                    PATH = "/home/worklab/Desktop/checkpoints/kitti_localizer_34/resnet34_epoch{}_end.pt".format(epoch)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "metrics": all_metrics
                        }, PATH)
                
                torch.cuda.empty_cache()
                
            # stop training when there is no further improvement
            if avg_loss < best_loss:
                epochs_since_improvement = 0
                best_loss = avg_loss
            else:
                epochs_since_improvement +=1
            
            print("{} epochs since last improvement.".format(epochs_since_improvement))
            # if epochs_since_improvement >= patience:
            #     break
                
        return model , all_metrics

def load_model(checkpoint_file,model,optimizer):
    """
    Reloads a checkpoint, loading the model and optimizer state_dicts and 
    setting the start epoch
    """
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    all_metrics = checkpoint['metrics']

    return model,optimizer,epoch,all_metrics

class Box_Loss(nn.Module):        
    def __init__(self):
        super(Box_Loss,self).__init__()
        
    def forward(self,output,target,epsilon = 1e-07):
        """ Compute the bbox iou loss for target vs output using tensors to preserve
        gradients for efficient backpropogation"""
        
        # minx miny maxx maxy
        minx,_ = torch.max(torch.cat((output[:,0].unsqueeze(1),target[:,0].unsqueeze(1)),1),1)
        miny,_ = torch.max(torch.cat((output[:,1].unsqueeze(1),target[:,1].unsqueeze(1)),1),1)
        maxx,_ = torch.min(torch.cat((output[:,2].unsqueeze(1),target[:,2].unsqueeze(1)),1),1)
        maxy,_ = torch.min(torch.cat((output[:,3].unsqueeze(1),target[:,3].unsqueeze(1)),1),1)

        zeros = torch.zeros(minx.shape).unsqueeze(1).to(device)
        delx,_ = torch.max(torch.cat(((maxx-minx).unsqueeze(1),zeros),1),1)
        dely,_ = torch.max(torch.cat(((maxy-miny).unsqueeze(1),zeros),1),1)
        intersection = torch.mul(delx,dely)
        a1 = torch.mul(output[:,2]-output[:,0],output[:,3]-output[:,1])
        a2 = torch.mul(target[:,2]-target[:,0],target[:,3]-target[:,1])
        #a1,_ = torch.max(torch.cat((a1.unsqueeze(1),zeros),1),1)
        #a2,_ = torch.max(torch.cat((a2.unsqueeze(1),zeros),1),1)
        union = a1 + a2 - intersection 
        iou = intersection / (union + epsilon)
        #iou = torch.clamp(iou,0)
        return 1- iou.sum()/(len(iou)+epsilon)
  
def plot_batch(model,batch,class_dict):
    """
    Given a batch and corresponding labels, plots both model predictions
    and ground-truth
    model - Localize_Net() object
    batch - batch from loader loading Detrac_Localize_Dataset() data
    class-dict - dict for converting between integer and string class labels
    """
    input = batch[0]
    label = batch[1]
    cls_label = label[:,4]
    reg_label = label[:,:4]
    cls_output, reg_output = model(input)
    
    _,cls_preds = torch.max(cls_output,1)
    batch = input.data.cpu().numpy()
    bboxes = reg_output.data.cpu().numpy()
    
    # define figure subplot grid
    batch_size = len(cls_label)
    row_size = min(batch_size,8)
    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
    
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        
        # get image
        im   = batch[i].transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        im   = std * im + mean
        im   = np.clip(im, 0, 1)
        
        # get predictions
        cls_pred = cls_preds[i].item()
        bbox = bboxes[i]
        
        # get ground truths
        cls_true = cls_label[i].item()
        reg_true = reg_label[i]
        
        wer = 3
        imsize = 224
        
        # convert to normalized coords
        reg_true = (reg_true+imsize*(wer-1)/2)/(imsize*wer)
        # convert to im coords
        reg_true = (reg_true* 224*wer - 224*(wer-1)/2).int()


        
        
        # transform bbox coords back into im pixel coords
        bbox = (bbox* 224*wer - 224*(wer-1)/2).astype(int)
        # plot pred bbox
        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)
       
        # plot ground truth bbox
        im = cv2.rectangle(im,(reg_true[0],reg_true[1]),(reg_true[2],reg_true[3]),(0.6,0.1,0.9),2)

        im = im.get()
        
        # title with class preds and gt
        label = "{} -> ({})".format(class_dict[cls_pred],class_dict[cls_true])

        if batch_size <= 8:
            axs[i].imshow(im)
            axs[i].set_title(label)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i//row_size,i%row_size].imshow(im)
            axs[i//row_size,i%row_size].set_title(label)
            axs[i//row_size,i%row_size].set_xticks([])
            axs[i//row_size,i%row_size].set_yticks([])
        plt.pause(.001)    
    #plt.close()

def move_dual_checkpoint_to_cpu(model,optimizer,checkpoint):
    model,optimizer,epoch,all_metrics = load_model(checkpoint_file, model, optimizer)
    model = nn.DataParallel(model,device_ids = [0])
    model = model.to(device)
    
    new_state_dict = {}
    for key in model.state_dict():
        new_state_dict[key.split("module.")[-1]] = model.state_dict()[key]
    
    new_checkpoint = checkpoint.split("resnet")[0] + "cpu_resnet34_epoch{}.pt".format(epoch)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': new_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        "metrics": all_metrics
        }, new_checkpoint)
    

class_dict = {
             'DontCare':8,
             'Van':0,
             'Cyclist':1,
             'Pedestrian':2,
             'Car':3,
             'Misc':4,
             'Truck':5,
             'Tram':6,
             'Person':7,
             'Background':9,
             
             8:'DontCare',
             0:'Van',
             1:'Cyclist',
             2:'Pedestrian',
             3:'Car',
             4:'Misc',
             5:'Truck',
             6:'Tram',
             7:'Person',
             9:'Background'
             }

#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    
    #checkpoint_file = "/home/worklab/Desktop/checkpoints/detrac_localizer_retrain2/try_this_one.pt"
    #checkpoint_file = "/home/worklab/Desktop/checkpoints/detrac_localizer_retrain3/resnet18_epoch20_end.pt"
    #checkpoint_file =  "/home/worklab/Desktop/checkpoints/kitti_localizer_34/resnet34_epoch18_save1.pt"
    #checkpoint_file = "/home/worklab/Desktop/checkpoints/kitti_localizer_34/resnet50_epoch19_save.pt"
    checkpoint_file = None
    checkpoint_file = "/home/worklab/Desktop/checkpoints/kitti_localizer_34/resnet34_epoch137_save.pt"
    patience = 4

    train_image_dir =    "/home/worklab/Desktop/KITTI/data_tracking_image_2/training/image_02"  
    train_label_dir =   "/home/worklab/Desktop/KITTI/data_tracking_label_2/training/label_02"
    train_calib_dir = "/home/worklab/Desktop/KITTI/data_tracking_calib/training/calib"
    
    #label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
    #train_image_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\Tracks"
    #test_image_dir =  "C:\\Users\\derek\\Desktop\\UA Detrac\\Tracks"
    
    # 1. CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        MULTI = True
    else:
        MULTI = False
    torch.cuda.empty_cache()   
    
    # 2. load model
    try:
        model
    except:
        model = ResNet_Localizer()
        if MULTI:
            model = nn.DataParallel(model,device_ids = [0,1])
    model = model.to(device)
    print("Loaded model.")
    
    
    # 3. create training params
    params = {'batch_size' : 64,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True
              }
    
    # 4. create dataloaders
    try:   
        len(train_data)
        len(test_data)
    except:
        detrac_ims = "/home/worklab/Desktop/detrac/DETRAC-all-data"
        detrac_labels = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
        #train_data = Localize_Dataset(detrac_ims,detrac_labels)
        train_data = Localization_Dataset(train_image_dir, train_label_dir,train_calib_dir)
        test_data =  Localization_Dataset(train_image_dir,train_label_dir,train_calib_dir,data_holdout = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        
    trainloader = data.DataLoader(train_data, **params)
    testloader = data.DataLoader(test_data, **params)
    
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    print("Got dataloaders. {},{}".format(datasizes['train'],datasizes['val']))
    
    # 5. define stochastic gradient descent optimizer    
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum = 0.3)
    
    # 6. decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = "min", factor = 0.3,patience = patience,verbose = True) 
    
    # 7. define start epoch for consistent labeling if checkpoint is reloaded
    start_epoch = -1
    all_metrics = None

    # 8. if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        model,optimizer,start_epoch,all_metrics = load_model(checkpoint_file, model, optimizer)
        #model,_,start_epoch = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
        print("Checkpoint loaded.")
     
    # 9. define losses
    losses = {"cls": [nn.CrossEntropyLoss()],
              "reg": [nn.MSELoss(), Box_Loss(),]
              }
    
    # losses = {"cls": [],
    #           "reg": [Box_Loss()]
    #           }
    
    if False:    
    # train model
        print("Beginning training.")
        model,all_metrics = train_model(model,
                            optimizer, 
                            exp_lr_scheduler,
                            losses,
                            dataloaders,
                            device,
                            patience = patience,
                            start_epoch = start_epoch+1,
                            all_metrics = all_metrics)
        
    
    
    
    
    
    if True:
        model.eval()
        error_list = []
        acc_list = []
        for i in range(0,500):
            print("On batch {}".format(i))
            i += 1
            
            data, label = next(iter(trainloader))
            wer = 3
            cls_out,reg_out = model(data.to(device))
            reg_out = (reg_out* 224*wer - 224*(wer-1)/2).int()
    
            pred = reg_out.data.cpu()
            
            # convert pred and labels to xysr  
            new_pred = torch.zeros(pred.shape)
            new_pred[:,0] = (pred[:,0] + pred[:,2]) / 2.0
            new_pred[:,1] = (pred[:,1] + pred[:,3]) / 2.0
            new_pred[:,2] = (pred[:,2] - pred[:,0])
            new_pred[:,3] = (pred[:,3] - pred[:,1]) / new_pred[:,2]
            
            new_label = torch.zeros(pred.shape)
            new_label[:,0] = (label[:,0] + label[:,2]) / 2.0
            new_label[:,1] = (label[:,1] + label[:,3]) / 2.0
            new_label[:,2] = (label[:,2] - label[:,0])
            new_label[:,3] = (label[:,3] - label[:,1]) / label[:,2]
       
            error = torch.mean((new_label - new_pred),dim = 0)
            acc = iou(new_label.double(),new_pred.double())
            
            if sum(error) == np.nan or sum(error) == np.inf:
                continue
                print("Nan")
            error_list.append(error)
            acc_list.append(acc)
        mean = torch.stack(error_list).mean(dim = 0)
        
        covariance = torch.zeros((4,4))
        for vec in error_list:
            covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
        covariance = covariance / (len(error_list) - 1)
        
        print("Localizer results: ")
        print("Average IOU: {}".format(sum(acc_list)/len(acc_list)))
        print("Mean Error: {}".format(mean))
        print("Error Covariance: {}".format(covariance))
        
    #benchmark speed
    if False:
        
        model.eval()
        #batch_sizes = []
        #fps = []
        #batch_time = []
        
        for batch_size in range (100,300,8):
            params = {'batch_size' : batch_size,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True
              }
            trainloader = data.DataLoader(train_data, **params)
            
            total_time = 0
            for i in range(100):
                batch = next(iter(trainloader))[0].to(device)
                
                start = time.time()
                out = model(batch)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                total_time += elapsed
            
            print("{} ims in {} sec, {} fps, {} per batch".format(100*batch_size,total_time, 100*batch_size/total_time,total_time/100))
       
        batch_sizes.append(batch_size)
        fps.append(100*batch_size/total_time)
        batch_time.append(total_time/100)
        
        torch.cuda.empty_cache()
        
        plt.figure()
        plt.plot(batch_sizes,)