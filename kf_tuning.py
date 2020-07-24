#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:20:41 2020

@author: worklab
"""

import torch
import numpy as np
import time
import random
import _pickle as pickle
import matplotlib.pyplot as plt
import cv2 

random.seed  = 0

#from detrac_files.detrac_tracking_dataset import Track_Dataset
from kitti_tracking_dataset import Track_Dataset
from torch_kf_dual import Torch_KF
from torch.utils.data import DataLoader



from PIL import Image
from torchvision.transforms import functional as F
from torchvision.ops import roi_align

from kitti_train_localizer import ResNet_Localizer, load_model, class_dict, plot_batch
plt.style.use("seaborn")

# need to make the set of matrices that are optimized over variable
# need to make variable rollout and pre-rollout lengths
# need to have variable learning rate
# need to have IoU metric of evaluation
# need to report time metric for each round of evaluations
# need to have variable batch size 

       
def test_outputs(bboxes,crops):
    """
    Description
    -----------
    Generates a plot of the bounding box predictions output by the localizer so
    performance of this component can be visualized
    
    Parameters
    ----------
    bboxes - tensor [n,4] 
        bounding boxes output for each crop by localizer network
    crops - tensor [n,3,width,height] (here width and height are both 224)
    """
    
    # define figure subplot grid
    batch_size = len(crops)
    row_size = min(batch_size,8)
    
    for i in range(0,len(crops)):    
        # get image
        im   = crops[i].data.cpu().numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        im   = std * im + mean
        im   = np.clip(im, 0, 1)
        
        # get predictions
        bbox = bboxes[i].data.cpu().numpy()
        
        wer = 3
        imsize = 224
        
        # transform bbox coords back into im pixel coords
        bbox = (bbox* imsize*wer - imsize*(wer-1)/2).astype(int)
        # plot pred bbox
        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)
        im = im.get()

        plt.imshow(im)
        plt.pause(1)
        
        
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


def score_tracker(tracker,batch,n_pre,n_post):
    """
    Evaluates a tracker by inially updating it with n_pre frames of object locations,
    then rolls out predictions for n_post frames and evaluates iou with ground
    truths for these frames. Returns average iou for all objects and rollout frames

    Parameters
    ----------
    tracker : Torch_KF object
        a Kalman Filter tracker
    batch : Float Tensor - [batch_size,(n_pre + n_post),4] 
        bounding boxes for several objects  for several frames
    n_pre : int
        number of frames used to initially update the tracker
    n_post : int
        number of frames used to evaluate the tracker

    Returns
    -------
    score : average bbox iou for all objects evaluated over n_post rollout predictions
    """
    obj_ids = [i for i in range(len(batch))]
    
    # don't want to always use first frame in track to initialize, so randomly pick an index
    first = np.random.randint(0,batch.shape[1]-2)
    
    tracker.add(batch[:,first,:8],obj_ids)
        
    running_mean_iou = 0
    # roll out a frame
    tracker.predict()
    
    # get predicted object locations
    objs = tracker.objs()
    objs = [objs[key] for key in objs]
    pred = torch.from_numpy(np.array(objs)).double()[:,:4]
        
        
    # evaluate
    score = iou(batch[:,first+1,:4],pred)
    #val = abs_err(batch[:,frame,:],pred)
        
    return score


def plot_states(ap_states,
                ap_covs,
                loc_meas,
                apst_states,
                apst_covs,
                gts,
                save_num = None
                ):
    
   
    
    #convert list into numpy array
    ap_states =   torch.stack(ap_states)
    ap_covs =     torch.stack(ap_covs)
    loc_meas =    torch.stack(loc_meas)
    apst_states = torch.stack(apst_states)
    apst_covs =   torch.stack(apst_covs)
    gts =         torch.stack(gts)
    titles = ["X coordinate", "Y coordinate", "Scale", "Ratio", "X dot", "Y dot", "S dot", "R dot"]
    
    # format covariances - want each 
    covs = torch.empty(ap_covs.shape[0]+apst_covs.shape[0],ap_covs.shape[1])
    covs[1::2,:]  = ap_covs
    covs[::2,:] = apst_covs
    means = torch.empty(ap_states.shape[0]+apst_states.shape[0],ap_states.shape[1])
    means[1::2,:] = ap_states
    means[::2,:] = apst_states
    
    xvals = [i/2 for i in range(len(covs))]
    ap_xvals = [i for i in range(1,len(ap_states)+1)]
    
    fig, axs = plt.subplots(int(len(gts[0]))//2,2,constrained_layout=True,figsize = (20,20))
    
    for i in range(len(gts[0])):
        legend = ["apriori","aposteriori","true","covariance"]
        
        # plot apriori state
        axs[i//2,i%2].plot(ap_xvals,ap_states[:,i],"--",linewidth = 3, color = (0.3,0.2,0.5))
        
        # plot measurement
        if i in range(len(loc_meas[0])):
            axs[i//2,i%2].plot(ap_xvals,loc_meas[:,i],".", markersize=15,color = (0.3,0.2,0.5))
            legend = ["apriori","measurement","aposteriori","true","covariance"]
        
        if i == 6:
            axs[i//2,i%2].set_ylim([-30,30])
            
        if i == 3:
           axs[i//2,i%2].set_ylim([0,2]) 
        
        if i == 7:
             axs[i//2,i%2].set_ylim([-0.3,0.3]) 
             
        # plot aposteriori state
        axs[i//2,i%2].plot(apst_states[:,i],"-", color = (0.4,0.1,0.5))
        
        # plot ground truth
        axs[i//2,i%2].plot(gts[:,i],"-", color = (0.2,0.7,0.3))
        
        # plot covariance
        axs[i//2,i%2].fill_between(xvals,means[:,i]-covs[:,i],covs[:,i]+means[:,i],color = (0.4,0.1,0.5,0.2))
        
        # set plot settings
        axs[i//2,i%2].legend(legend)
        axs[i//2,i%2].set_title(titles[i])
        # axs[i//2,i%2].set_xlim([0,len(loc_meas)])
        
    plt.pause(3)
    if save_num is not None:
        plt.savefig("states_{}.png".format(save_num))

##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################


if __name__ == "__main__":

    # define parameters
    
    b         = 50 # batch size
    n_pre     = 1      # number of frames used to initially tune tracker
    n_post    = 15     # number of frame rollouts used to evaluate tracker
    
    
    localizer = ResNet_Localizer()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache() 
        
    #resnet_checkpoint = "/home/worklab/Desktop/checkpoints/kitti_localizer_34/cpu_resnet34_epoch137.pt"
    #resnet_checkpoint = "/home/worklab/Desktop/checkpoints/kitti_localizer_34/cpu_resnet34_epoch18.pt"
    resnet_checkpoint = "/home/worklab/Data/cv/checkpoints/cpu_kitti_resnet34_epoch70.pt"

    cp = torch.load(resnet_checkpoint)
    localizer.load_state_dict(cp['model_state_dict']) 
    localizer = localizer.to(device)
    localizer.eval()
    
    try:
        loader
    except:
        # worklab GTX 1080 workstation
        # train_im_dir =    "/home/worklab/Desktop/KITTI/data_tracking_image_2/training/image_02"  
        # train_lab_dir =   "/home/worklab/Desktop/KITTI/data_tracking_label_2/training/label_02"
        # train_calib_dir = "/home/worklab/Desktop/KITTI/data_tracking_calib/training/calib"
        
        ## worklab Quadro workstation
        train_im_dir =    "/home/worklab/Data/cv/KITTI/data_tracking_image_2/training/image_02" 
        train_lab_dir =   "/home/worklab/Data/cv/KITTI/data_tracking_label_2/training/label_02"
        train_calib_dir = "/home/worklab/Data/cv/KITTI/data_tracking_calib/training/calib"
        
        dataset = Track_Dataset(train_im_dir,train_lab_dir,n = (n_pre + n_post+1))
        
        # 3. create training params
        params = {'batch_size' : b,
                  'shuffle'    : True,
                  'num_workers': 0,
                  'drop_last'  : True
                  }
        
        # returns batch_size x (n_pre + n_post) x 4 tensor
        loader = DataLoader(dataset, **params)
    
    
    # # # create initial values for each matrix starting with unfit params
    # tracker = Torch_KF("cpu",INIT = None)
    # kf_params = {
    #         "P":tracker.P0.squeeze(0),
    #         "Q":tracker.Q.squeeze(0),
    #         "R":tracker.R.squeeze(0),
    #         "F":tracker.F,
    #         "H":tracker.H,
    #         "mu_Q":tracker.mu_Q,
    #         "mu_R":tracker.mu_R,
    #         "mu_R2":tracker.mu_R2,
    #         "R2":tracker.R2
    #         }
        
    with open("kitti_velocity8_unfitted.cpkl","rb") as f:
               kf_params = pickle.load(f)
               kf_params["mu_Q"] = torch.zeros(8)
        
    # fit Q and mu_Q
    if False:
        error_vectors = []
        scores = []
        for iteration in range(20000):
            
            # grab batch
            batch, ims = next(iter(loader))
            
            # initialize tracker
            tracker = Torch_KF("cpu",INIT = kf_params, ADD_MEAN_Q = False)
        
            obj_ids = [i for i in range(len(batch))]
            
            # don't want to always use first frame in track to initialize, so randomly pick an index
            first = np.random.randint(0,batch.shape[1]-2)
            
            tracker.add(batch[:,first,:8],obj_ids)
            
            # roll out a frame
            tracker.predict()
            
            # get predicted object locations
            objs = tracker.objs()
            objs = [objs[key] for key in objs]
            pred = torch.from_numpy(np.array(objs)).double()
            
            # get ground truths
            gt = batch[:,first+1,:8]
            error = gt - pred
            error_vectors.append(error)
            
            # get ious
            scores.append(score_tracker(tracker,batch,n_pre,1))
            
            print("Finished iteration {}".format(iteration))
            
        # summary metrics    
        error_vectors = torch.cat(error_vectors,dim = 0)
        mean = torch.mean(error_vectors, dim = 0)
        
        covariance = torch.zeros((8,8))
        for vec in error_vectors:
            covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
        
        covariance = covariance / error_vectors.shape[0]
        kf_params["mu_Q"] = mean
        kf_params["Q"] = covariance
        
        with open("kitti_velocity8_Q2.cpkl","wb") as f:
              pickle.dump(kf_params,f)
        
        print("---------- Model 1-step errors ----------")
        print("Average 1-step IOU: {}".format(sum(scores)/len(scores)))
        print("Mean 1-step state error: {}".format(mean))
        print("1-step covariance: {}".format(covariance))
        
        
    # fit R and mu_R
    if False:
        
        with open("kitti_velocity8_Q2.cpkl",'rb') as f:
                kf_params = pickle.load(f) 
        
        for ber in [2.2]:
            skewed_iou = []
            localizer_iou = []
            meas_errors = []
            
            for iteration in range(500):
                batch,ims = next(iter(loader))
                frame_idx = 0
                gt = batch[:,frame_idx,:4]
                
                # get starting error
                degradation = np.array([2,2,4,0.01]) *0 # should roughly equal localizer error covariance
                skew = np.random.normal(0,degradation,(len(batch),4))
                gt_skew = gt + skew
                skewed_iou.append(iou(gt_skew,gt))
    
                
                
                
                # ims are collated by frame,then batch index
                relevant_ims = ims[frame_idx]
                frames =[]
                for idx,item in enumerate(relevant_ims):
                    with Image.open(item) as im:
                           im = F.to_tensor(im)
                           frame = F.normalize(im,mean=[0.3721, 0.3880, 0.3763],
                                 std=[0.0555, 0.0584, 0.0658])
                           #correct smaller frames
                           if frame.shape[1] < 375:
                              new_frame = torch.zeros([3,375,frame.shape[2]])
                              new_frame[:,:frame.shape[1],:] = frame
                              frame = new_frame
                           if frame.shape[2] < 1242:
                              new_frame = torch.zeros([3,375,1242])
                              new_frame[:,:,:frame.shape[2]] = frame
                              frame = new_frame   
                           
                           MASK = False 
                           if MASK:
                               
                               other_objs = dataset.frame_objs[item]
                               
                               # create copy of frame
                               frame_copy = frame.clone()
                               
                               # mask each other object in frame
                               for obj in other_objs:
                                   xmin = (obj[0] - obj[2] / 2.0).astype(int)
                                   ymin = (obj[1] - obj[2]*obj[3] / 2.0).astype(int)
                                   xmax = (obj[0] + obj[2] / 2.0).astype(int)
                                   ymax = (obj[1] + obj[2]*obj[3] / 2.0).astype(int)
                                   
                                   frame[:,ymin:ymax,xmin:xmax] = 0
                               
                               # restore gt_skew pixels
                               o = gt_skew[idx]
                               xmin = (o[0] - o[2] / 2.0).int()
                               ymin = (o[1] - o[2]*obj[3] / 2.0).int()
                               xmax = (o[0] + o[2] / 2.0).int()
                               ymax = (o[1] + o[2]*obj[3] / 2.0).int()
                               #frame[:,ymin:ymax,xmin:xmax] = frame_copy[:,ymin:ymax,xmin:xmax]
                               #plt.imshow(frame.transpose(2,0).transpose(0,1))
                               #plt.pause(5)
                                                                
                           frames.append(frame)
                frames = torch.stack(frames).to(device)
                
                # crop image
                boxes = gt_skew
                
                # convert xysr boxes into xmin xmax ymin ymax
                # first row of zeros is batch index (batch is size 0) for ROI align
                new_boxes = np.zeros([len(boxes),5]) 
        
                # use either s or s x r for both dimensions, whichever is larger,so crop is square
                #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
                box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
                    
                #expand box slightly
                #ber = 2.15
                box_scales = box_scales * ber# box expansion ratio
                
                new_boxes[:,1] = boxes[:,0] - box_scales/2
                new_boxes[:,3] = boxes[:,0] + box_scales/2 
                new_boxes[:,2] = boxes[:,1] - box_scales/2 
                new_boxes[:,4] = boxes[:,1] + box_scales/2 
                for i in range(len(new_boxes)):
                    new_boxes[i,0] = i # set image index for each
                    
                torch_boxes = torch.from_numpy(new_boxes).float().to(device)
                
                # crop using roi align
                crops = roi_align(frames,torch_boxes,(224,224))
                
                _,reg_out = localizer(crops)
                torch.cuda.synchronize()
        
                # 5b. convert to global image coordinates 
                    
                # these detections are relative to crops - convert to global image coords
                wer = 3 # window expansion ratio, was set during training
                
                detections = (reg_out* 224*wer - 224*(wer-1)/2)
                detections = detections.data.cpu()
                
                # plot outputs
                if True and iteration % 100 == 0:
                    batch_size = 32
                    row_size = 8
                    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
                
                    for i in range(0,batch_size):
                        
                        # get image
                        im   = crops[i].cpu().numpy().transpose((1,2,0))
                        mean = np.array([0.485, 0.456, 0.406])
                        std  = np.array([0.229, 0.224, 0.225])
                        im   = std * im + mean
                        im   = np.clip(im, 0, 1)
                        
                        bbox = detections[i]
                        
                        
                        wer = 3
                        imsize = 224
                        
                        # convert xysr to xyxy
                        reg_true = torch.zeros([len(gt),4])
                        reg_true[:,0] =  gt[:,0] - gt[:,2]/2.0
                        reg_true[:,1] =  gt[:,1] - gt[:,2]*gt[:,3]/2.0
                        reg_true[:,2] =  gt[:,0] + gt[:,2]/2.0
                        reg_true[:,3] =  gt[:,1] + gt[:,2]*gt[:,3]/2.0
                        
                        reg_true[:,0] = reg_true[:,0] - new_boxes[:,1]
                        reg_true[:,1] = reg_true[:,1] - new_boxes[:,2]
                        reg_true[:,2] = reg_true[:,2] - new_boxes[:,1]
                        reg_true[:,3] = reg_true[:,3] - new_boxes[:,2]
                        reg = reg_true[i] * 224/box_scales[i]
                        
                        # plot pred bbox
                        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.9,0.2,0.2),2)
                       
                        # plot ground truth bbox
                        im = cv2.rectangle(im,(reg[0],reg[1]),(reg[2],reg[3]),(0.0,0.8,0.0),2)
                
                        im = im.get()
                        
                        axs[i//row_size,i%row_size].imshow(im)
                        axs[i//row_size,i%row_size].set_xticks([])
                        axs[i//row_size,i%row_size].set_yticks([])
                        plt.pause(.001)    
                    plt.close()
                
                # add in original box offsets and scale outputs by original box scales
                detections[:,0] = detections[:,0]*box_scales/224 + new_boxes[:,1]
                detections[:,2] = detections[:,2]*box_scales/224 + new_boxes[:,1]
                detections[:,1] = detections[:,1]*box_scales/224 + new_boxes[:,2]
                detections[:,3] = detections[:,3]*box_scales/224 + new_boxes[:,2]
        
                # convert into xysr form 
                output = np.zeros([len(detections),4])
                output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
                output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
                output[:,2] = (detections[:,2] - detections[:,0])
                output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
                pred = torch.from_numpy(output)
                
                
                
                
                
                # evaluate localizer
                localizer_iou.append(iou(pred,gt))
                error = (gt[:,:4]-pred)
                meas_errors.append(error)
                    
                
                # if iteration % 5 == 0:
                #     print("Finished iteration {}".format(iteration))
                
            meas_errors = torch.stack(meas_errors)
            meas_errors = meas_errors.view(-1,4)
            mean = torch.mean(meas_errors, dim = 0)    
            covariance = torch.zeros((4,4))
            for vec in meas_errors:
                covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
                
            covariance = covariance / meas_errors.shape[0]
            
            kf_params["mu_R"] = mean
            kf_params["R"] = covariance
            
            
            print("---------- Localizer 1-step errors with ber = {}----------".format(ber))
            print("Average starting IOU: {}".format(sum(skewed_iou)/len(skewed_iou)))
            print("Average 1-step IOU: {}".format(sum(localizer_iou)/len(localizer_iou)))
            print("Mean 1-step state error: {}".format(mean))
            print("1-step covariance: {}".format(covariance))
       

        
        with open("kitti_velocity8_QR2.cpkl",'wb') as f:
                pickle.dump(kf_params,f)
        
        
        
        
        
        
        
###########################################################################################################################################################################        

        
    if True:
        with open("kitti_velocity8_QR2.cpkl","rb") as f:
              kf_params = pickle.load(f)        
              #kf_params["P"][[0,1,2,3],[0,1,2,3]] = torch.from_numpy(np.array([1,1,2,0.2])).float()
              #kf_params["R"] 
        skewed_iou = []        # how far off each skewed measurement is during init
        starting_iou = []     # after initialization, how far off are we
        a_priori_iou = {}      # after prediction step, how far off is the state
        localizer_iou = {}     # how far off is the localizer
        a_posteriori_iou = {}  # after updating, how far off is the state
        
        for i in range(n_pre,n_pre+n_post):
            a_priori_iou[i] = []
            localizer_iou[i] = []
            a_posteriori_iou[i] = []
        
        model_errors = []
        meas_errors = []
        
        degradation = np.array([2,2,4,0.01]) *0 # should roughly equal localizer error covariance
        
        for iteration in range(200):
            
            batch,ims = next(iter(loader))
            
            # initialize tracker
            tracker = Torch_KF("cpu",INIT = kf_params, ADD_MEAN_Q = True, ADD_MEAN_R = True)
        
            obj_ids = [i for i in range(len(batch))]
            
            tracker.add(batch[:,0,:8],obj_ids)
            
            # initialize storage
            ap_states = []
            ap_covs = []
            loc_meas = []
            apst_states = []
            apst_covs = []
            gts = []
            
            apst_states.append(tracker.X[0].clone())
            apst_covs.append(torch.diag(tracker.P[0]).clone())
            gts.append(batch[0,n_pre-1,:].clone())
            
            # initialize tracker
            for frame in range(1,n_pre):
                tracker.predict()
                
                # here, rather than updating with ground truth we degrade ground truth by some amount
                measurement = batch[:,frame,:4]
                skew = np.random.normal(0,degradation,(len(batch),4))
                measurement_skewed = measurement + skew
                
                skewed_iou.append(iou(measurement,measurement_skewed))
                tracker.update2(measurement_skewed,obj_ids)
                
            
            # cumulative error so far
            objs = tracker.objs()
            objs = [objs[key] for key in objs]
            starting = torch.from_numpy(np.array(objs)).double()
            starting_iou.append(iou(starting,batch[:,n_pre-1,:]))
            
            
            for frame_idx in range(n_pre,n_pre + n_post):
                gt = batch[:,frame_idx,:]
        
                # get a priori error
                tracker.predict()
                objs = tracker.objs()
                objs = [objs[key] for key in objs]
                a_priori = torch.from_numpy(np.array(objs)).double()
                a_priori_iou[frame_idx].append(iou(a_priori,gt))
            
                ap_states.append(tracker.X[0].clone())
                ap_covs.append(torch.diag(tracker.P[0]).clone())

                # at this point, we have gt, the correct bounding boxes for this frame, 
                # and the tracker states, the estimate of the state for this frame
                # expand state estimate and get localizer prediction
                # shift back into global coordinates
                
                # ims are collated by frame,then batch index
                relevant_ims = ims[frame_idx]
                frames =[]
                for item in relevant_ims:
                    with Image.open(item) as im:
                           im = F.to_tensor(im)
                           frame = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                           #correct smaller frames
                           if frame.shape[1] < 375:
                              new_frame = torch.zeros([3,375,frame.shape[2]])
                              new_frame[:,:frame.shape[1],:] = frame
                              frame = new_frame
                           if frame.shape[2] < 1242:
                              new_frame = torch.zeros([3,375,1242])
                              new_frame[:,:,:frame.shape[2]] = frame
                              frame = new_frame   
                              
                           frames.append(frame)
                frames = torch.stack(frames).to(device)
                
                # crop image
                boxes = a_priori
                
                # convert xysr boxes into xmin xmax ymin ymax
                # first row of zeros is batch index (batch is size 0) for ROI align
                new_boxes = np.zeros([len(boxes),5]) 
        
                # use either s or s x r for both dimensions, whichever is larger,so crop is square
                #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
                box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
                    
                #expand box slightly
                ber = 2.15
                box_scales = box_scales * ber# box expansion ratio
                
                new_boxes[:,1] = boxes[:,0] - box_scales/2
                new_boxes[:,3] = boxes[:,0] + box_scales/2 
                new_boxes[:,2] = boxes[:,1] - box_scales/2 
                new_boxes[:,4] = boxes[:,1] + box_scales/2 
                for i in range(len(new_boxes)):
                    new_boxes[i,0] = i # set image index for each
                    
                torch_boxes = torch.from_numpy(new_boxes).float().to(device)
                
                # crop using roi align
                crops = roi_align(frames,torch_boxes,(224,224))
                
                _,reg_out = localizer(crops)
                torch.cuda.synchronize()
        
                # 5b. convert to global image coordinates 
                    
                # these detections are relative to crops - convert to global image coords
                wer = 3 # window expansion ratio, was set during training
                
                detections = (reg_out* 224*wer - 224*(wer-1)/2)
                detections = detections.data.cpu()
                
                # add in original box offsets and scale outputs by original box scales
                detections[:,0] = detections[:,0]*box_scales/224 + new_boxes[:,1]
                detections[:,2] = detections[:,2]*box_scales/224 + new_boxes[:,1]
                detections[:,1] = detections[:,1]*box_scales/224 + new_boxes[:,2]
                detections[:,3] = detections[:,3]*box_scales/224 + new_boxes[:,2]
        
                # convert into xysr form 
                output = np.zeros([len(detections),4])
                output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
                output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
                output[:,2] = (detections[:,2] - detections[:,0])
                output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
                pred = torch.from_numpy(output)
                
                # evaluate localizer
                localizer_iou[frame_idx].append(iou(pred,gt))
                error = (gt[:,:4]-pred)
                meas_errors.append(error)
                
                loc_meas.append(pred[0].clone())
                
                # evaluate a posteriori estimate
                tracker.update(pred,obj_ids)
                objs = tracker.objs()
                objs = [objs[key] for key in objs]
                a_posteriori = torch.from_numpy(np.array(objs)).double()
                a_posteriori_iou[frame_idx].append(iou(a_posteriori,gt))
                
                apst_states.append(tracker.X[0].clone())
                apst_covs.append(torch.diag(tracker.P[0]).clone())
                
                gts.append(gt[0].clone())
            
            if True and iteration < 10:
                plot_states(ap_states,
                            ap_covs,
                            loc_meas,
                            apst_states,
                            apst_covs,
                            gts,
                            save_num = iteration)
                #break
            else:
                break
                
            if iteration % 50 == 0:
                print("Finished iteration {}".format(iteration))
            
        # meas_errors = torch.stack(meas_errors)
        # meas_errors = meas_errors.view(-1,4)
        # mean = torch.mean(meas_errors, dim = 0)    
        # covariance = torch.zeros((4,4))
        # for vec in meas_errors:
        #     covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
            
        # covariance = covariance / meas_errors.shape[0]
        
        # kf_params["mu_R"] = mean
        # kf_params["R"] = covariance
        
        # with open("kitti_velocity8_QR.cpkl",'wb') as f:
        #       pickle.dump(kf_params,f)
                
        print("------------------Results: --------------------")
       # print("Skewed initialization IOUs: {}".format(sum(skewed_iou)/len(skewed_iou)))
        print("Starting state IOUs: {}".format(sum(starting_iou)/len(starting_iou)))
     
        for key in a_priori_iou.keys():
            print("Frame {}".format(key))
            print("A priori state IOUs: {}".format(sum(a_priori_iou[key])/len(a_priori_iou[key])))
            print("Localizer state IOUs: {}".format(sum(localizer_iou[key])/len(localizer_iou[key])))
            print("A posteriori state IOUs: {}".format(sum(a_posteriori_iou[key])/len(a_posteriori_iou[key])))
