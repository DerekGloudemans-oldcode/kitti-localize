""" 
This file contains utilities for loading and showing tracks from the KITTI dataset 
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F

import cv2
import PIL
from PIL import Image
from math import cos,sin
from scipy.signal import savgol_filter

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

class Track_Dataset(data.Dataset):
    """
    Creates an object for referencing the KITTI object tracking dataset (training set)
    """
    
    def __init__(self, image_dir, label_dir,data_holdout = [18,19,20],n = 7 , mode = "xysr"):
        """ initializes object. By default, the first track (cur_track = 0) is loaded 
        such that next(object) will pull next frame from the first track"""

        # stores files for each set of images and each label
        dir_list = next(os.walk(image_dir))[1]
        self.track_list = [os.path.join(image_dir,item) for item in dir_list]
        self.label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
        self.track_list.sort()
        self.label_list.sort()
        
        # for rollout
        self.n = n
        
        # final data storage list
        self.all_data = []
        self.all_labels = []
        self.frame_objs = {}
        
        # separate each object into its own data point
        for i in range(0,len(self.track_list)):
            if i in data_holdout:
                continue
            
            else:
                objs = {}    
                frames = [os.path.join(self.track_list[i],item) for item in os.listdir(self.track_list[i])]
                frames.sort()
                labels = self.parse_label_file(i)
                
                for j in range(0,len(frames)):
                    if len(labels[j]) > 0: # there is at least one object in this frame
                        self.frame_objs[frames[j]] = []
                        for obj in labels[j]:
                            if obj['class'] not in ['Pedestrian', 'Cyclist', 'DontCare']:
                                # add obj id to dictionary if this is the first frame its in
                                # first list holds bboxes, second holds frames
                                if obj["id"] not in objs:
                                    objs[obj["id"]] = [[],[],[],[]]
                                  
                                
                                bbox = obj['bbox2d']
                                new_bbox = np.zeros(4)
                                # covnert to xysr
                                if mode == "xysr":
                                    new_bbox[0] = (bbox[2] + bbox[0])/2.0
                                    new_bbox[1] = (bbox[3] + bbox[1])/2.0
                                    new_bbox[2] = (bbox[2] - bbox[0])
                                    new_bbox[3] = (bbox[3] - bbox[1])/new_bbox[2]
                                elif mode == "xywh":
                                    new_bbox[0] = (bbox[2] + bbox[0])/2.0
                                    new_bbox[1] = (bbox[3] + bbox[1])/2.0
                                    new_bbox[2] = (bbox[2] - bbox[0])
                                    new_bbox[3] = (bbox[3] - bbox[1])
                                
                                objs[obj['id']][0].append(new_bbox)
                                objs[obj['id']][1].append(frames[j])
                                objs[obj['id']][2].append(obj['truncation'])
                                objs[obj['id']][3].append(obj['occlusion'])
                                self.frame_objs[frames[j]].append(new_bbox)
                
                for key in objs:
                    obj = objs[key]
                    labels = (np.array(obj[0]),np.array(obj[2]),np.array(obj[3]))
                    data = obj[1]
                    if len(data) >= self.n:
                        self.all_data.append(data)
                        self.all_labels.append(labels)
                    
                del objs
    
        # estimate speed 
        with_speed = []
        for (bboxes,truncation,occlusion) in self.all_labels:     
            speeds = np.zeros(bboxes.shape)  
            speeds[:len(speeds)-1,:] = bboxes[1:,:] - bboxes[:len(bboxes)-1,:]
            speeds[-1,:] = speeds[-2,:]
            #plt.figure()
            #plt.plot(speeds[:,0])
            try:
                speeds = savgol_filter(speeds,5,2,axis = 0)
            except:
                print(speeds.shape)
                print(bboxes.shape)
            #plt.plot(speeds[:,0])
            #plt.legend(["Unsmoothed","Smoothed"])
            #plt.show()
            combined = np.concatenate((bboxes,speeds),axis = 1)
            with_speed.append((combined,truncation,occlusion))
        self.all_labels = with_speed
             
            
    def __len__(self):
        """ return number of objects"""
        return len(self.all_data)
    
    def parse_label_file(self,idx):
        """parse label text file into a list of numpy arrays, one for each frame"""
        f = open(self.label_list[idx])
        line_list = []
        for line in f:
            line = line.split()
            line_list.append(line)
            
        # each line corresponds to one detection
        det_dict_list = []  
        for line in line_list:
            # det_dict holds info on one detection
            det_dict = {}
            det_dict['frame']      = int(line[0])
            det_dict['id']         = int(line[1])
            det_dict['class']      = str(line[2])
            det_dict['truncation'] = int(line[3])
            det_dict['occlusion']  = int(line[4])
            det_dict['alpha']      = float(line[5]) # obs angle relative to straight in front of camera
            x_min = int(round(float(line[6])))
            y_min = int(round(float(line[7])))
            x_max = int(round(float(line[8])))
            y_max = int(round(float(line[9])))
            det_dict['bbox2d']     = np.array([x_min,y_min,x_max,y_max])
            length = float(line[12])
            width = float(line[11])
            height = float(line[10])
            det_dict['dim'] = np.array([length,width,height])
            x_pos = float(line[13])
            y_pos = float(line[14])
            z_pos = float(line[15])
            det_dict['pos'] = np.array([x_pos,y_pos,z_pos])
            det_dict['rot_y'] = float(line[16])
            det_dict_list.append(det_dict)
        
        # pack all detections for a frame into one list
        label_list = []
        idx = 0
        frame_det_list = []
        for det in det_dict_list:
            assigned = False
            while assigned == False:
                if det['frame'] == idx:
                    frame_det_list.append(det)
                    assigned = True
                # no more detections from frame idx, advance to next
                else:
                    label_list.append(frame_det_list)
                    frame_det_list = []
                    idx = idx + 1
        label_list.append(frame_det_list) # append last frame detections
        return label_list

    def parse_calib_file(self,idx):
        """parse calib file to get  camera projection matrix"""
        f = open(self.calib_list[idx])
        line_list = []
        for line in f:
            line = line.split()
            line_list.append(line)
        line = line_list[2] # get line corresponding to left color camera
        vals = np.zeros([12])
        for i in range(0,12):
            vals[i] = float(line[i+1])
        self.calib = vals.reshape((3,4))
        
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
    
        # load image and get label        
        (data,truncation,occlusion) = self.all_labels[index]
        
        # if track is too short, just use the next index instead
        while len(data) <= self.n:
            index = (index + 1) % len(self.all_labels)
            data = self.all_labels[index]
        
        start = np.random.randint(0,len(data)-self.n)
        data = data[start:start+self.n,:]
        truncation = truncation[start:start+self.n]
        occlusion = occlusion[start:start+self.n]
        
        ims = self.all_data[index]
        ims = ims[start:start+self.n]
        
        return data, ims, truncation, occlusion
     
        
        

def get_coords_3d(det_dict,P):
    """ returns the pixel-space coordinates of an object's 3d bounding box
        computed from the label and the camera parameters matrix
        for the idx object in the current frame
        det_dict - object representing one detection
        P - camera calibration matrix
        bbox3d - 8x2 numpy array with x,y coords for ________ """     
    # create matrix of bbox coords in physical space 

    l = det_dict['dim'][0]
    w = det_dict['dim'][1]
    h = det_dict['dim'][2]
    x_pos = det_dict['pos'][0]
    y_pos = det_dict['pos'][1]
    z_pos = det_dict['pos'][2]
    ry = det_dict['rot_y']
    cls = det_dict['class']
        
        
    # in absolute space (meters relative to obj center)
    obj_coord_array = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],
                                [0,0,0,0,-h,-h,-h,-h],
                                [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]])
    
    # apply object-centered rotation here
    R = np.array([[cos(ry),0,sin(ry)],[0,1,0],[-sin(ry),0,cos(ry)]])
    rotated_corners = np.matmul(R,obj_coord_array)
    
    rotated_corners[0,:] += x_pos
    rotated_corners[1,:] += y_pos
    rotated_corners[2,:] += z_pos
    
    # transform with calibration matrix
    # add 4th row for matrix multiplication
    zeros = np.zeros([1,np.size(rotated_corners,1)])
    rotated_corners = np.concatenate((rotated_corners,zeros),0)

    
    pts_2d = np.matmul(P,rotated_corners)
    pts_2d[0,:] = pts_2d[0,:] / pts_2d[2,:]        
    pts_2d[1,:] = pts_2d[1,:] / pts_2d[2,:] 
    
    # apply camera space rotation here?
    return pts_2d[:2,:] ,pts_2d[2,:], rotated_corners

    


 ################################# Tester Code ################################    
if __name__ == "__main__":    
#    train_im_dir =    "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
#    train_lab_dir =   "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
#    train_calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\data_tracking_calib(1)\\training\\calib"
    
    # worklab GTX 1080 workstation
    # train_im_dir =    "/home/worklab/Desktop/KITTI/data_tracking_image_2/training/image_02"  
    # train_lab_dir =   "/home/worklab/Desktop/KITTI/data_tracking_label_2/training/label_02"
    # train_calib_dir = "/home/worklab/Desktop/KITTI/data_tracking_calib/training/calib"
    
    ## worklab Quadro workstation
    train_im_dir =    "/home/worklab/Data/cv/KITTI/data_tracking_image_2/training/image_02" 
    train_lab_dir =   "/home/worklab/Data/cv/KITTI/data_tracking_label_2/training/label_02"
    train_calib_dir = "/home/worklab/Data/cv/KITTI/data_tracking_calib/training/calib"
    
    test = Track_Dataset(train_im_dir,train_lab_dir)
    # test.load_track(10)
    
    
    
    # im,label = next(test)
    # frame = 0
    # while im:
        
    #     cv_im = pil_to_cv(im)
    #     if True:
    #         cv_im = plot_bboxes_3d(cv_im,label,test.calib)
    #         #cv_im = plot_bboxes_2d(cv_im,label)
    #     cv2.imshow("Frame",cv_im)
    #     key = cv2.waitKey(1) & 0xff
    #     #time.sleep(1/30.0)
    #     if False:
    #         cv2.imwrite("temp{}.png".format(frame),cv_im)
    #     frame +=1
    #     if key == ord('q'):
    #         break
        
    #     # load next frame
    #     im,label = next(test)
    
        
    # cv2.destroyAllWindows()
    
    idx = np.random.randint(0,len(test))
    print(test[idx])
    
    
            