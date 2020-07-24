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

def get_dataset_mean_stdev(im_dir):
    """
    Returns mean and standard deviation for each color channel in the dataset

    Parameters
    ----------
    im_dir : string
        Path to tracking dataset data folders

    Returns
    -------
    mean : torch tensor [3]
        average value for each color channel in the dataset
    std : torch tensor [3]
        standard deviation for each color channel in the images of the dataset

    """
    dir_list = next(os.walk(im_dir))[1]
    track_list = [os.path.join(im_dir,item) for item in dir_list]
    
    color_avgs = []
    
    for track in track_list:
        frames = [os.path.join(track,item) for item in os.listdir(track)]
        for frame in frames:
            im = Image.open(frame)
            
            im = F.to_tensor(im)
            mean = im.mean(dim = 1).mean(dim = 1)
            color_avgs.append(mean)
    
    color_avgs = torch.stack(color_avgs)
    mean = color_avgs.mean(dim = 0)
    std = color_avgs.std(dim = 0)
    
    return mean, std

class Localization_Dataset(data.Dataset):
    """
    Creates an object for referencing the KITTI object tracking dataset (training set)
    """
    
    def __init__(self, image_dir, label_dir,calib_dir,data_holdout = [18,19,20]):
        """ initializes object. By default, the first track (cur_track = 0) is loaded 
        such that next(object) will pull next frame from the first track"""

        # stores files for each set of images and each label
        dir_list = next(os.walk(image_dir))[1]
        self.track_list = [os.path.join(image_dir,item) for item in dir_list]
        self.label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
        self.calib_list = [os.path.join(calib_dir,item) for item in os.listdir(calib_dir)]
        self.track_list.sort()
        self.label_list.sort()
        self.calib_list.sort()
          
        self.im_tf = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness = 0.3,contrast = 0.3,saturation = 0.2)
                        ]),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.1, 5.3), value=(0.3721, 0.3880, 0.3763)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 2.3), value=(0.3721, 0.3880, 0.3763)),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=(0.3721, 0.3880, 0.3763)),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.35), ratio=(0.3, 3.3), value=(0.3721, 0.3880, 0.3763)),
                transforms.RandomErasing(p=0.05, scale=(0.02, 0.6), ratio=(0.3, 3.3), value=(0.3721, 0.3880, 0.3763)),

                transforms.Normalize(mean=[0.3721, 0.3880, 0.3763],
                                 std=[0.0555, 0.0584, 0.0658])
                ])

        # for denormalizing
        self.denorm = transforms.Normalize(mean = [-0.3721/0.0555, -0.3880/0.0584, -0.3763/0.0658],
                                           std = [1/0.0555, 1/0.0584, 1/0.0658])
        
        self.class_dict = {
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
        
        
        # final data storage list
        self.all_data = []
        
        # separate each object into its own data point
        for i in range(0,len(self.track_list)):
            if i in data_holdout:
                continue
            else:
                frames = [os.path.join(self.track_list[i],item) for item in os.listdir(self.track_list[i])]
                frames.sort()
                labels = self.parse_label_file(i)
                calib = self.parse_calib_file(i)
                
                for j in range(0,len(frames)):
                    if len(labels[j]) > 0: # there is at least one object in this frame
                        for obj in labels[j]:
                            if obj['class'] not in ['DontCare', 'Pedestrian','Cyclist']:
                                bbox = obj['bbox2d']
                                if bbox[2]-bbox[0] > 1 and bbox[3] - bbox[1] > 1: #verify not too small such that it will throw an error
                                    self.all_data.append([frames[j],obj,calib,labels[j]])
                    
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
        """ returns item indexed from all frames in all tracks from data
        
        Data Characterization:
            - It is assumed that the bounding box for the object of interest will be roughly correct, and should be expanded by 2x
            Thus, an expansion of between 1.5 and 2.5 will be used
            - A uniform random distribution of object shift will be used to avoid bias
            - If desired, all other objects will be masked 
        """
        
    
        # load image and get label        
        cur = self.all_data[index]
        im = Image.open(cur[0]) # image
        label = cur[1] # bounding box etc for object of interest
        others = cur[3] # the other object bboxes besides object of interest
        
        
        
        # copy so that original coordinates aren't overwritten
        bbox = label["bbox2d"].copy()
        
        bbox_width  = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        # flip sometimes
        if np.random.rand() > 0.5:
            im= F.hflip(im)
            # reverse coords and also switch xmin and xmax
            bbox[[2,0]] = im.size[0] - bbox[[0,2]]
        
        # randomly shift the center of the crop
        shift_scale = 70
        x_shift = np.random.uniform(-im.size[0]/shift_scale,im.size[0]/shift_scale)
        y_shift = np.random.normal(-im.size[1]/shift_scale,im.size[1]/shift_scale)
        
        # temp for troubleshooting
        # x_shift = 0
        # y_shift = 0
        
        # randomly expand the box between 1.25 and 2.75x, again using uniform dist
        # bufferx = np.random.uniform(bbox_width*0.125,bbox_width*0.875)
        # buffery = np.random.uniform(bbox_height*0.125,bbox_height*0.875)
        bufferx = np.random.uniform(-bbox_width*0.25,bbox_width*0.25)
        buffery = np.random.uniform(-bbox_height*0.25,bbox_height*0.25)
        
        # temp for troubleshooting
        # bufferx = 200 
        # buffery = 200
        
        # corners of the cropped image are defined relative to the bbox
        minx = max(0,bbox[0]-bufferx)
        miny = max(0,bbox[1]-buffery)
        maxx = min(im.size[0],bbox[2]+bufferx)
        maxy = min(im.size[1],bbox[3]+buffery)
        
        # the crop is shifted so it is no longer centered on the bbox
        minx = minx + x_shift
        maxx = maxx + x_shift
        miny = miny + y_shift
        maxy = maxy + y_shift
        
        # correct 0-size crops
        if maxx-minx < 1 or maxy-miny < 1:
            minx = max(0,bbox[0])
            miny = max(0,bbox[1])
            maxx = min(im.size[0],bbox[2])
            maxy = min(im.size[1],bbox[3])
        
        im_crop = F.crop(im,miny,minx,maxy-miny,maxx-minx)
        del im 
        
        if im_crop.size[0] == 0 or im_crop.size[1] == 0:
            print("Oh no!")
            print(im_crop.size,minx,miny,maxx,maxy)
            print(bbox)
            print(bufferx,buffery)
            print(x_shift,y_shift)
            print(label['bbox2d'])
            raise Exception
        
        # shift bbox            
        bbox[0] = bbox[0] - minx
        bbox[1] = bbox[1] - miny
        bbox[2] = bbox[2] - minx
        bbox[3] = bbox[3] - miny
        
        # resize image
        orig_size = im_crop.size
        im_crop = F.resize(im_crop, (224,224))
        
        # scale bbox
        bbox[0] = bbox[0] * 224/orig_size[0]
        bbox[2] = bbox[2] * 224/orig_size[0]
        bbox[1] = bbox[1] * 224/orig_size[1]
        bbox[3] = bbox[3] * 224/orig_size[1]
        
        # apply random affine transformation
        y = np.zeros(5)
        y[0:4] = bbox
        y[4] = self.class_dict[label["class"]]
        
        #im_crop,y = self.random_affine_crop(im_crop,y)

        
        
        # convert image and label to tensors
        im_t = self.im_tf(im_crop)
        return im_t, y

    def show(self,index):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """
        
        # detrac 
        # mean = np.array([0.485, 0.456, 0.406])
        # stddev = np.array([0.229, 0.224, 0.225])
        
        # KITTI
        mean = np.array([0.3721,0.3880,0.3763])
        stddev = np.array([0.0555, 0.0584, 0.0658])
        
        im,label = self[index]
        
        im = self.denorm(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        #cv_im = plot_bboxes_2d(cv_im,label,metadata['ignored_regions'])
        cv2.rectangle(cv_im,(int(label[0]),int(label[1])),(int(label[2]),int(label[3])),(255,0,0),2)    
    
        cv2.imshow("Class: {}".format(label[4]),cv_im)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
     
        
        
def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def plot_text(im,offset,cls,idnum,class_colors):
    """ Plots filled text box on original image, 
        utility function for plot_bboxes_2d """
    
    text = "{}: {}".format(idnum,cls)
    
    font_scale = 1.0
    font = cv2.FONT_HERSHEY_PLAIN
    
    # set the rectangle background to white
    rectangle_bgr = class_colors[cls]
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    
    # set the text start position
    text_offset_x = int(offset[0])
    text_offset_y = int(offset[1])
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)


def plot_bboxes_2d(im,label):
    """ Plots rectangular bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file 
    bbox_im -  cv2 im with bboxes and labels plotted
    """
    
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(255,100,0),
            'Person':(255,50,0),
            'Car': (0,255,150),
            'Van': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    for det in label:
        bbox = det['bbox2d']
        cls = det['class']
        idnum = det['id']
        
        cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[cls], 1)
        if cls != 'DontCare':
            plot_text(cv_im,(bbox[0],bbox[1]),cls,idnum,class_colors)
    return cv_im


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

    
def draw_prism(im,coords,color):
    """ draws a rectangular prism on a copy of an image given the x,y coordinates 
    of the 8 corner points, does not make a copy of original image
    im - cv2 image
    coords - 2x8 numpy array with x,y coords for each corner
    prism_im - cv2 image with prism drawn"""
    prism_im = im.copy()
    coords = np.transpose(coords).astype(int)
    #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
    edge_array= np.array([[0,1,0,1,1,0,0,0],
                          [1,0,1,0,0,1,0,0],
                          [0,1,0,1,0,0,1,1],
                          [1,0,1,0,0,0,1,1],
                          [1,0,0,0,0,1,0,1],
                          [0,1,0,0,1,0,1,0],
                          [0,0,1,0,0,1,0,1],
                          [0,0,0,1,1,0,1,0]])

    # plot lines between indicated corner points
    for i in range(0,8):
        for j in range(0,8):
            if edge_array[i,j] == 1:
                cv2.line(prism_im,(coords[i,0],coords[i,1]),(coords[j,0],coords[j,1]),color,1)
    return prism_im


def plot_bboxes_3d(im,label,P, style = "normal"):
    """ Plots rectangular prism bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file
    P - camera calibration matrix
    bbox_im -  cv2 im with bboxes and labels plotted
    style - string, "ground_truth" or "normal"  ground_truth plots boxes as white
    """
        
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(200,800,0),
            'Person':(160,30,0),
            'Car': (0,255,150),
            'Van': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    for i in range (0,len(label)):
        if label[i]['pos'][2] > 2 and label[i]['truncation'] < 1:
            cls = label[i]['class']
            idnum = label[i]['id']
            if cls != "DontCare":
                bbox_3d,_,_ = get_coords_3d(label[i],P)
                if style == "ground_truth": # for plotting ground truth and predictions
                    cv_im = draw_prism(cv_im,bbox_3d,(255,255,255))
                else:
                    cv_im = draw_prism(cv_im,bbox_3d,class_colors[cls])
                    plot_text(cv_im,(bbox_3d[0,4],bbox_3d[1,4]),cls,idnum,class_colors)
    return cv_im

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
    
    test = Localization_Dataset(train_im_dir,train_lab_dir,train_calib_dir)
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
    #mean,std = get_dataset_mean_stdev(train_im_dir)
        
    for i in range(10):
        test.show(np.random.randint(0,len(test)))
    
    
            