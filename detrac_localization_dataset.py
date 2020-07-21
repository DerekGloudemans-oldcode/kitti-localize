"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
"""

import os
import numpy as np
import random 
import math
random.seed = 0

import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

try:
    from detrac_files.detrac_plot_utils_copy import pil_to_cv, plot_bboxes_2d
except:
    from detrac_plot_utils_copy import pil_to_cv, plot_bboxes_2d


class Localize_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    and returning single object images for localization. Note that this dataset
    does not automatically separate training and validation data, so you'll 
    need to partition data manually by separate directories
    """
    
    def __init__(self, image_dir, label_dir):
        """ initializes object
        image dir - (string) - a directory containing a subdirectory for each track sequence
        label dir - (string) - a directory containing a label file per sequence
        """

        # stores files for each image
        dir_list = next(os.walk(image_dir))[1]
        track_list = [os.path.join(image_dir,item) for item in dir_list]
        track_list.sort()
        
        # parse labels and store in dict keyed by track name
        label_list = {}
        for item in os.listdir(label_dir):
            name = item.split("_v3.xml")[0]
            label_list[name] = self.parse_labels(os.path.join(label_dir,item))
        
        self.im_tf = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness = 0.3,contrast = 0.3,saturation = 0.2)
                        ]),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),

                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])

        # for denormalizing
        self.denorm = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
        
        # for storing data
        self.all_data = []
        
        # parse and store all labels and image names in a list such that
        # all_data[i] returns dict with image name, label and other stats
        # track_offsets[i] retuns index of first frame of track[i[]]
        for i in range(len(track_list)): #[0]: #

            images = [os.path.join(track_list[i],frame) for frame in os.listdir(track_list[i])]
            images.sort() 
            labels,metadata = label_list[track_list[i].split("/")[-1]]
            
            # each iteration of the loop gets one image
            print (len(images),len(labels))
            for j in range(len(images)):
                try:
                    image = images[j]
                    label = labels[j]
                    
                    # each iteration gets one label (one detection)
                    for k in range(len(label)):
                        detection = label[k]
                        self.all_data.append((image,detection))

                except:
                    # this occurs because Detrac was dumbly labeled and they didn't include empty annotations for frames without objects
                    # parse_labels corrects this mostly, except for trailing frames
                    # so we just pass because there are no objects or labels anyway
                    pass
                    print("Error: tried to load label {} for track {} but it doesnt exist. Labels is length {}".format(j,track_list[i],len(labels))) 
                
                    
        # in case it is later important which files are which
        self.track_list = track_list
        self.label_list = label_list
        


    def __len__(self):
        """ returns total number of frames in all tracks"""
        return len (self.all_data)
       # return self.total_num_frames
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
    
        # load image and get label        
        cur = self.all_data[index]
        im = Image.open(cur[0])
        label = cur[1]
        
        # get class and convert to kitti class
        cls = label['class_num']
        
        detrac_to_kitti = {
            0: 3,
            1: 3,
            2: 3,
            3: 0,
            4: 3,
            5: 3,
            6: 4,
            7: 5,
            8: 0,
            9: 5,
            10:5,
            11:5,
            12:5    
            }
        cls = detrac_to_kitti[cls]
                

        # crop image so that only relevant portion is showing for one object
        # copy so that original coordinates aren't overwritten
        bbox = label["bbox"].copy()
        
        bbox_width = bbox[2] - bbox[0]
                
        # flip sometimes
        if np.random.rand() > 0.5:
            im= F.hflip(im)
            # reverse coords and also switch xmin and xmax
            bbox[[2,0]] = im.size[0] - bbox[[0,2]]
        
        # randomly shift the center of the crop
        shift_scale = 90
        x_shift = np.random.normal(0,im.size[0]/shift_scale)
        y_shift = np.random.normal(0,im.size[1]/shift_scale)
        #x_shift = 0
        #y_shift = 0
        
        #buffer  = 0#min(bbox[2]-bbox[0],bbox[3]-bbox[1])/3# max(-5,np.random.normal(70,im.size[1]/shift_scale))
        bufferx = max(0,np.random.normal(bbox_width/4.0,10))
        buffery = max(0,bufferx*(np.random.rand()+0.5))
        # note may have indexed these wrongly
        minx = max(0,bbox[0]-bufferx)
        miny = max(0,bbox[1]-buffery)
        maxx = min(im.size[0],bbox[2]+bufferx)
        maxy = min(im.size[1],bbox[3]+buffery)
    
        minx = minx + x_shift
        maxx = maxx + x_shift
        miny = miny + y_shift
        maxy = maxy + y_shift
    
        im_crop = F.crop(im,miny,minx,maxy-miny,maxx-minx)
        del im 
        
        if im_crop.size[0] == 0 or im_crop.size[1] == 0:
            print("Oh no!")
            raise Exception
            
        bbox[0] = bbox[0] - minx
        bbox[1] = bbox[1] - miny
        bbox[2] = bbox[2] - minx
        bbox[3] = bbox[3] - miny
         
        orig_size = im_crop.size
        im_crop = F.resize(im_crop, (224,224))
        
        bbox[0] = bbox[0] * 224/orig_size[0]
        bbox[2] = bbox[2] * 224/orig_size[0]
        bbox[1] = bbox[1] * 224/orig_size[1]
        bbox[3] = bbox[3] * 224/orig_size[1]
        
        # apply random affine transformation
        y = np.zeros(5)
        y[0:4] = bbox
        y[4] = cls
        
        #im_crop,y = self.random_affine_crop(im_crop,y)

        
        
        # convert image and label to tensors
        im_t = self.im_tf(im_crop)
        return im_t, y
    
    
    
    def random_affine_crop(self,im,y,imsize = 224,tighten = 0.02,max_scaling = 1.5):
        """
        Performs transforms that affect both X and y, as the transforms package 
        of torchvision doesn't do this elegantly
        inputs: im - image
                 y  - 1 x 5 numpy array of bbox corners and class. the order is: 
                    min x, min y, max x max y
        outputs: im - transformed image
                 y  - 1 x 5 numpy array of bbox corners and class
        """
    
        #define parameters for random transform
        scale2 = min(max_scaling,max(random.gauss(0.5,1),imsize/min(im.size))) +0.5 # verfify that scale will at least accomodate crop size
        scale = min(1.5, max(0.75,np.random.normal(1,0.25)))
        #scale = 1
        shear = 0# (random.random()-0.5)*30 #angle
        rotation = 0#(random.random()-0.5) * 30 #angle
        
        # transform matrix
        im = transforms.functional.affine(im,rotation,(0,0),scale,shear,fillcolor = (int(0.485*255),int(0.456*255),int(255*0.406)))
        (xsize,ysize) = im.size
        
        # only transform coordinates for positive examples (negatives are [0,0,0,0,0])
        # clockwise from top left corner
        if True:
            
            # image transformation matrix
            shear = math.radians(-shear)
            rotation = math.radians(-rotation)
            M = np.array([[scale*np.cos(rotation),-scale*np.sin(rotation+shear)], 
                          [scale*np.sin(rotation), scale*np.cos(rotation+shear)]])
            
            
            # add 5th point corresponding to image center
            corners = np.array([[y[0],y[1]],[y[2],y[1]],[y[2],y[3]],[y[0],y[3]],[int(xsize/2),int(ysize/2)]])
            new_corners = np.matmul(corners,M)
            
            # Resulting corners make a skewed, tilted rectangle - realign with axes
            old_class = y[4]
            y = np.ones(5)
            y[0] = np.min(new_corners[:4,0])
            y[1] = np.min(new_corners[:4,1])
            y[2] = np.max(new_corners[:4,0])
            y[3] = np.max(new_corners[:4,1])
            y[4] = old_class
            # shift so transformed image center aligns with original image center
            xshift = xsize/2 - new_corners[4,0]
            yshift = ysize/2 - new_corners[4,1]
            y[0] = y[0] + xshift
            y[1] = y[1] + yshift
            y[2] = y[2] + xshift
            y[3] = y[3] + yshift
            
            # brings bboxes in slightly on positive examples
            if tighten != 0:
                xdiff = y[2] - y[0]
                ydiff = y[3] - y[1]
                y[0] = y[0] + xdiff*tighten
                y[1] = y[1] + ydiff*tighten
                y[2] = y[2] - xdiff*tighten
                y[3] = y[3] - ydiff*tighten
            
        # get center of crop location
        crop_x = int(im.size[0]/2)
        crop_y = int(im.size[1]/2)
        
        #crop_x = int(np.random.normal((y[2]+y[0])/2 , 50))
        #crop_y = int(np.random.normal((y[1]+y[3])/2 , 50))
        
        # move crop if too close to edge
        pad = 0
        if crop_x < pad:
            crop_x = im.size[0]/2 - imsize/2 # center
        if crop_y < pad:
            crop_y = im.size[1]/2 - imsize/2 # center
        if crop_x > im.size[0] - imsize - pad:
            crop_x = im.size[0]/2 - imsize/2 # center
        if crop_y > im.size[0] - imsize - pad:
            crop_y = im.size[0]/2 - imsize/2 # center  
        im = transforms.functional.crop(im,crop_y,crop_x,imsize,imsize)
        
        # This is done to get a uniform scale and good pixel fill 
#        im = transforms.functional.resize(im,224)
#        y[0:4] = y[0:4] * 224/imsize
        # transform bbox points into cropped coords
        y[0] = y[0] - crop_x
        y[1] = y[1] - crop_y
        y[2] = y[2] - crop_x
        y[3] = y[3] - crop_y
        
                
        return im,y

    
    
    def parse_labels(self,label_file):
        """
        Returns a set of metadata (1 per track) and a list of labels (1 item per
        frame, where an item is a list of dictionaries (one dictionary per object
        with fields id, class, truncation, orientation, and bbox
        """
        
        class_dict = {
            'Sedan':0,
            'Hatchback':1,
            'Suv':2,
            'Van':3,
            'Police':4,
            'Taxi':5,
            'Bus':6,
            'Truck-Box-Large':7,
            'MiniVan':8,
            'Truck-Box-Med':9,
            'Truck-Util':10,
            'Truck-Pickup':11,
            'Truck-Flatbed':12,
            
            0:'Sedan',
            1:'Hatchback',
            2:'Suv',
            3:'Van',
            4:'Police',
            5:'Taxi',
            6:'Bus',
            7:'Truck-Box-Large',
            8:'MiniVan',
            9:'Truck-Box-Med',
            10:'Truck-Util',
            11:'Truck-Pickup',
            12:'Truck-Flatbed'
            }
        
        
        
        tree = ET.parse(label_file)
        root = tree.getroot()
        
        # get sequence attributes
        seq_name = root.attrib['name']
        
        # get list of all frame elements
        #frames = root.getchildren()
        frames = list(root)
        # first child is sequence attributes
        seq_attrs = frames[0].attrib
        
        # second child is ignored regions
        ignored_regions = []
        for region in frames[1]:
            coords = region.attrib
            box = np.array([float(coords['left']),
                            float(coords['top']),
                            float(coords['left']) + float(coords['width']),
                            float(coords['top'])  + float(coords['height'])])
            ignored_regions.append(box)
        frames = frames[2:]
        
        # rest are bboxes
        all_boxes = []
        frame_counter = 1
        for frame in frames:
            while frame_counter < int(frame.attrib['num']):
                # this means that there were some frames with no detections
                all_boxes.append([])
                frame_counter += 1
            
            frame_counter += 1
            frame_boxes = []
            #boxids = frame.getchildren()[0].getchildren()
            boxids = list(list(frame)[0])
            for boxid in boxids:
                #data = boxid.getchildren()
                data = list(boxid)
                coords = data[0].attrib
                stats = data[1].attrib
                bbox = np.array([float(coords['left']),
                                float(coords['top']),
                                float(coords['left']) + float(coords['width']),
                                float(coords['top'])  + float(coords['height'])])
                det_dict = {
                        'id':int(boxid.attrib['id']),
                        'class':stats['vehicle_type'],
                        'class_num':class_dict[stats['vehicle_type']],
                        'color':stats['color'],
                        'orientation':float(stats['orientation']),
                        'truncation':float(stats['truncation_ratio']),
                        'bbox':bbox
                        }
                
                frame_boxes.append(det_dict)
            all_boxes.append(frame_boxes)
        
        sequence_metadata = {
                'sequence':seq_name,
                'seq_attributes':seq_attrs,
                'ignored_regions':ignored_regions
                }
        return all_boxes, sequence_metadata
    
    def show(self,index):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """
        mean = np.array([0.485, 0.456, 0.406])
        stddev = np.array([0.229, 0.224, 0.225])
        
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
        

if __name__ == "__main__":
    #### Test script here
    try:
        test
    except:
        try:
            label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
            image_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\Tracks"
            test = Localize_Dataset(image_dir,label_dir)
    
        except:
            label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
            image_dir = "/home/worklab/Desktop/detrac/DETRAC-all-data"
            test = Localize_Dataset(image_dir,label_dir)
    idx = np.random.randint(0,len(test))
    test.show(idx)
    test.show(idx)
    
    cv2.destroyAllWindows()