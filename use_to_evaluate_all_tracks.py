#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:41:17 2020

@author: worklab
"""


if False:
        
        with open("kitti_velocity8_QR2.cpkl","rb") as f:
            kf_params = pickle.load(f)
            
        all_ious = []
        
        # evaluate each track once
        dataset = Track_Dataset(train_im_dir,train_lab_dir,n = 5)
        
        for iteration in range(len(dataset)):
            ious = []
            
            gts = dataset.all_labels[iteration]
            frames = dataset.all_data[iteration]
            
            tracker = Torch_KF("cpu",INIT = kf_params, ADD_MEAN_Q = True, ADD_MEAN_R = False)
            tracker.add(torch.from_numpy(gts[:1,:8]),[0])
            
            for j in range(1,len(gts)):
                if j > 20:
                    break
                tracker.predict()
                
                # get measurement
                item = frames[j]
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
                              
                frame = frame.unsqueeze(0).to(device)
                
                # crop image
                apriori = tracker.objs()[0]
                boxes = torch.from_numpy(apriori).unsqueeze(0) 
                
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
                crops = roi_align(frame,torch_boxes,(224,224))
                
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
                
                
                # update
                tracker.update(pred,[0])
                gt = torch.from_numpy(gts[j]).unsqueeze(0).float()
                aposteriori = torch.from_numpy(tracker.objs()[0]).unsqueeze(0).float()
                ious.append(iou(aposteriori,gt))
            
            all_ious.append(ious)
            print("Finished tracklet {}".format(iteration))