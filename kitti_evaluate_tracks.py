import _pickle as pickle
from kitti_train_localizer import class_dict
from torch_kf_dual import Torch_KF
from kitti_track_utils import skip_track

from KITTI_eval_simple import evaluate
# for evaluation script
import sys,os,copy,math
from munkres import Munkres
from collections import defaultdict
try:
    from ordereddict import OrderedDict # can be installed using pip
except:
    from collections import OrderedDict # only included from python 2.7 on
import mailpy


def write_kitti_output(tracklets,sequence_name, out_dir):
    """
    Description
    -----------
    Parses results in the format output by skip_track and writes into KITTI label
    file format, saving a file of the same name as the input in desired directory
    
    Parameters
    ----------
    tracks - list of lists, one per frame
        Each sublist contains dicts, one per estimated object location, with fields
        "bbox", "id", and "class_num"
    sequence_name - string 4-digit number
        Name of input sequence (0000,0001, etc.)
    out_dir - string
        Name of directory to which output .txt file is written

    Returns
    -------
    None.

    """
    
    # values separated by spaces
    
    output_lines = []
    
    for i,frame in enumerate(tracklets):
        for obj in frame:
            filler = "invalid"
            occluded = -1
            truncated = -1
            alpha = -10.0
            
            cls = class_dict[obj["class_num"]]
            xmin = obj["bbox"][0]
            xmax = obj["bbox"][2]
            ymin = obj["bbox"][1]
            ymax = obj["bbox"][3]
            confidence = 0.90
            
            # 3D coords, filled with default values for now
            others = "{:6f} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}".format(-1000,-1000,-1000,-10,-1,-1,-1)
            #others = "{} {} {} {} {} {} {}".format(filler,filler,filler,filler,filler,filler,filler)
            output_line = "{} {} {} {} {} {} {:6f} {:6f} {:6f} {:6f} {} {:6f} \n".format(
                i,
                obj["id"],
                cls,
                occluded,
                truncated,
                alpha,
                xmin,
                ymin,
                xmax,
                ymax,
                others,
                confidence)
            
            output_lines.append(output_line)
    out_file =  os.path.join(out_dir,str(sequence_name) + ".txt")       
    with open(out_file, "w" ) as f:
        f.writelines(output_lines)
    
    return out_file

if __name__ == "__main__":
        
    # input parameters
    overlap = 0.2
    conf_cutoff = 3
    iou_cutoff = 0.75
    det_step = 5
    srr = 1
    ber = 2.1 #1.95
    init_frames = 1
    matching_cutoff = 100
    mask_others = True
    
   
    SHOW = False
    
    # get list of all files in directory and corresponding path to track and labels
    track_dir =   "/home/worklab/Desktop/KITTI/data_tracking_image_2/training/image_02"  
    label_dir =   "/home/worklab/Desktop/KITTI/data_tracking_label_2/training/label_02"
    output_directory = "_working_outputs"
    
    # sort sequences into dictionary
    track_dict = {}
    ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #ids = [17,18,19,20]
    
    for id in ids:
        track_dict[id] = {"frames":os.path.join(track_dir,str(id).zfill(4)),
                          "labels":os.path.join(label_dir,str(id).zfill(4) +".txt")
                          }
        
    # get filter covariance matrices
    with open("kitti_velocity7_QR.cpkl" ,"rb") as f:
             kf_params = pickle.load(f)
    

    # track and evaluate each sequence    
    for id in track_dict:
        tracker = Torch_KF("cpu",mod_err = 1, meas_err = 1, state_err = 0, INIT =kf_params, ADD_MEAN_Q = False)
        frames = track_dict[id]["frames"]
        
        preds, Hz, time_metrics = skip_track(frames,
                                                          tracker,
                                                          detector_resolution = 1024,
                                                          det_step = det_step,
                                                          init_frames = init_frames,
                                                          fsld_max = det_step,
                                                          matching_cutoff = matching_cutoff,
                                                          iou_cutoff = iou_cutoff,
                                                          conf_cutoff = conf_cutoff,
                                                          srr = srr,
                                                          ber = ber,
                                                          mask_others = mask_others,                                               
                                                          PLOT = SHOW)
        
        pred_path = write_kitti_output(preds,str(id).zfill(4),output_directory)
        print("wrote output file {}".format(id) )
    
    
    # get results at the end
    mapping_path = "./data/tracking/evaluate_tracking.seqmap"
    file_id = "evaluations_{}".format(id) # give a unique tag to the output files
    address = None

    # create mail messenger and debug output object
    if address:
      mail = mailpy.Mail(address)
    else:
      mail = mailpy.Mail("")

    # evaluate results and send notification email to user
    success = evaluate(output_directory,label_dir,mapping_path,mail,file_id = file_id)

        