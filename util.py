import argparse
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

from bbox import *

def arg_parse():
    """
    Parse arguements for Yolo-v3
    
    """
    parser = argparse.ArgumentParser(description='pyann yolo-v3 detection module')
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--datacfg", dest = 'datacfg', help = "Data config file",
                        default = "cfg/coco.data", type = str)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
                        default = "cfg/yolov3-coco.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "weights/yolov3-coco.weights", type = str)
    parser.add_argument("--video", dest = 'video', help = "Video to run detection upon",
                        default = None, type = str)
    parser.add_argument("--save_video", dest = 'save_video', help = "Save the detected video",
                        default = False, action='store_true')
    return parser.parse_args()

def parse_datacfg(datacfg_file):
    """
    Takes a data configuration file
    
    Returns a map of options. 
    """
    with open(datacfg_file, 'r') as file:
        datacfg_map = {}
        lines = file.read().split('\n')     #store the lines in a list
        lines = [x for x in lines if len(x) > 0] #ignore empty lines 
        lines = [x for x in lines if x[0] != '#'] #ignore comment lines
        lines = [x.rstrip().lstrip() for x in lines]

        for line in lines:
            key,value = line.split("=")
            datacfg_map[key.strip()] = value.strip()

    return datacfg_map

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    with open(cfgfile, 'r') as file:
        block = {}
        blocks = []

        lines = file.read().split('\n')     #store the lines in a list
        lines = [x for x in lines if len(x) > 0] #ignore empty lines 
        lines = [x for x in lines if x[0] != '#'] #ignore comment lines
        lines = [x.rstrip().lstrip() for x in lines]

        for line in lines:
            if line[0] == "[":
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()
            else:
                key,value = line.split("=")
                block[key.strip()] = value.strip()
        blocks.append(block)
    return blocks

def load_classes(namesfile):
    with open(namesfile, 'r') as file:
        names = file.read().split('\n')
    return names

def load_colors(colorfile):
    colors = []

    with open(colorfile, 'r') as file:
        lines = file.read().split('\n')
        for line in lines:
            r, g, b = line.split(',')
            color = (int(r), int(g), int(b))
            colors.append(color)

    return colors

def unique(tensor):
    tenspr_np = tensor.cpu().numpy()
    unique_np = np.unique(tenspr_np)
    unique_tensor = torch.from_numpy(unique_np)

    # copy data to the device same as tensor's 
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_result(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    ind_nz = torch.nonzero(prediction[:,:,4])
    if ind_nz.size(0) == 0:
        # nothing detected
        return -1

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2]/2
    box_a[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3]/2
    box_a[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2]/2
    box_a[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3]/2
    prediction[:, :, :4] = box_a[:, :, :4]
    del box_a

    batch_size = prediction.size(0)
    write = False

    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:, 5 : 5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        image_pred = torch.cat((image_pred[:, :5], max_conf, max_conf_score), 1)

        #Get rid of the zero entries
        none_zero_index = torch.nonzero(image_pred[:, 4])
        image_pred = image_pred[none_zero_index.squeeze(), :].view(-1, 7)

        #Get the various classes detected in the imag
        try:
            img_classes = unique(image_pred[:, -1])
        except:
            continue
        # Do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred*(image_pred[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred[class_mask_ind].view(-1, 7)

            # sort the detections by objectiveness
            # maximum objectness confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            #if nms has to be done
            if nms:
                # for each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at
                    try:
                        ious = bbox_iou(image_pred_class[i, :].unsqueeze(0), image_pred_class[i+1:, :])
                    except ValueError:
                        break
                    except IndexError:
                        break

                    # Remove out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:, :] *= iou_mask
                    iou_mask_index = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[iou_mask_index].view(-1, 7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            image_pred_class = torch.cat((batch_ind, image_pred_class), 1)

            if not write:
                output = image_pred_class
                write = True
            else:
                output = torch.cat((output, image_pred_class))

    return output

def showfps(orig_im, frames, start_time):
    cv2.imshow("frame", orig_im)
    
    during = time.time() - start_time
    if during > 2:
        print("FPS of the camera is {:5.2f}".format(frames / during))
        return True
    else:
        return False
