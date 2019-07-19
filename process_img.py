import random

import torch 
import numpy as np
import cv2

from util import *

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_frame(frame, inp_dim):
    dim = frame.shape[1], frame.shape[0]
    img = (letterbox_image(frame, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, frame, dim

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = cv2.imread(img)
    return prep_frame(orig_im, inp_dim)

colors = load_colors("data/colors.data")
def write_frame(x, frame, classes):
    c1 = tuple(x[1:3].int()) # top-left point
    c2 = tuple(x[3:5].int()) # bottom-right point

    cls = int(x[-1])

    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(frame, c1, c2, color, 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + text_size[0] + 3, c1[1] + text_size[1] + 4
    cv2.rectangle(frame, c1, c2, color, -1)
    cv2.putText(frame, label, (c1[0], c1[1] + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return frame

def write_image(x, orig_images, classes):
    img = orig_images[int(x[0])]
    write_frame(x, img, classes)
