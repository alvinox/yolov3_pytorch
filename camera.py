import time
import os

import numpy as np
import cv2
import torch
import torch.nn as nn

from util import *
from process_img import *
from darknet import Darknet

if __name__ == '__main__':
    args = arg_parse()

    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    datacfg = args.datacfg
    det = args.det
    CUDA = torch.cuda.is_available()

    datacfg_map = parse_datacfg(datacfg)
    num_classes = int(datacfg_map['classes'])
    classes = load_classes(datacfg_map['names'])

    #Set up the neural network
    print("Loading network.....")
    model = Darknet()
    model.blocks = parse_cfg(args.cfgfile)
    model.net_info, model.module_list = model.create_modules()
    model.load_weights(args.weightsfile)
    # print(model.module_list)
    print("Network successfully loaded")

    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()

    cap_source = args.video if args.video else 0
    cap = cv2.VideoCapture(cap_source)
    assert cap.isOpened(), 'Cannot capture source'


    save_video = args.save_video
    if save_video:
        video_name = os.path.basename(args.video) if args.video else 'camera'
        video_name = os.path.splitext(video_name)[0]
        det_name = "{}/det_{}.mp4".format(det, video_name)
        property_fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(det_name, cv2.VideoWriter_fourcc(*"mp4v"), property_fps, size, True)
        write_frames = []

    if not os.path.exists(det):
        os.makedirs(det)

    frames = 0
    start_time = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_frame(frame, inp_dim)

            if CUDA:
                img = img.cuda()

            with torch.no_grad():
                prediction = model(Variable(img), CUDA)
            output = write_result(prediction, confidence, num_classes, nms = True, nms_conf = nms_thresh)
            
            if type(output) == int:
                if save_video:
                    write_frames.append(orig_im)
                frames += 1            
                clear = showfps(orig_im, frames, start_time)
                if clear:
                    start_time = time.time()
                    frames = 0
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            output[:, 1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            output[:, [1 ,3]] *= frame.shape[1]
            output[:, [2 ,4]] *= frame.shape[0]

            for out in output:
                # draw box and label on original images
                write_frame(out, orig_im, classes)
            if save_video:
                write_frames.append(orig_im)

            frames += 1
            clear = showfps(orig_im, frames, start_time)
            if clear:
                start_time = time.time()
                frames = 0
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        elif args.video:
            break

    cap.release()
    if save_video:
        print('saving {} ...'.format(det_name))
        total_frames = len(write_frames)
        print('{} frames to save.'.format(total_frames))
        for i, frame in enumerate(write_frames):
            writer.write(frame)
            if i % 1000 == 0 and i != 0:
                print('%.2f%% saved' % (i/total_frames * 100))
                writer.release();
        print('{} successfully saved.'.format(det_name))
        
