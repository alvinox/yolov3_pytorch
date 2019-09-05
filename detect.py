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
    images = args.images
    det = args.det
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    datacfg = args.datacfg
    CUDA = torch.cuda.is_available()
    #CUDA = False

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

    load_images_start = time.time()
    try:
        img_extension = [".png", ".jpeg", ".jpg"]
        img_list = [os.path.join(os.path.realpath("."), images, img) for img in os.listdir(images) if os.path.splitext(img)[1].lower() in img_extension]
    except NotADirectoryError:
        img_list = []
        img_list.append(os.path.join(os.path.realpath("."), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    if not os.path.exists(det):
        os.makedirs(det)


    batches = []
    for img in img_list:
        batches.append(prep_image(img, inp_dim))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list)

    load_images_end = time.time()
 
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 1 if len(img_list) % batch_size != 0 else 0
    if batch_size != 1:
        num_batches = len(img_list) // batch_size + leftover
        im_batches = [torch.cat(im_batches[i*batch_size : \
                                           min((i+1)*batch_size, len(img_list))]) \
                                           for i in range(num_batches)]

    
    write = False
    
    detect_start = time.time()
    #objs = {}

    for batch_index, batch in enumerate(im_batches):
        det_batch_start = time.time()
        if CUDA:
            batch = batch.cuda()


        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B * (bbox properties * no. of anchors) * grid_h * grid_w --> B * bbox * (all the boxes) 
        # Put every proposed box as a row
        with torch.no_grad():
            # call Darknet forward
            prediction = model(Variable(batch), CUDA)

        det_batch_end = time.time()
        det_batch_time = det_batch_end - det_batch_start

        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results
        prediction = write_result(prediction, confidence, num_classes, nms = True, nms_conf = nms_thresh)
        if type(prediction) == int:
            for im_ind, image in enumerate(img_list[batch_index*batch_size : \
                                                    min((batch_index+1)*batch_size, len(img_list))]):
                image = os.path.basename(image)
                print("{0:20s} predicted in {1:6.3} seconds".format(image, det_batch_time/batch_size))
                print("{0:20s}".format("no object detected."))
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += batch_index*batch_size
        if not write:
            output = prediction
            write = True
        else:
            output = torch.cat((output, prediction))

        if CUDA:
            torch.cuda.synchronize()

        for im_ind, image in enumerate(img_list[batch_index*batch_size : \
                                                min((batch_index+1)*batch_size, len(img_list))]):
            im_global_ind = batch_index*batch_size + im_ind
            image = os.path.basename(image)
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_global_ind]
            print("{0:20s} predicted in {1:6.3} seconds".format(image, det_batch_time/batch_size))
            print("{0:20s} {1:s}".format("objects detected:", ", ".join(objs)))
            print("----------------------------------------------------------")

    try:
        output
    except NameError:
        print("nothing deteted in every image.")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim_list[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim_list[:, 1].view(-1, 1))/2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast_end = time.time()

    for out in output:
        # draw box and label on original images
        write_image(out, orig_ims, classes)
    for i, image in enumerate(img_list):
        image = os.path.basename(image)
        det_name = "{}/det_{}".format(args.det, image)
        cv2.imwrite(det_name, orig_ims[i])

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading images", load_images_end - load_images_start))
    print("{:25s}: {:2.3f}".format("Transform images to batch", detect_start - load_images_end))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(img_list)) +  " images)", output_recast_end - detect_start))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - output_recast_end))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_images_end)/len(img_list)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()
