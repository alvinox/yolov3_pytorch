import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

from util import *

class MaxPoolStride(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPoolStride, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = kernel_size - 1

    def forward(self, x):
        if self.stride != 1:
            x = nn.MaxPool2d(self.kernel_size, self.stride)(x)
        else:
            padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
            x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return x

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes

    def predict_transform(self, prediction, inp_dim, CUDA = True):
        batch_size = prediction.size(0)
        stride = inp_dim // prediction.size(2)
        grid_size = inp_dim // stride
        bbox_attrs = 5 + self.num_classes
        num_anchors = len(self.anchors)

        prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

        #Add the center offsets
        grid_len = np.arange(0, grid_size)
        a, b = np.meshgrid(grid_len, grid_len)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)
        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1)
        x_y_offset = x_y_offset.repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

        prediction[:, :, :2] += x_y_offset

        #log space transform height and the width
        anchors = torch.FloatTensor(self.anchors)
        if CUDA:
            anchors = anchors.cuda()
        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

        #Softmax the class scores
        prediction[:, :, 5 : bbox_attrs] = torch.sigmoid(prediction[:, :, 5 : bbox_attrs])
        prediction[:, :, :2] *= stride

        return prediction

class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.blocks = []
        self.net_info = {}
        self.module_list = nn.ModuleList()
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def create_modules(self):
        blocks = self.blocks
        net_info = blocks[0] #Captures the information about the input and pre-processing

        module_list = nn.ModuleList()
        index = 0 #indexing blocks helps with implementing route  layers (skip connections)

        prev_filters = 3
        output_filters = []

        for x in blocks:
            module = nn.Sequential()

            if (x["type"] == "net" or x["type"] == "network"):
                continue

            #If it's a convolutional layer
            if (x["type"] == "convolutional"):
                #Get the info about the layer
                activation = x["activation"]
                try:
                    batch_normalize = int(x["batch_normalize"])
                except KeyError:
                    batch_normalize = 0
                bias = not batch_normalize

                filters = int(x["filters"])
                padding = int(x["pad"])
                kernel_size = int(x["size"])
                stride = int(x["stride"])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                #Add the convolutional layer
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
                module.add_module("conv_{0}".format(index), conv)

                #Add the Batch Norm Layer
                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_normalize_{0}".format(index), bn)

                #Add the Activate Layer
                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace = True)
                    module.add_module("leaky_{0}".format(index), activn)
            
            #If it's a upsample layer
            elif (x["type"] == "upsample"):
                stride = int(x["stride"])
                upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
                module.add_module("upsample_{0}".format(index), upsample)

            #If it's a route layer
            elif (x["type"] == "route"):
                layers = x["layers"].split(',')
                #Start  of a route
                start = int(layers[0])
                if start < 0:
                    start = index + start

                #end, if there exists one.
                try:
                    end = int(layers[1])
                    if end < 0:
                        end = index + end
                    filters = output_filters[start] + output_filters[end]
                except:
                    filters = output_filters[start]

                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)

            #If it's a shortcut layer
            elif (x["type"] == "shortcut"):
                shortcut = EmptyLayer()
                module.add_module("shortcut_{0}".format(index), shortcut)

            #If it's a maxpool layer
            elif (x["type"] == "maxpool"):
                stride = int(x["stride"])
                size = int(x["size"])

                maxpool = MaxPoolStride(size, stride)

                module.add_module("maxpool_{0}".format(index), maxpool)
     
            #Yolo is the detection layer
            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(x) for x in mask]
                anchors = x["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                anchors=[anchors[i] for i in mask]
 
                num_classes = int(x["classes"])

                yolo_layer = YoloLayer(anchors, num_classes)
                module.add_module("yolo_{0}".format(index), yolo_layer)

            else:
                print("Error. unknown layer: {0}".format(x["type"]))
                assert False

            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)
            index += 1
        return (net_info, module_list)

    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]
        outputs = {} #We cache the outputs for the route layer

        write = False
        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or \
               module_type == "upsample" or \
               module_type == "maxpool" :
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"].split(',')
                #Start  of a route
                start = int(layers[0])
                if start < 0:
                    start = i + start

                #end, if there exists one.
                try:
                    end = int(layers[1])
                    if end < 0:
                        end = i + end
                    x1 = outputs[start]
                    x2 = outputs[end]
                    x = torch.cat((x1, x2), 1)
                except:
                    x = outputs[start]
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x

            elif module_type == "yolo":
                #Get the input dimensions
                inp_dim = int(self.net_info["height"])
                
                #Output the result
                yolo_layer = self.module_list[i][0]
                x = yolo_layer.predict_transform(x, inp_dim, CUDA)

                # cat results from different grid_size into detections
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                
                outputs[i] = outputs[i-1]

        return detections

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen during train

        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        #The rest of the values are the weights
        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            block = self.blocks[i+1]
            module_type = block["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(block["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr += num_bn_biases

                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean.data)
                    bn_running_var = bn_running_var.view_as(bn.running_var.data)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.data.copy_(bn_running_mean)
                    bn.running_var.data.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr : ptr+num_biases])
                    ptr += num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #copy the data to model
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_conv_weights = conv.weight.numel()

                #Load the weights
                conv_weights = torch.from_numpy(weights[ptr : ptr+num_conv_weights])
                ptr += num_conv_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



