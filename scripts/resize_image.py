import os
import numpy as np
import cv2

def get_test_input(file_name, h, w):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (h, w))
    base_name = os.path.basename(file_name)
    print(base_name)
    new_file_name = "new_" + base_name
    cv2.imwrite(new_file_name, img)

img = "/home/alvinox/workspace/YOLOv3-pytorch/imgs/herd_of_horses.jpg"
get_test_input(img, 416, 416)
