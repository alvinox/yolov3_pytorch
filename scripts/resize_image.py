import os
import numpy as np
import cv2

def resize_h_w(file_name, w, h):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (w, h))
    base_name = os.path.basename(file_name)
    new_file_name = "new_" + base_name
    cv2.imwrite(new_file_name, img)
    print('eog', new_file_name)

def resize_factor(file_name, factor):
    img = cv2.imread(file_name)
    h = int(img.shape[0] * factor)
    w = int(img.shape[1] * factor)
    resize_h_w(file_name, w, h)

img = "tet/ab54341c1afa3faddb97b3602790df1c.jpg"
resize_factor(img, 0.4)
