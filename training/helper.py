# -*- coding: UTF-8 -*-

import os
import uuid
from PIL import Image
import numpy as np

def filter_img(img, threshold=170):
    '对于黑白图像，将像素值大于某一个值的部分变成白色，小于这个值的部分变成黑色'
    filtered = Image.new(img.mode, img.size)
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
            if img.getpixel((x, y)) > threshold:
                filtered.putpixel((x, y), 255)
            else:
                filtered.putpixel((x, y), 0)
    return filtered

def random_name():
    return "{}.bmp".format(uuid.uuid4().hex)

def image_to_numpy(img):
    return np.array(img)
