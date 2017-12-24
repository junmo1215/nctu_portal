# -*- coding: UTF-8 -*-

import os
import uuid
from PIL import Image
import numpy as np
import cv2

IMAGE_PADDING = 2
IMAGE_SIZE = (28, 28)

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

def cut_image(imgry):
    open_cv_image = np.array(imgry)
    image, contours, hierarchy = cv2.findContours(open_cv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key = lambda x:x[1])

    images = []
    lh = 0
    rh = 0
    for (c,_) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print(x, y, w, h)
        if 10 < h < 70:
            # 排除0这类的数字可能会捕捉到两个边界的情况
            if x > lh and x + w < rh:
                continue

            lh = x
            rh = x + w
            temp = imgry.crop((x-IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING))
            # temp.show()
            images.append(temp.resize(IMAGE_SIZE))
    # print(len(ary))
    return images

def pretreatment0(image):
    im = image.convert('L')
    im = filter_img(im, 98)
    return cut_image(im)
    # if len(imgs) != 4:
    #     print("cut image warning: {}".format(img_file))
    #     continue

    # for i in range(4):
    #     imgs[i].resize(MODEL_INPUT_SIZE).save(os.path.join(OUTPUT_FOLDER, img_file[i], random_name()))
    # return result
