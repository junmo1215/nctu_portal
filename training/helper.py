# -*- coding: UTF-8 -*-

import os
import uuid
from PIL import Image, ImageOps
import numpy as np
import cv2

IMAGE_PADDING = 2
IMAGE_SIZE = (28, 28)

def filter_img(img, threshold=170):
    '对于黑白图像，将像素值大于某一个值的部分变成黑色，小于这个值的部分变成白色'
    filtered = Image.new(img.mode, img.size)
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
            if img.getpixel((x, y)) <= threshold:
                filtered.putpixel((x, y), 255)
            else:
                filtered.putpixel((x, y), 0)
    return filtered

def random_name():
    return "{}.bmp".format(uuid.uuid4().hex)

def image_to_numpy(img):
    return np.array(img)

# from PIL import ImageDraw
def cut_image(imgry):
    # draw = ImageDraw.Draw(imgry)
    open_cv_image = np.array(imgry)
    image, contours, hierarchy = cv2.findContours(open_cv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key = lambda x:x[1])

    images = []
    lh = 0
    rh = 0
    for (c,_) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print(x, y, w, h)
        if 10 < h < 70 and 5 < w < 70:
            # 排除0这类的数字可能会捕捉到两个边界的情况
            if x > lh and x + w < rh:
                continue

            lh = x
            rh = x + w

            # 这种情况应该是两个数字连在一起了，从中间切开
            # TODO: 也有可能是三个数字连在一起了，这边暂时直接去掉，后面保存图片的时候会考虑 
            if w > h:
                temp = subimage(imgry, (x-IMAGE_PADDING, y-IMAGE_PADDING, x + w // 2 + IMAGE_PADDING, y + h + IMAGE_PADDING))
                images.append(temp)

                temp = subimage(imgry, (x + w // 2 -IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING))
                images.append(temp)
            else:
                # draw.rectangle((x-IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING), outline="Red")
                temp = subimage(imgry, (x-IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING))
                images.append(temp)
    # print(len(ary))
    return images

def subimage(imgry, box):
    temp = imgry.crop(box)

    (left, top, right, bottom) = box
    w, h = right - left, bottom - top
    if w < h:
        size = (h, h)
        delta = (h - w) // 2
        box = (delta, 0)
    else:
        size = (w, w)
        delta = (w - h) // 2
        box = (0, delta)

    new_img = Image.new(size=size, mode='L', color=0)
    new_img.paste(temp, box)
    # print(w, h)
    # ratio = IMAGE_SIZE[0] / max(w, h)
    # w, h = ratio * w, ratio * h
    # print(w, h)
    new_img = new_img.resize(IMAGE_SIZE)
    # temp.thumbnail(IMAGE_SIZE, Image.ANTIALIAS)
    # 加入了反色处理，这种情况好像容易分割一点
    # 理论上不确定为什么，用验证码测试结果比之前好很多
    new_img = ImageOps.invert(new_img)
    return new_img

# OUTPUT_FOLDER = os.path.join("single_letters", "0")
def pretreatment0(image, is_fake_img=False):
    im = image.convert('L')
    im = filter_img(im, 98)
    # results = cut_image(im)
    # im.save(os.path.join(OUTPUT_FOLDER, random_name()))
    return cut_image(im)
    # if len(imgs) != 4:
    #     print("cut image warning: {}".format(img_file))
    #     continue

    # for i in range(4):
    #     imgs[i].resize(MODEL_INPUT_SIZE).save(os.path.join(OUTPUT_FOLDER, img_file[i], random_name()))
    # return result

def get_number_color(img, is_fake_img):
    colors = img.getcolors()
    # 数字的颜色像素点最多，所以按照像素点数量从大到小排序
    # 结果的第一个是数字像素点的个数以及对应的颜色
    # 排序结果类似于[(1331, 99), (653, 233), (631, 232), (626, 240)]
    num_color_index = 0
    if is_fake_img:
        num_color_index = 1
    return sorted(colors, key=lambda x: x[0], reverse=True)[num_color_index][1]

def pretreatment1(image, is_fake_img=False):
    """
    这种样式的验证码图片训练集中都是自己生成的，背景全是空白
    所以数字对应的颜色像素数量比背景要少
    """
    im = image.convert('L')
    im = filter_img(im, get_number_color(im, is_fake_img))
    return cut_image(im)
