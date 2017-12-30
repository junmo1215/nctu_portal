# -*- coding: UTF-8 -*-

import os
from PIL import Image, ImageOps
import numpy as np
from helper import random_name
import cv2

IMAGE_FOLDER = os.path.join("data", "labeled", "2", "test")
OUTPUT_FOLDER = os.path.join("single_letters", "2", "test")
IMAGE_SIZE = 50
MODEL_INPUT_SIZE = (28, 28)

def main():
    for i in range(10):
        dir_path = os.path.join(OUTPUT_FOLDER, str(i))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    for img_file in os.listdir(IMAGE_FOLDER):
        if img_file.endswith('.jpg'):
            # print(img_file)
            im = Image.open(os.path.join(IMAGE_FOLDER, img_file))
            im = im.convert('L')

            step = "filter_img_"
            # imgry = get_num_img(imgry)
            
            imgry = filter_img(im, get_background_color(im))
            # imgry.save(os.path.join(OUTPUT_FOLDER, step + img_file), 'jpeg')

            # 去除干扰线
            step = "remove_lines_img_"
            imgry = and_with_bias(imgry, 2)
            imgry = or_with_bias(imgry, 1)

            # im3.show()
            # imgry.save(os.path.join(OUTPUT_FOLDER, step + img_file), 'jpeg')

            imgs = cut_image(imgry)
            # for img in imgs:
            #     img.save(os.path.join(OUTPUT_FOLDER, random_name()))
            if len(imgs) != 4:
                print("cut image warning: {}".format(img_file))
                continue

            for i in range(4):
                # temp_img = resize_image(imgs[i])
                imgs[i].resize(MODEL_INPUT_SIZE).save(os.path.join(OUTPUT_FOLDER, img_file[i], random_name()))

# def get_num_img(imgry):
#     colors = imgry.getcolors()
#     colors = sorted(colors, key=lambda x: x[0], reverse=True)
#     for num, color in colors:
#         imgry = filter_img(im, get_number_color(im))
IMAGE_PADDING = 2
from PIL import ImageDraw
def cut_image(imgry):
    draw = ImageDraw.Draw(imgry)
    open_cv_image = np.array(imgry)
    image, contours, hierarchy = cv2.findContours(open_cv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key = lambda x:x[1])

    images = []
    lh = 0
    rh = 0
    for (c,_) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print(x, y, w, h)
        if 20 < h < 70 and 10 < w < 70:
            # 排除0这类的数字可能会捕捉到两个边界的情况
            if x > lh and x + w < rh:
                continue

            lh = x
            rh = x + w

            # 这种情况应该是两个数字连在一起了，从中间切开
            # TODO: 也有可能是三个数字连在一起了，这边暂时直接去掉，后面保存图片的时候会考虑 
            if w > h:
                draw.rectangle((x-IMAGE_PADDING, y-IMAGE_PADDING, x + w // 2 + IMAGE_PADDING, y + h + IMAGE_PADDING), outline="Red")
                temp = subimage(imgry, (x-IMAGE_PADDING, y-IMAGE_PADDING, x + w // 2 + IMAGE_PADDING, y + h + IMAGE_PADDING))
                images.append(temp)

                draw.rectangle((x + w // 2 -IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING), outline="Red")
                temp = subimage(imgry, (x + w // 2 -IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING))
                images.append(temp)
            else:
                draw.rectangle((x-IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING), outline="Red")
                temp = subimage(imgry, (x-IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING))
                images.append(temp)
    # print(len(ary))
    imgry.save(os.path.join(OUTPUT_FOLDER, random_name()), 'jpeg')
    return images

def subimage(imgry, box):
    temp = imgry.crop(box)
    temp = temp.resize((IMAGE_SIZE, IMAGE_SIZE))
    # 加入了反色处理，这种情况好像容易分割一点
    # 理论上不确定为什么，用验证码测试结果比之前好很多
    temp = ImageOps.invert(temp)
    return temp


def filter_img(img, threshold=170):
    '对于黑白图像，将像素值大于某一个值的部分变成黑色，小于这个值的部分变成白色'
    filtered = Image.new(img.mode, img.size)
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
            if img.getpixel((x, y)) != threshold:
                filtered.putpixel((x, y), 255)
            else:
                filtered.putpixel((x, y), 0)
    return filtered

def or_with_bias(imgry, bias):
    im = np.array(imgry) / 255
    im = im.astype(np.bool_)
    (row, col) = im.shape

    empty_rows = np.zeros(shape=(bias, col))
    empty_cols = np.zeros(shape=(row, bias))

    sub_ims = [
        np.concatenate((im[bias:, :], empty_rows), axis=0),
        np.concatenate((empty_rows, im[:row - bias, :]), axis=0),
        np.concatenate((im[:, bias:], empty_cols), axis=1),
        np.concatenate((empty_cols, im[:, :col - bias]), axis=1)
    ]

    for sub_im in sub_ims:
        # Image.fromarray((255 * sub_im).astype("uint8")).show()
        im = np.logical_or(im, sub_im)

    return Image.fromarray((255 * im).astype("uint8"))


def and_with_bias(imgry, bias):
    im = np.array(imgry) / 255
    im = im.astype(np.bool_)
    (row, col) = im.shape

    empty_rows = np.zeros(shape=(bias, col))
    empty_cols = np.zeros(shape=(row, bias))

    sub_ims = [
        np.concatenate((im[bias:, :], empty_rows), axis=0),
        np.concatenate((empty_rows, im[:row - bias, :]), axis=0),
        np.concatenate((im[:, bias:], empty_cols), axis=1),
        np.concatenate((empty_cols, im[:, :col - bias]), axis=1)
    ]

    for sub_im in sub_ims:
        # Image.fromarray((255 * sub_im).astype("uint8")).show()
        im = np.logical_and(im, sub_im)

    return Image.fromarray((255 * im).astype("uint8"))

def get_background_color(img):
    colors = img.getcolors()
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    return colors[0][1]

def get_number_and_line_color(img):
    colors = img.getcolors()
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    # 取出三种数量最多的颜色
    # 最多的是背景
    # 然后两种不出意外是数字的颜色和干扰线的颜色
    # 这里取出这两个颜色中较大的一个
    # TODO: 有个问题，要是干扰线或者数字的颜色比背景要浅，会导致后面一步的过滤全部都变成白色，可以通过mask的方式解决

    max_color = 0
    for num, color in colors[1:3]:
        # print("num: {}\tcolor:{}".format(num, color))
        if color > max_color:
            max_color = color
    return max_color



if __name__ == "__main__":
    main()
