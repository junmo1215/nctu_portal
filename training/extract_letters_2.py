# -*- coding: UTF-8 -*-

import os
from PIL import Image, ImageOps
import numpy as np
from helper import random_name, subimage
import cv2

IMAGE_FOLDER = os.path.join("data", "labeled", "2")
OUTPUT_FOLDER = os.path.join("single_letters", "2")
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
            # im = im.convert('L')

            step = "filter_img_"
            # imgry = get_num_img(imgry)
            
            imgry = filter_img(im, get_number_color2(im))
            imgry.save(os.path.join(OUTPUT_FOLDER, step + img_file), 'jpeg')

            imgry = imgry.convert('L')
            # 去除干扰线
            # step = "remove_lines_img_"
            # imgry = and_with_bias(imgry, 2)
            # imgry = or_with_bias(imgry, 1)

            # im3.show()
            # imgry.save(os.path.join(OUTPUT_FOLDER, step + img_file), 'jpeg')

            step = "filter_im_"
            imgs = cut_image(imgry)
            # Image.fromarray((mask * 255).astype("uint8")).save(os.path.join(OUTPUT_FOLDER, step + img_file), 'jpeg')
            # for img in imgs:
            #     img.save(os.path.join(OUTPUT_FOLDER, random_name()))
            if len(imgs) != 4:
                print("cut image warning: {}".format(img_file))
                continue

            for i in range(4):
                # temp_img = resize_image(imgs[i])
                imgs[i].save(os.path.join(OUTPUT_FOLDER, img_file[i], random_name()))

def get_number_color2(im):
    colors = im.getcolors()
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    im_arr = np.array(im)
    top_n = 3
    statistic = {}
    for i in range(top_n):
        statistic[i] = 0
    # 统计每个颜色在
    for i in range(im.width):
        for j in range(top_n):
            color = colors[j][1]
            if color in im_arr[:, i]:
                statistic[j] += 1
    # print(statistic)
    for index, num in statistic.items():
        if num / im.width > 0.8:
            continue
        return colors[index][1]

# def get_num_img(imgry):
#     colors = imgry.getcolors()
#     colors = sorted(colors, key=lambda x: x[0], reverse=True)
#     for num, color in colors:
#         imgry = filter_img(im, get_number_color(im))

# def cut_image(imgry):
#     """
#     把图片切割成数字，图片必须是二值化之后的结果（像素值只有0或者255）
#     返回切割后的四张图片
#     """
#     img_arr = np.array(imgry)
#     # 白色点的位置
#     dot_lists = np.argwhere(img_arr == 255)
#     if dot_lists.size == 0:
#         return []

#     top = dot_lists[:, 0].min()
#     bottom = dot_lists[:, 0].max()

#     # 让图像居中
#     padding_top = (IMAGE_SIZE - (bottom - top)) // 2
#     top = top - padding_top
#     bottom = top + IMAGE_SIZE

#     x = dot_lists[:, 1]
#     switch = False
#     results = []
#     l = 0
#     r = 0
#     # index 的取值是从x.min到x.max + 1，这样才能触发最后一次box右边界的检测
#     for index in range(x.min(), x.max() + 2):
#         if (index in x) and (switch == False):
#             l = index
#             switch = True
#         elif ((index in x) == False) and switch:
#             r = index - 1
#             switch = False

#             box = (l, top, r, bottom)
#             # print(box)
#             results.append(subimage(imgry, box))
#             # results.append(imgry.(box))

#     # print(box)
#     return results

FILE = os.path.join("data", "labeled", "2", "test", random_name())
BOX_SIZE = 1
MIN_NUM_WIDTH = 10
def cut_image(imgry):
    img_arr = np.array(imgry)
    # dot_lists = np.argwhere(img_arr == 255)
    # ys = dot_lists[:, 0]

    dot_lists = np.argwhere(img_arr == 255)
    # 检查二值化之后的图片是否全部是黑色
    if dot_lists.shape[0] == 0:
        return []
    rows = dot_lists[:, 0]
    top = rows.min()
    bottom = rows.max()

    cols = dot_lists[:, 1]
    col_min = cols.min()
    col_max = cols.max()

    # 开始往下落小球
    mask = np.zeros(shape=img_arr.shape)

    for col in range(col_min, col_max + 1, 5):
        # position = (0, 80)
        # mask = drop(img_arr, position, mask)
        position = (0, col)
        mask = drop(img_arr, position, mask)

    # Image.fromarray((mask * 255).astype("uint8")).save(FILE)
    # raise

    # 查看有哪些小球能落到底部，按照他们的轨迹作为边缘切割图片
    bottom_line = mask[-1, :]
    # print(bottom_line)
    edges = np.where(bottom_line == 1)[0]

    # 左右没有掉下来的话手动补上最边缘
    if col_min + MIN_NUM_WIDTH < edges.min():
        edges = np.insert(edges, 0, col_min)
    if col_max - MIN_NUM_WIDTH > edges.max():
        edges = np.append(edges, col_max)

    # print(edges)

    temp = []
    for col in range(edges.min(), edges.max() + 1):
        if (col in edges) and ((col + 1) not in edges):
            temp.append(col)
    # print(temp)
    edges = temp

    results = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        width = right - left
        if width > MIN_NUM_WIDTH:
            if width > bottom - top:
                # box = (left, top, left + width // 2, bottom)
                results.append(subimage(imgry, (
                    left, top, left + width // 2, bottom)))
                results.append(subimage(imgry, (
                    left + width // 2, top, right, bottom)))
            else:
                # box = (left, top, right, bottom)
                results.append(subimage(imgry, (
                    left, top, right, bottom)))
    return results

def drop(img_arr, position, mask):
    row, col = position
    h, w = img_arr.shape
    while row is not None and col is not None:

        # 判断是否出界
        if row >= h or col >= w or col < 0:
            break

        # 判断这个部分是否之前走过
        if np.sum(mask[row:row+BOX_SIZE, col:col+BOX_SIZE] != 1) == 0:
            break

        mask[row:row+BOX_SIZE, col:col+BOX_SIZE] = 1

        row, col = next_position(row, col, img_arr)

    return mask

def next_position(row, col, im_arr):
    row += BOX_SIZE
    box = np.zeros(shape=im_arr.shape)
    box[row:row+BOX_SIZE, col:col+BOX_SIZE] = 1
    if is_collision(box, im_arr) == False:
        return row, col
    row -= BOX_SIZE

    col += BOX_SIZE
    box = np.zeros(shape=im_arr.shape)
    box[row:row+BOX_SIZE, col:col+BOX_SIZE] = 1
    if is_collision(box, im_arr) == False:
        return row, col
    col -= BOX_SIZE

    col -= BOX_SIZE
    box = np.zeros(shape=im_arr.shape)
    box[row:row+BOX_SIZE, col:col+BOX_SIZE] = 1
    if is_collision(box, im_arr) == False:
        return row, col
    col += BOX_SIZE

    return (None, None)

def is_collision(box1, box2):
    return np.sum(np.logical_and(box1, box2)) > 0

# IMAGE_PADDING = 2
# from PIL import ImageDraw
# def cut_image(imgry):
#     draw = ImageDraw.Draw(imgry)
#     open_cv_image = np.array(imgry)
#     image, contours, hierarchy = cv2.findContours(open_cv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key = lambda x:x[1])

#     images = []
#     lh = 0
#     rh = 0
#     for (c,_) in cnts:
#         (x, y, w, h) = cv2.boundingRect(c)
#         # print(x, y, w, h)
#         if 20 < h < 70 and 10 < w < 70:
#             # 排除0这类的数字可能会捕捉到两个边界的情况
#             if x > lh and x + w < rh:
#                 continue

#             lh = x
#             rh = x + w

#             # 这种情况应该是两个数字连在一起了，从中间切开
#             # TODO: 也有可能是三个数字连在一起了，这边暂时直接去掉，后面保存图片的时候会考虑 
#             if w > h:
#                 draw.rectangle((x-IMAGE_PADDING, y-IMAGE_PADDING, x + w // 2 + IMAGE_PADDING, y + h + IMAGE_PADDING), outline="Red")
#                 temp = subimage(imgry, (x-IMAGE_PADDING, y-IMAGE_PADDING, x + w // 2 + IMAGE_PADDING, y + h + IMAGE_PADDING))
#                 images.append(temp)

#                 draw.rectangle((x + w // 2 -IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING), outline="Red")
#                 temp = subimage(imgry, (x + w // 2 -IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING))
#                 images.append(temp)
#             else:
#                 draw.rectangle((x-IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING), outline="Red")
#                 temp = subimage(imgry, (x-IMAGE_PADDING, y-IMAGE_PADDING, x + w + IMAGE_PADDING, y + h + IMAGE_PADDING))
#                 images.append(temp)
#     # print(len(ary))
#     imgry.save(os.path.join(OUTPUT_FOLDER, random_name()), 'jpeg')
#     return images

# def subimage(imgry, box):
#     temp = imgry.crop(box)
#     temp = temp.resize((IMAGE_SIZE, IMAGE_SIZE))
#     # 加入了反色处理，这种情况好像容易分割一点
#     # 理论上不确定为什么，用验证码测试结果比之前好很多
#     temp = ImageOps.invert(temp)
#     return temp


def filter_img2(img, threshold):
    '对于黑白图像，将像素值大于某一个值的部分变成黑色，小于这个值的部分变成白色'
    filtered = Image.new(img.mode, img.size)
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
            if img.getpixel((x, y)) != threshold:
                filtered.putpixel((x, y), (0, 0, 0))
            else:
                filtered.putpixel((x, y), (255, 255, 255))
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
