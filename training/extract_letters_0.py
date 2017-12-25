# -*- coding: UTF-8 -*-

import os
from PIL import Image
import numpy as np
from helper import filter_img, random_name, cut_image

IMAGE_FOLDER = os.path.join("data", "labeled", "0", "test")
OUTPUT_FOLDER = os.path.join("single_letters", "0", "test")
IMAGE_SIZE = 32
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
            imgry = filter_img(im, 98)
            # imgry.save(os.path.join(OUTPUT_FOLDER, step + img_file), 'jpeg')
            imgs = cut_image(imgry)
            if len(imgs) != 4:
                print("cut image warning: {}".format(img_file))
                continue

            for i in range(4):
                imgs[i].resize(MODEL_INPUT_SIZE).save(os.path.join(OUTPUT_FOLDER, img_file[i], random_name()))


# def cut_image(imgry):
#     """
#     把图片切割成数字，图片必须是二值化之后的结果（像素值只有0或者255）
#     返回切割后的四张图片
#     """
#     img_arr = np.array(imgry)
#     # 黑色点的位置
#     dot_lists = np.argwhere(img_arr == 0)
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

#             # 让图像居中
#             padding_left = (IMAGE_SIZE - (r - l)) // 2
#             l = l - padding_left
#             r = l + IMAGE_SIZE

#             box = (l, top, r, bottom)
#             results.append(imgry.crop(box))

#     # print(box)
#     return results

if __name__ == "__main__":
    main()
