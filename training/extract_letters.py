# coding = UTF8

"""
将验证码图片切割成可以喂给神经网络的固定大小的四张图片
"""

import os
import argparse
from helper import pretreatment0, pretreatment1, pretreatment2, random_name
from PIL import Image

IMAGE_SIZE = 32
MODEL_INPUT_SIZE = (28, 28)

def main():
    for i in range(10):
        dir_path = os.path.join(OUTPUT_FOLDER, str(i))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    count = 0
    success = 0
    total = len(os.listdir(IMAGE_FOLDER))
    pretreatment_method = eval("pretreatment{}".format(IMAGE_TYPE))
    for img_file in os.listdir(IMAGE_FOLDER):
        if img_file.endswith('.jpg'):
            if count % 200 == 0:
                print("done: {} / {}, success: {}".format(count, total, success))
            count += 1

            # print(img_file)
            im = Image.open(os.path.join(IMAGE_FOLDER, img_file))
            imgs = pretreatment_method(im, is_fake_img=True)
            if len(imgs) != 4:
                print("cut image warning: {}".format(img_file))
                continue

            success += 1
            for i in range(4):
                imgs[i].save(os.path.join(OUTPUT_FOLDER, img_file[i], random_name()))
    print("done: {} / {}, success: {}".format(count, total, success))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("captcha_type", type=int,
                        default=-1,
                        help='captcha_type, value in {0, 1, 2, 3}')
    args = parser.parse_args()

    IMAGE_TYPE = args.captcha_type

    IMAGE_FOLDER = os.path.join("data", "labeled", str(IMAGE_TYPE))
    OUTPUT_FOLDER = os.path.join("single_letters", str(IMAGE_TYPE))
    main()
