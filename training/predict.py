# coding = UTF8

"""
predict captcha in nctu portal
"""

import argparse
import sys
import os
from PIL import Image
from helper import pretreatment0, pretreatment1
import keras
import pickle
import numpy as np

MODELS = {}
MODEL_LABELS_FILE = os.path.join("model", "model_labels.dat")
LABEL = None

def load_models():
    global MODELS, LABEL
    for i in range(2):
        model_file = os.path.join("model", "captcha_model{}.hdf5".format(i))
        print(model_file)
        MODELS[i] = keras.models.load_model(model_file)

    with open(MODEL_LABELS_FILE, 'rb') as f:
        LABEL = pickle.load(f)

def predict(image, captcha_type=-1):
    """
    predict captcha in given image

    return:
        result if recognized successfully
        error message such as "ERROR: [error reason]"
    """
    if not MODELS:
        print("INFO: loading models...")
        load_models()

    # 尝试根据type识别四种验证码
    for t in range(2):
        print("INFO: trying type {} ...".format(t))
        if captcha_type in {-1, t}:
            # 图片预处理，得到分割好的四张resize之后的图片
            print("INFO: pretreatment...")
            pretreatment_method = eval("pretreatment{}".format(t))
            arr_image = pretreatment_method(image)

            if len(arr_image) == 4:
                print("INFO: predict...")
                results = []
                for im in arr_image:
                    im_arr = np.array(im)
                    im_arr = np.expand_dims(im_arr, axis=2)
                    im_arr = np.expand_dims(im_arr, axis=0)
                    predict_result = MODELS[t].predict(im_arr)
                    num = LABEL.inverse_transform(predict_result)[0]
                    print("INFO: predict: ", num)
                    results.append(num)
                return "".join(results)
                # predict_method = eval("predict{}".format(t))
                # return predict_method(arr_image)
    return "ERROR:can't split the captcha"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str,
                        help='Path to the captcha image.')
    parser.add_argument("--captcha_type", type=int,
                        default=-1,
                        help='captcha_type, value in {0, 1, 2, 3}')
    args = parser.parse_args()

    image = Image.open(args.image_path)
    result = predict(image, captcha_type=args.captcha_type)
    print(result)
