# -*- coding: UTF-8 -*-
'''
给下载下来的图片加上正确的标签作为文件名

运行label_img.py后在浏览器上打开localhost.5000
在每个图片后面输入正确的标签回车或者光标移开就可以实现标记了
标记后的图片会从unlabeled文件夹移动到data\labeled文件夹，相同标记的文件会加上随机后缀
'''

import os
from flask import Flask, render_template, request
from shutil import copyfile
import random
import logging

app = Flask(__name__, static_folder="unlabeled", static_url_path='/unlabeled')

@app.route('/')
def index():
    img_list = get_images()
    return render_template('index.html', img_list=img_list,
                           total=len(img_list))

def get_images():
    images = []
    index = 0
    for file in os.listdir("unlabeled"):
        if file.endswith(".jpg"):
            images.append({"index": index, "src": file})
            index += 1
    return images

@app.route('/label')
def label_img():
    file_name = request.args.get("fileName")
    value = request.args.get("value")
    path = os.path.join('unlabeled', file_name)
    while os.path.isfile(os.path.join('data', 'labeled', value + ".jpg")):
        value = value + '_' + str(random.randint(0, 1000))
    copyfile(path, os.path.join('data', 'labeled', value + ".jpg"))
    os.remove(path)
    app.logger.info('source_path: %s value: %s destination_path: %s' % (path, value, os.path.join('data', 'labeled', value + ".jpg")))
    return "OK"

if __name__ == '__main__':
    handler = logging.FileHandler('label_img.log')
    app.logger.addHandler(handler)
    handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    app.run(host='0.0.0.0', debug=True)
