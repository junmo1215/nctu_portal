# -*- coding: UTF-8 -*-
'''
下载验证码图片
文件名使用随机数
参数：
    NUM 本次需要下载的图片数量
'''

import sys
import os
import uuid
import requests

# 下载图片数量
NUM = 2

HEADS = {
"Accept": "image/webp,image/apng,image/*,*/*;q=0.8", 
"Accept-Encoding": "gzip, deflate, br", 
"Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8", 
"Cache-Control": "no-cache", 
"Connection": "keep-alive", 
"Cookie": "_ga=GA1.3.532806953.1472735297; _gid=GA1.3.821060795.1512569286; PHPSESSID=srnn8hmaepkjd9v0vm5kalkti3; __utma=156794426.532806953.1472735297.1512556804.1512569295.201; __utmc=156794426;__utmz=156794426.1475251422.22.9.utmcsr=it.nctu.edu.tw|utmccn=(referral)|utmcmd=referral|utmcct=/", 
"Host": "portal.nctu.edu.tw", 
"Pragma": "no-cache", 
"Referer": "https://portal.nctu.edu.tw/portal/login.php", 
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36"
}

def main():
    '主函数'
    dir_name = "unlabeled"
    for index in range(0, NUM):
        
        # resp = requests.get("https://portal.nctu.edu.tw/captcha/pic.php?t=1491972085457",
        #                     verify=False)
        resp = requests.get("https://portal.nctu.edu.tw/captcha/cool-php-captcha/pic.php", verify=False)
        with open(os.path.join(dir_name, dir_name + str(index) + '.jpg'), 'wb') as f:
            f.write(resp.content)

def download1():
    output_dir = os.path.join("data", "labeled", "1")
    for i in range(0, NUM):
        # 请求页面，拿到名称
        resp = requests.get("http://localhost:8081/simple-php-captcha-master/index.php")
        content = resp.text
        index = content.index("[code] =>")
        label = content[index+10: index+14]
        # 下载图片
        image = requests.get("http://localhost:8081/simple-php-captcha-master/simple-php-captcha.php?_CAPTCHA&t=0.12561100+1513940509", cookies=resp.cookies)
        with open(os.path.join(output_dir, "{}_{}.jpg".format(label, uuid.uuid4().hex)), 'wb') as f:
            f.write(image.content)

def download0():
    # 下载之前要修改这两个值
    php_session_id=""
    label = "0000"
    output_dir = os.path.join("data", "labeled", "0")
    cookies = {
        "PHPSESSID": php_session_id
    }

    for i in range(0, NUM):
        image = requests.get("https://portal.nctu.edu.tw/captcha/pitctest/pic.php", cookies=cookies, verify=False)
        with open(os.path.join(output_dir, "{}_{}.jpg".format(label, uuid.uuid4().hex)), 'wb') as f:
            f.write(image.content)

def download2():
    output_dir = os.path.join("data", "labeled", "2")
    for i in range(0, NUM):
        # 下载图片
        image = requests.get("http://localhost:8081/securimage-master/securimage_show.php")
        # print(resp.text)

        # 获取标记
        resp = requests.get("http://localhost:8081/securimage-master/test.php", cookies=image.cookies)
        label = resp.text.strip()
        # print(label)
        with open(os.path.join(output_dir, "{}_{}.jpg".format(label, uuid.uuid4().hex)), 'wb') as f:
            f.write(image.content)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        NUM = int(sys.argv[1])
    # main()
    # download1()
    download2()
