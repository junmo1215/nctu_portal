# -*- coding: UTF-8 -*-
'''
下载验证码图片
文件名使用随机数
参数：
    NUM 本次需要下载的图片数量
'''

import sys
import os
import requests

# 下载图片数量
NUM = 1

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

if __name__ == "__main__":
    if len(sys.argv) == 2:
        NUM = int(sys.argv[1])
    main()
