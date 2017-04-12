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
NUM = 50

def main():
    '主函数'
    dir_name = "unlabeled"
    for index in range(0, NUM):
        resp = requests.get("https://portal.nctu.edu.tw/captcha/pic.php?t=1491972085457",
                            verify=False)
        with open(os.path.join(dir_name, dir_name + str(index) + '.jpg'), 'wb') as f:
            f.write(resp.content)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        NUM = int(sys.argv[1])
    main()
