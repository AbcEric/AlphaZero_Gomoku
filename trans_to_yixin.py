#
# 将对弈过程转换为Yixin格式：
#
# 输入：在train_config.yaml中"play_steps"中记录
# 输出：desktop/yixin.sav
#

from mytoolkit import print_time, load_config
import logging.config
# logging设置只能执行一次，要确保最先执行，在其它包含文件之前，否则包含文件会WARNING及以上才会记录。
# logging.config.dictConfig(load_config('./conf/train_config.yaml')['train_logging'])
# 目的得到当前程序名，便于定位。
# _logger = logging.getLogger(__name__)

import time


# 生成Yinxin对弈格式复盘：
# play_steps= 112 113 143 142 127 128 114 159 82 98 67 97 81 95 96 110 66 111 83 80 84 85 51 99 36  winner= 1
def transfer_to_yixin_format(play_steps):
    steps = play_steps.split(",")
    fp = open("C:/Users/Eric/Desktop/yixin.sav", "w")
    fp.writelines("15\n")
    fp.writelines("15\n")
    fp.writelines("%d\n" % len(steps))
    for one in steps:
        x = int(one)/15
        y = int(one)%15
        fp.write("%d %d\n" % (y, x))
    fp.close()
    return

if __name__ == '__main__':
    conf = load_config('./conf/train_config.yaml')
    print("play_steps = ", conf["play_steps"])

    transfer_to_yixin_format(conf["play_steps"])
    print("对应的Yixin格式文件已生成：desktop/yixin.save")
    input("press any key to quit ...")

    exit(0)
