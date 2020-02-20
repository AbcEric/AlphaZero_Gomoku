#
# 采用Yixin Engine作为联系对手进行对局：
#
# @author: Eric Li
# 与Gomocup Protoc协议兼容，通过管道Pipe与AI引擎进行通讯，协议可参考：http://petr.lastovicka.sweb.cz/protocl2en.htm
#

from subprocess import *
import time, string
import threading

# shell=True,  bufsize=10
p = Popen('/Program Files/Yixin/engine.exe', stdin=PIPE, stdout=PIPE)


# 接收返回结果的线程：
def get_yixin_answer():
    global p
    while True:
        line = str(p.stdout.readline().decode("GBK").strip())
        if not line:                # 空则跳出
            break
        print(">>>>>>", line)
        if not line.startswith("MESS"):
            print("not MESS")
        else:
            print("MESS")


    print("look up!!! EXIT ===")    # 跳出


# 发送命令：
def send_command(command=""):
    if not command.endswith("\r\n"):
        command = command + "\r\n"                  # 必须要有，代表一行的命令结束。

    p.stdin.write(command.encode('GBK'))  # 编码格式与系统有关？这里若为UTF-8，显示中文不正常。
    p.stdin.flush()  # 把输入的命令强制输出
    time.sleep(1)
    return


# 启动线程
w = threading.Thread(target=get_yixin_answer)
w.start()

# cmd = "START 15\r\n"                  # 必须要有，代表一行的命令结束。
send_command("START 15")
time.sleep(2)
print("start 15 end ...")
send_command("INFO timeout_turn 1000")  # 每步思考时间：最长1秒，时间越长水平越高。
time.sleep(2)
print("info end ...")

# send_command("BEGIN")
#
send_command("TURN 8, 8")
time.sleep(2)
print("turn end ...")



while 1:
    s = input("Please input: ")
    print(s)
    send_command("TURN " + s)


# for line in iter(subp.stdout.readline, ''):
# for line in iter(subp.stdout.read()):
#     print("..")
#     print(line, end='')

