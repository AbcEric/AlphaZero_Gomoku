
import numpy as np
import logging, time, random
_logger = logging.getLogger(__name__)       # 目的是得到当前文件名


class YixinPlayer(object):
    """
    Yixin AI player
    """

    def __init__(self, pEngine, qResponses, timeout=1000):
        self.player = None
        self.first = True
        self.steps = 0
        self.p = pEngine
        self.q_responses = qResponses
        self.timeout = timeout                  # 时间越长，水平越高。
        # 初始化：
        self.send_command("START 15")
        self.send_command("INFO timeout_turn %d" % self.timeout)  # 每步思考时间：最长1秒，时间越长水平越高。

    # time.sleep(1)

    def set_player_ind(self, p):
        self.player = p

    def send_command(self, cmd=""):
        if not cmd.endswith("\r\n"):
            cmd = cmd + "\r\n"  # 必须要有，代表一行的命令结束。

        self.p.stdin.write(cmd.encode('GBK'))  # 编码格式与系统有关？这里若为UTF-8，显示中文不正常。
        self.p.stdin.flush()  # 把输入的命令强制输出

        # rand = random.randint(1, 3)
        #
        # if cmd.startswith("TURN"):
        #     if random.random() > 0.5:
        #         cmd = "info nbestsym %d\r\n" % rand
        #         _logger.debug("cmd=%s" % cmd)
        #         self.p.stdin.write(cmd.encode('GBK'))
        #         self.p.stdin.flush()

        return

    def get_aimove(self):
        aimove = self.q_responses.get()
        _logger.debug("q_responses=%s" % aimove)
        x = aimove.split(",")

        # 添加正则检验：
        return int(x[0]), int(x[1])

    def get_action(self, board):
        try:
            if board.last_move == -1:
                # 先走：
                _logger.debug("player2(AI) first ...")
                # self.send_command("START 15")
                # # time.sleep(2)
                # self.send_command("INFO timeout_turn %d" % self.timeout)  # 每步思考时间：最长1秒，时间越长水平越高。
                # # time.sleep(1)
                self.send_command("BEGIN")
                x, y = self.get_aimove()

                _logger.debug("AI的第一步走棋 = %d,%d" % (x, y))
                move = board.location_to_move([x, y])
                self.first = False
                self.steps += 1
            else:
                _logger.debug("Last_move=%d" % board.last_move)
                x1 = int(board.last_move/15)
                y1 = board.last_move % 15

                if self.first:
                    _logger.debug("Player1 first (第一步） ... ")
                    # time.sleep(2)
                    self.send_command("START 15")
                    self.send_command("INFO timeout_turn %d" % self.timeout)  # 每步思考时间：最长1秒，时间越长水平越高。
                    self.first = False

                self.send_command("TURN %d, %d" % (x1, y1))
                x2, y2 = self.get_aimove()
                _logger.debug("AI对(%d)的应对 = %d,%d" % (board.last_move, x2, y2))
                self.steps += 1

                # 开始的10步，每一步有一半机率回退：
                if self.steps < 10 and random.random() > 0.5:
                    self.send_command("TAKEBACK %d,%d" % (x2, y2))
                    self.send_command("TAKEBACK %d,%d" % (x1, y1))
                    _logger.debug("undo [%d,%d] and [%d,%d], redo the last one." % (x2, y2, x1, y1))
                    self.send_command("TURN %d,%d" % (x1, y1))
                    x2, y2 = self.get_aimove()
                    _logger.debug("new move = [%d, %d]" % (x2, y2))         # 有些重走后也没改变

                move = board.location_to_move([x2, y2])

        except Exception as e:
            move = -1

        if move == -1 or move not in board.availables:
            _logger.info("invalid move: %d" % move)
            move = self.get_action(board)

        return move

    def __str__(self):
        return "Yixin {}".format(self.player)


def monitor_yixin_response(p, q_responses):
    _logger.info("Now monitor Yinxin's answer ....")
    while True:
        line = p.stdout.readline().decode("GBK").strip()
        if not line:  # 空则跳出
            break
        _logger.debug(">>>>>> %s" % line)
        if line.startswith("MESS"):
            continue
        elif line.startswith("OK"):
            continue
        elif line.startswith("ERROR"):
            _logger.error("Please check: %s" % line)
            exit(0)
        else:
            # 只有返回棋子位置才发送到pipe:
            q_responses.put(line)

    _logger.error("EXIT === Monitor child_process exit, please cheak !!!!!")  # 跳出



