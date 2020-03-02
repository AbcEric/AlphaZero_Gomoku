#
# Gomoku对弈GUI：即可人工与AI，也可与训练模型对弈。（20200229）
#
# 1.采用alpha beta剪枝实现AI，通过max-min，打分较为合理；
# 2.pygame图形界面更友好，设计更合理。
#
# 参考：https://blog.csdn.net/marble_xu/article/details/90340389
#

from mytoolkit import print_time, load_config
import logging.config
# logging设置只能执行一次，要确保最先执行，在其它包含文件之前，否则包含文件会WARNING及以上才会记录。
logging.config.dictConfig(load_config('./conf/train_config.yaml')['train_logging'])
# 目的得到当前程序名，便于定位。
_logger = logging.getLogger(__name__)

import pygame, time, random
from pygame.locals import *
from enum import IntEnum
from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_keras import PolicyValueNet               # Keras

GAME_VERSION = 'V1.0'

AI_SEARCH_DEPTH = 4
AI_LIMITED_MOVE_NUM = 20

class CHESS_TYPE(IntEnum):
    NONE = 0,
    SLEEP_TWO = 1,
    LIVE_TWO = 2,
    SLEEP_THREE = 3
    LIVE_THREE = 4,
    CHONG_FOUR = 5,
    LIVE_FOUR = 6,
    LIVE_FIVE = 7,

CHESS_TYPE_NUM = 8

FIVE = CHESS_TYPE.LIVE_FIVE.value
FOUR, THREE, TWO = CHESS_TYPE.LIVE_FOUR.value, CHESS_TYPE.LIVE_THREE.value, CHESS_TYPE.LIVE_TWO.value
SFOUR, STHREE, STWO = CHESS_TYPE.CHONG_FOUR.value, CHESS_TYPE.SLEEP_THREE.value, CHESS_TYPE.SLEEP_TWO.value

SCORE_MAX = 0x7fffffff
SCORE_MIN = -1 * SCORE_MAX
SCORE_FIVE, SCORE_FOUR, SCORE_SFOUR = 100000, 10000, 1000
SCORE_THREE, SCORE_STHREE, SCORE_TWO, SCORE_STWO = 100, 10, 8, 2

REC_SIZE = 50
CHESS_RADIUS = REC_SIZE // 2 - 2
CHESS_LEN = 15
MAP_WIDTH = CHESS_LEN * REC_SIZE
MAP_HEIGHT = CHESS_LEN * REC_SIZE

INFO_WIDTH = 200
BUTTON_WIDTH = 140
BUTTON_HEIGHT = 50

SCREEN_WIDTH = MAP_WIDTH + INFO_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT


class MAP_ENTRY_TYPE(IntEnum):
    MAP_EMPTY = 0,
    MAP_PLAYER_ONE = 1,         # 对应白棋
    MAP_PLAYER_TWO = 2,
    MAP_NONE = 3,  # out of map range


class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = [[0 for x in range(self.width)] for y in range(self.height)]
        self.steps = []

    def reset(self):
        for y in range(self.height):
            for x in range(self.width):
                self.map[y][x] = 0
        self.steps = []

    def reverseTurn(self, turn):
        if turn == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
            return MAP_ENTRY_TYPE.MAP_PLAYER_TWO
        else:
            return MAP_ENTRY_TYPE.MAP_PLAYER_ONE

    def getMapUnitRect(self, x, y):
        map_x = x * REC_SIZE
        map_y = y * REC_SIZE

        return (map_x, map_y, REC_SIZE, REC_SIZE)

    def MapPosToIndex(self, map_x, map_y):
        x = map_x // REC_SIZE
        y = map_y // REC_SIZE
        return (x, y)

    def isInMap(self, map_x, map_y):
        if (map_x <= 0 or map_x >= MAP_WIDTH or
                map_y <= 0 or map_y >= MAP_HEIGHT):
            return False
        return True

    def isEmpty(self, x, y):
        return (self.map[y][x] == 0)

    def click(self, x, y, type):
        self.map[y][x] = type.value
        self.steps.append((x, y))

    def drawChess(self, screen):
        # player_one = (255, 251, 240)
        # player_two = (88, 87, 86)

        # 交换颜色：player1执黑先行
        player_one = (88, 87, 86)
        player_two = (255, 251, 240)

        player_color = [player_one, player_two]

        font = pygame.font.SysFont(None, REC_SIZE * 2 // 3)

        for i in range(len(self.steps)):
            x, y = self.steps[i]
            map_x, map_y, width, height = self.getMapUnitRect(x, y)
            pos, radius = (map_x + width // 2, map_y + height // 2), CHESS_RADIUS
            turn = self.map[y][x]
            if turn == 1:
                op_turn = 2
            else:
                op_turn = 1

            pygame.draw.circle(screen, player_color[turn - 1], pos, radius)

            msg_image = font.render(str(i), True, player_color[op_turn - 1], player_color[turn - 1])
            msg_image_rect = msg_image.get_rect()
            msg_image_rect.center = pos
            screen.blit(msg_image, msg_image_rect)

        if len(self.steps) > 0:
            last_pos = self.steps[-1]
            map_x, map_y, width, height = self.getMapUnitRect(last_pos[0], last_pos[1])
            purple_color = (255, 0, 255)
            point_list = [(map_x, map_y), (map_x + width, map_y),
                          (map_x + width, map_y + height), (map_x, map_y + height)]
            pygame.draw.lines(screen, purple_color, True, point_list, 1)

    def drawBackground(self, screen):
        color = (0, 0, 0)
        for y in range(self.height):
            # draw a horizontal line
            start_pos, end_pos = (REC_SIZE // 2, REC_SIZE // 2 + REC_SIZE * y), (
            MAP_WIDTH - REC_SIZE // 2, REC_SIZE // 2 + REC_SIZE * y)
            if y == (self.height) // 2:
                width = 2
            else:
                width = 1
            pygame.draw.line(screen, color, start_pos, end_pos, width)

        for x in range(self.width):
            # draw a horizontal line
            start_pos, end_pos = (REC_SIZE // 2 + REC_SIZE * x, REC_SIZE // 2), (
            REC_SIZE // 2 + REC_SIZE * x, MAP_HEIGHT - REC_SIZE // 2)
            if x == (self.width) // 2:
                width = 2
            else:
                width = 1
            pygame.draw.line(screen, color, start_pos, end_pos, width)

        rec_size = 8
        pos = [(3, 3), (11, 3), (3, 11), (11, 11), (7, 7)]
        for (x, y) in pos:
            pygame.draw.rect(screen, color, (
            REC_SIZE // 2 + x * REC_SIZE - rec_size // 2, REC_SIZE // 2 + y * REC_SIZE - rec_size // 2, rec_size,
            rec_size))


class ChessAI:
    def __init__(self, chess_len):
        self.len = chess_len
        # [horizon, vertical, left diagonal, right diagonal]
        self.record = [[[0, 0, 0, 0] for x in range(chess_len)] for y in range(chess_len)]
        self.count = [[0 for x in range(CHESS_TYPE_NUM)] for i in range(2)]

    def reset(self):
        for y in range(self.len):
            for x in range(self.len):
                for i in range(4):
                    self.record[y][x][i] = 0

        for i in range(len(self.count)):
            for j in range(len(self.count[0])):
                self.count[i][j] = 0

    def click(self, map, x, y, turn):
        map.click(x, y, turn)

    def isWin(self, board, turn):
        return self.evaluate(board, turn, True)

    # evaluate score of point, to improve pruning efficiency
    def evaluatePointScore(self, board, x, y, mine, opponent):
        dir_offset = [(1, 0), (0, 1), (1, 1), (1, -1)]  # direction from left to right
        for i in range(len(self.count)):
            for j in range(len(self.count[0])):
                self.count[i][j] = 0

        board[y][x] = mine
        self.evaluatePoint(board, x, y, mine, opponent, self.count[mine - 1])
        mine_count = self.count[mine - 1]
        board[y][x] = opponent
        self.evaluatePoint(board, x, y, opponent, mine, self.count[opponent - 1])
        opponent_count = self.count[opponent - 1]
        board[y][x] = 0

        mscore = self.getPointScore(mine_count)
        oscore = self.getPointScore(opponent_count)

        return (mscore, oscore)

    # check if has a none empty position in it's radius range
    def hasNeighbor(self, board, x, y, radius):
        start_x, end_x = (x - radius), (x + radius)
        start_y, end_y = (y - radius), (y + radius)

        for i in range(start_y, end_y + 1):
            for j in range(start_x, end_x + 1):
                if i >= 0 and i < self.len and j >= 0 and j < self.len:
                    if board[i][j] != 0:
                        return True
        return False

    # get all positions near chess
    def genmove(self, board, turn):
        fives = []
        mfours, ofours = [], []
        msfours, osfours = [], []
        if turn == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
            mine = 1
            opponent = 2
        else:
            mine = 2
            opponent = 1

        moves = []
        radius = 1

        for y in range(self.len):
            for x in range(self.len):
                if board[y][x] == 0 and self.hasNeighbor(board, x, y, radius):
                    mscore, oscore = self.evaluatePointScore(board, x, y, mine, opponent)
                    point = (max(mscore, oscore), x, y)

                    if mscore >= SCORE_FIVE or oscore >= SCORE_FIVE:
                        fives.append(point)
                    elif mscore >= SCORE_FOUR:
                        mfours.append(point)
                    elif oscore >= SCORE_FOUR:
                        ofours.append(point)
                    elif mscore >= SCORE_SFOUR:
                        msfours.append(point)
                    elif oscore >= SCORE_SFOUR:
                        osfours.append(point)

                    moves.append(point)

        if len(fives) > 0: return fives

        if len(mfours) > 0: return mfours

        if len(ofours) > 0:
            if len(msfours) == 0:
                return ofours
            else:
                return ofours + msfours

        moves.sort(reverse=True)

        # FIXME: decrease think time: only consider limited moves with higher scores
        if self.maxdepth > 2 and len(moves) > AI_LIMITED_MOVE_NUM:
            moves = moves[:AI_LIMITED_MOVE_NUM]
        return moves

    def __search(self, board, turn, depth, alpha=SCORE_MIN, beta=SCORE_MAX):
        score = self.evaluate(board, turn)
        if depth <= 0 or abs(score) >= SCORE_FIVE:
            return score

        moves = self.genmove(board, turn)
        bestmove = None
        self.alpha += len(moves)

        # if there are no moves, just return the score
        if len(moves) == 0:
            self.bestmove = (random.randint(6, 8), random.randint(6, 8))
            return score

        for _, x, y in moves:
            board[y][x] = turn

            if turn == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
                op_turn = MAP_ENTRY_TYPE.MAP_PLAYER_TWO
            else:
                op_turn = MAP_ENTRY_TYPE.MAP_PLAYER_ONE

            score = - self.__search(board, op_turn, depth - 1, -beta, -alpha)

            board[y][x] = 0
            self.belta += 1

            # alpha/beta pruning
            if score > alpha:
                alpha = score
                bestmove = (x, y)
                if alpha >= beta:
                    break

        if depth == self.maxdepth and bestmove:
            # print("bestmove = ", bestmove)
            self.bestmove = bestmove

        return alpha

    def search(self, board, turn, depth=4):
        self.maxdepth = depth
        self.bestmove = None
        score = self.__search(board, turn, depth)
        x, y = self.bestmove
        return score, x, y

    def findBestChess(self, board, turn):
        time1 = time.time()
        self.alpha = 0
        self.belta = 0
        score, x, y = self.search(board, turn, AI_SEARCH_DEPTH)
        time2 = time.time()
        print('time[%.2f] (%d, %d), score[%d] alpha[%d] belta[%d]' % (
        (time2 - time1), x, y, score, self.alpha, self.belta))
        return (x, y)

    def getPointScore(self, count):
        score = 0
        if count[FIVE] > 0:
            return SCORE_FIVE

        if count[FOUR] > 0:
            return SCORE_FOUR

        # FIXME: the score of one chong four and no live three should be low, set it to live three
        if count[SFOUR] > 1:
            score += count[SFOUR] * SCORE_SFOUR
        elif count[SFOUR] > 0 and count[THREE] > 0:
            score += count[SFOUR] * SCORE_SFOUR
        elif count[SFOUR] > 0:
            score += SCORE_THREE

        if count[THREE] > 1:
            score += 5 * SCORE_THREE
        elif count[THREE] > 0:
            score += SCORE_THREE

        if count[STHREE] > 0:
            score += count[STHREE] * SCORE_STHREE
        if count[TWO] > 0:
            score += count[TWO] * SCORE_TWO
        if count[STWO] > 0:
            score += count[STWO] * SCORE_STWO

        return score

    # calculate score, FIXME: May Be Improved
    def getScore(self, mine_count, opponent_count):
        mscore, oscore = 0, 0
        if mine_count[FIVE] > 0:
            return (SCORE_FIVE, 0)
        if opponent_count[FIVE] > 0:
            return (0, SCORE_FIVE)

        if mine_count[SFOUR] >= 2:
            mine_count[FOUR] += 1
        if opponent_count[SFOUR] >= 2:
            opponent_count[FOUR] += 1

        if mine_count[FOUR] > 0:
            return (9050, 0)
        if mine_count[SFOUR] > 0:
            return (9040, 0)

        if opponent_count[FOUR] > 0:
            return (0, 9030)
        if opponent_count[SFOUR] > 0 and opponent_count[THREE] > 0:
            return (0, 9020)

        if mine_count[THREE] > 0 and opponent_count[SFOUR] == 0:
            return (9010, 0)

        if (opponent_count[THREE] > 1 and mine_count[THREE] == 0 and mine_count[STHREE] == 0):
            return (0, 9000)

        if opponent_count[SFOUR] > 0:
            oscore += 400

        if mine_count[THREE] > 1:
            mscore += 500
        elif mine_count[THREE] > 0:
            mscore += 100

        if opponent_count[THREE] > 1:
            oscore += 2000
        elif opponent_count[THREE] > 0:
            oscore += 400

        if mine_count[STHREE] > 0:
            mscore += mine_count[STHREE] * 10
        if opponent_count[STHREE] > 0:
            oscore += opponent_count[STHREE] * 10

        if mine_count[TWO] > 0:
            mscore += mine_count[TWO] * 6
        if opponent_count[TWO] > 0:
            oscore += opponent_count[TWO] * 6

        if mine_count[STWO] > 0:
            mscore += mine_count[STWO] * 2
        if opponent_count[STWO] > 0:
            oscore += opponent_count[STWO] * 2

        return (mscore, oscore)

    def evaluate(self, board, turn, checkWin=False):
        self.reset()

        if turn == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
            mine = 1
            opponent = 2
        else:
            mine = 2
            opponent = 1

        for y in range(self.len):
            for x in range(self.len):
                if board[y][x] == mine:
                    self.evaluatePoint(board, x, y, mine, opponent)
                elif board[y][x] == opponent:
                    self.evaluatePoint(board, x, y, opponent, mine)

        mine_count = self.count[mine - 1]
        opponent_count = self.count[opponent - 1]
        if checkWin:
            return mine_count[FIVE] > 0
        else:
            mscore, oscore = self.getScore(mine_count, opponent_count)
            return (mscore - oscore)

    def evaluatePoint(self, board, x, y, mine, opponent, count=None):
        dir_offset = [(1, 0), (0, 1), (1, 1), (1, -1)]  # direction from left to right
        ignore_record = True
        if count is None:
            count = self.count[mine - 1]
            ignore_record = False
        for i in range(4):
            if self.record[y][x][i] == 0 or ignore_record:
                self.analysisLine(board, x, y, i, dir_offset[i], mine, opponent, count)

    # line is fixed len 9: XXXXMXXXX
    def getLine(self, board, x, y, dir_offset, mine, opponent):
        line = [0 for i in range(9)]

        tmp_x = x + (-5 * dir_offset[0])
        tmp_y = y + (-5 * dir_offset[1])
        for i in range(9):
            tmp_x += dir_offset[0]
            tmp_y += dir_offset[1]
            if (tmp_x < 0 or tmp_x >= self.len or
                    tmp_y < 0 or tmp_y >= self.len):
                line[i] = opponent  # set out of range as opponent chess
            else:
                line[i] = board[tmp_y][tmp_x]

        return line

    def analysisLine(self, board, x, y, dir_index, dir, mine, opponent, count):
        # record line range[left, right] as analysized
        def setRecord(self, x, y, left, right, dir_index, dir_offset):
            tmp_x = x + (-5 + left) * dir_offset[0]
            tmp_y = y + (-5 + left) * dir_offset[1]
            for i in range(left, right + 1):
                tmp_x += dir_offset[0]
                tmp_y += dir_offset[1]
                self.record[tmp_y][tmp_x][dir_index] = 1

        empty = MAP_ENTRY_TYPE.MAP_EMPTY.value
        left_idx, right_idx = 4, 4

        line = self.getLine(board, x, y, dir, mine, opponent)

        while right_idx < 8:
            if line[right_idx + 1] != mine:
                break
            right_idx += 1
        while left_idx > 0:
            if line[left_idx - 1] != mine:
                break
            left_idx -= 1

        left_range, right_range = left_idx, right_idx
        while right_range < 8:
            if line[right_range + 1] == opponent:
                break
            right_range += 1
        while left_range > 0:
            if line[left_range - 1] == opponent:
                break
            left_range -= 1

        chess_range = right_range - left_range + 1
        if chess_range < 5:
            setRecord(self, x, y, left_range, right_range, dir_index, dir)
            return CHESS_TYPE.NONE

        setRecord(self, x, y, left_idx, right_idx, dir_index, dir)

        m_range = right_idx - left_idx + 1

        # M:mine chess, P:opponent chess or out of range, X: empty
        if m_range >= 5:
            count[FIVE] += 1

        # Live Four : XMMMMX
        # Chong Four : XMMMMP, PMMMMX
        if m_range == 4:
            left_empty = right_empty = False
            if line[left_idx - 1] == empty:
                left_empty = True
            if line[right_idx + 1] == empty:
                right_empty = True
            if left_empty and right_empty:
                count[FOUR] += 1
            elif left_empty or right_empty:
                count[SFOUR] += 1

        # Chong Four : MXMMM, MMMXM, the two types can both exist
        # Live Three : XMMMXX, XXMMMX
        # Sleep Three : PMMMX, XMMMP, PXMMMXP
        if m_range == 3:
            left_empty = right_empty = False
            left_four = right_four = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:  # MXMMM
                    setRecord(self, x, y, left_idx - 2, left_idx - 1, dir_index, dir)
                    count[SFOUR] += 1
                    left_four = True
                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:  # MMMXM
                    setRecord(self, x, y, right_idx + 1, right_idx + 2, dir_index, dir)
                    count[SFOUR] += 1
                    right_four = True
                right_empty = True

            if left_four or right_four:
                pass
            elif left_empty and right_empty:
                if chess_range > 5:  # XMMMXX, XXMMMX
                    count[THREE] += 1
                else:  # PXMMMXP
                    count[STHREE] += 1
            elif left_empty or right_empty:  # PMMMX, XMMMP
                count[STHREE] += 1

        # Chong Four: MMXMM, only check right direction
        # Live Three: XMXMMX, XMMXMX the two types can both exist
        # Sleep Three: PMXMMX, XMXMMP, PMMXMX, XMMXMP
        # Live Two: XMMX
        # Sleep Two: PMMX, XMMP
        if m_range == 2:
            left_empty = right_empty = False
            left_three = right_three = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:
                    setRecord(self, x, y, left_idx - 2, left_idx - 1, dir_index, dir)
                    if line[left_idx - 3] == empty:
                        if line[right_idx + 1] == empty:  # XMXMMX
                            count[THREE] += 1
                        else:  # XMXMMP
                            count[STHREE] += 1
                        left_three = True
                    elif line[left_idx - 3] == opponent:  # PMXMMX
                        if line[right_idx + 1] == empty:
                            count[STHREE] += 1
                            left_three = True

                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:
                    if line[right_idx + 3] == mine:  # MMXMM
                        setRecord(self, x, y, right_idx + 1, right_idx + 2, dir_index, dir)
                        count[SFOUR] += 1
                        right_three = True
                    elif line[right_idx + 3] == empty:
                        # setRecord(self, x, y, right_idx+1, right_idx+2, dir_index, dir)
                        if left_empty:  # XMMXMX
                            count[THREE] += 1
                        else:  # PMMXMX
                            count[STHREE] += 1
                        right_three = True
                    elif left_empty:  # XMMXMP
                        count[STHREE] += 1
                        right_three = True

                right_empty = True

            if left_three or right_three:
                pass
            elif left_empty and right_empty:  # XMMX
                count[TWO] += 1
            elif left_empty or right_empty:  # PMMX, XMMP
                count[STWO] += 1

        # Live Two: XMXMX, XMXXMX only check right direction
        # Sleep Two: PMXMX, XMXMP
        if m_range == 1:
            left_empty = right_empty = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:
                    if line[left_idx - 3] == empty:
                        if line[right_idx + 1] == opponent:  # XMXMP
                            count[STWO] += 1
                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:
                    if line[right_idx + 3] == empty:
                        if left_empty:  # XMXMX
                            # setRecord(self, x, y, left_idx, right_idx+2, dir_index, dir)
                            count[TWO] += 1
                        else:  # PMXMX
                            count[STWO] += 1
                elif line[right_idx + 2] == empty:
                    if line[right_idx + 3] == mine and line[right_idx + 4] == empty:  # XMXXMX
                        count[TWO] += 1

        return CHESS_TYPE.NONE


class Button:
    def __init__(self, screen, text, x, y, color, enable):
        self.screen = screen
        self.width = BUTTON_WIDTH
        self.height = BUTTON_HEIGHT
        self.button_color = color
        self.text_color = (255, 255, 255)
        self.enable = enable
        self.font = pygame.font.SysFont(None, BUTTON_HEIGHT * 2 // 3)

        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.topleft = (x, y)
        self.text = text
        self.init_msg()

    def init_msg(self):
        if self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
        else:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
        self.msg_image_rect = self.msg_image.get_rect()
        self.msg_image_rect.center = self.rect.center

    def draw(self):
        if self.enable:
            self.screen.fill(self.button_color[0], self.rect)
        else:
            self.screen.fill(self.button_color[1], self.rect)
        self.screen.blit(self.msg_image, self.msg_image_rect)


class StartButton(Button):
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [(26, 173, 25), (158, 217, 157)], True)

    def click(self, game):
        if self.enable:
            if game.MODE in ["12", "21"]:
                mode = random.choice(["12", "21"])
                game.MODE = mode
                if mode == "12":
                    print("ChessAI is player1")
                else:
                    print("MCTS is player1")

            game.start()
            game.winner = None
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class GiveupButton(Button):
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [(230, 67, 64), (236, 139, 137)], False)

    def click(self, game):
        if self.enable:
            game.is_play = False
            if game.winner is None:
                game.winner = game.map.reverseTurn(game.player)
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class Game:
    def __init__(self, caption):
        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.buttons = []
        self.buttons.append(StartButton(self.screen, 'Start', MAP_WIDTH + 30, 15))
        self.buttons.append(GiveupButton(self.screen, 'Giveup', MAP_WIDTH + 30, BUTTON_HEIGHT + 45))
        self.is_play = False

        self.map = Map(CHESS_LEN, CHESS_LEN)
        self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
        self.action = None
        self.AI = ChessAI(CHESS_LEN)
        self.useAI = False
        self.winner = None

        self.MODE = "21"        # 0:Human 1:AI 2:MCTS
        # 载入自己训练的模型：
        self.useMCTS = False
        model_file = 'current_policy.model'
        conf = load_config('./conf/train_config.yaml')
        _logger.info("AI model: %s" % model_file)
        best_policy = PolicyValueNet(conf["board_width"], conf["board_height"], model_file)

        # 采用训练的AI模型作为对手：n_playout越大，水平越高，速度明显快很多！
        self.mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=600)

        self.board = Board(width=conf["board_width"], height=conf["board_height"], n_in_row=conf["n_in_row"])
        self.board.init_board(0)        # 均是player1先行
        self.play_steps = []

    # 按Start按钮开始新的一局：
    def start(self):
        self.is_play = True
        self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
        self.map.reset()
        self.board.init_board(0)

        if self.MODE == "12":
            _logger.info("AI is player1 ...")
            self.useAI = True
        elif self.MODE == "21":     # MCTS先行
            _logger.info("MCTS is player1 ...")
            self.useMCTS = True

    def play(self):
        self.clock.tick(60)

        light_yellow = (247, 238, 214)
        pygame.draw.rect(self.screen, light_yellow, pygame.Rect(0, 0, MAP_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(MAP_WIDTH, 0, INFO_WIDTH, SCREEN_HEIGHT))

        for button in self.buttons:
            button.draw()

        if self.is_play and not self.isOver():
            if self.useAI:
                # 当前该AI走棋：
                x, y = self.AI.findBestChess(self.map.map, self.player)

                self.checkClick(x, y, True) # Ture代表checkClick中不更新useAI状态
                self.useAI = False          # AI等待走棋
                if self.MODE == "12" or self.MODE == "21":  # AI vs MCTS
                    if not self.isOver():
                        self.useMCTS = True
                        # 马上执行下一个：
                        # print("turn to mcts ...")

            if self.useMCTS:
                # 采用自定义模型：
                # print("now mcts get_action ...")
                move = self.mcts_player.get_action(self.board)
                [x1, y1] = self.board.move_to_location(move)
                x, y = y1, 14 - x1
                print("AI response: [%d %d]-%d, original board(%d %d)" % (x, y, move, x1, y1))

                self.checkClick(x, y, True)
                self.useMCTS = False

                if self.MODE == "12" or self.MODE == "21":  # AI vs MCTS
                    if not self.isOver():
                        self.useAI = True

            if self.action is not None:
                # 鼠标按下：
                self.checkClick(self.action[0], self.action[1])
                self.action = None

            if not self.isOver():
                self.changeMouseShow()

        if self.isOver():
            self.showWinner()

        self.map.drawBackground(self.screen)
        self.map.drawChess(self.screen)

    def changeMouseShow(self):
        map_x, map_y = pygame.mouse.get_pos()
        x, y = self.map.MapPosToIndex(map_x, map_y)
        if self.map.isInMap(map_x, map_y) and self.map.isEmpty(x, y):
            pygame.mouse.set_visible(False)
            light_red = (213, 90, 107)
            pos, radius = (map_x, map_y), CHESS_RADIUS
            pygame.draw.circle(self.screen, light_red, pos, radius)
        else:
            pygame.mouse.set_visible(True)

    def checkClick(self, x, y, isAI=False):
        self.map.click(x, y, self.player)

        # ADD:
        move = self.board.location_to_move([14 - y, x])
        # print("you press: [%d %d]-%d, original board(%d %d)" % (x, y, move, 14 - y, x))
        # 采用自己训练的模型：
        if self.MODE == "12" or self.MODE == "21":
           self.board.do_move(move)

        self.play_steps.append(move)

        if self.AI.isWin(self.map.map, self.player):
            print("AI isWin: ", self.player)
            self.winner = self.player
            self.click_button(self.buttons[1])
        else:
            self.player = self.map.reverseTurn(self.player)
            if not isAI:
                self.useAI = True

    def mouseClick(self, map_x, map_y):
        if self.is_play and self.map.isInMap(map_x, map_y) and not self.isOver():
            x, y = self.map.MapPosToIndex(map_x, map_y)
            print("[%d, %d] -> [%d, %d]" % (map_x, map_y, x, y))

            if self.map.isEmpty(x, y):
                # 在play中调用上面的checkClick()
                self.action = (x, y)

    def isOver(self):
        return self.winner is not None

    def showWinner(self):
        def showFont(screen, text, location_x, locaiton_y, height):
            font = pygame.font.SysFont(None, height)
            font_image = font.render(text, True, (0, 0, 255), (255, 255, 255))
            font_image_rect = font_image.get_rect()
            font_image_rect.x = location_x
            font_image_rect.y = locaiton_y
            screen.blit(font_image, font_image_rect)

        if self.winner == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
            msg = 'Winner is Black - player1'
        else:
            msg = 'Winner is White - player2'

        # 当按giveup时：会循环不断显示，要判断是否为空
        if self.play_steps != []:
            _logger.warning("{'steps':[%s],'winner':%d,'start_player':1}" % (
                ",".join(map(str, self.play_steps)), self.winner))
            self.play_steps = []

        showFont(self.screen, msg, MAP_WIDTH + 25, SCREEN_HEIGHT - 60, 30)
        pygame.mouse.set_visible(True)

    def click_button(self, button):
        if button.click(self):
            for tmp in self.buttons:
                if tmp != button:
                    tmp.unclick()

    def check_buttons(self, mouse_x, mouse_y):
        for button in self.buttons:
            if button.rect.collidepoint(mouse_x, mouse_y):
                self.click_button(button)
                break


if __name__ == '__main__':

    game = Game("FIVE CHESS " + GAME_VERSION)

    while True:
        game.play()
        # 不断更新调用，更科学：
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                game.mouseClick(mouse_x, mouse_y)
                game.check_buttons(mouse_x, mouse_y)
