import os.path as osp
import numpy as np

DIR = osp.dirname(__file__)


MOVE_LEFT =  [1, 0, 0]
MOVE_RIGHT = [0, 1, 0]
ATTACK =     [0, 0, 1]
STAY_IDLE =  [0, 0, 0]


DEFAULT_CONFIG = osp.join(DIR, 'data', 'basic.cfg')


ACTIONS_LIST = [MOVE_LEFT, MOVE_RIGHT, ATTACK, STAY_IDLE]
ACTION_NAMES = ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK', 'STAY_IDLE']

ACTION_CLASSES = len(ACTIONS_LIST)

REPEAT = 4

VIZDOOM_TO_TF = [1, 2, 0]

DATA_PATH = 'data'

OBS_WIDTH = 160
OBS_HEIGHT = 120
OBS_CHANNELS = 3

STATE_SHAPE = [OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS]
NO_OBJECTIVE_SHAPE = [2, 2] # todo : depends on vocabulary size

STATE_AFTER_GAME_END = np.zeros((OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS), dtype=np.uint8)

