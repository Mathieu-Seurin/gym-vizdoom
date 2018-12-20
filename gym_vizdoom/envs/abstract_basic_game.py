from abc import ABC, abstractmethod
from os import path as osp
import numpy as np

from collections import OrderedDict

import time
import random

from gym.utils import seeding

from vizdoom import DoomGame, ViZDoomErrorException
import warnings

from gym_vizdoom.envs.constants import (DEFAULT_CONFIG,
                                        ACTIONS_LIST,
                                        REPEAT,
                                        DATA_PATH,
                                        STAY_IDLE,
                                        STATE_SHAPE,
                                        STATE_AFTER_GAME_END,
                                        MOVE_LEFT,
                                        MOVE_RIGHT)

from gym_vizdoom.envs.util import real_get_frame_rgb

class BasicGoalGame(ABC):
    # num_instance = 0

    def __init__(self,
                 dir,
                 wad,
                 initial_skip,
                 random_location=False):

        self.initial_skip = initial_skip // REPEAT
        self.random_spawn_location = random_location

        self.observation_shape = STATE_SHAPE
        self.objective_shape = None

        self.wad = osp.join(osp.dirname(__file__), DATA_PATH, dir, wad)

        self.just_started = True
        self.just_started_seed = None

        self.min_reward = -600

    def seed(self, seed):
        if seed is not None:
            if not self.just_started:
                self.game.set_seed(seed)
            else:
                self.just_started_seed = seed

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = self.make_action(action)
        done = self.is_done()
        state = self.get_state(done)
        info = vars(self).copy()
        info.pop('game', None) # infos for openai baselines need to be picklable, game is not

        if (not done) and self.game.get_episode_time() > self.time_out + REPEAT + self.initial_skip*REPEAT + 10:
            warnings.warn(
                "Timeout, something is wrong. \n Time is {} and Timeout is {}".format(self.game.get_episode_time(), self.time_out),
                stacklevel=4)
            #done = True

        #print("{} {} {}".format(self.game.get_episode_time(), self.step_counter, REPEAT))
        return state, self.reward_shaping(reward), done, info

    def reset(self):

        self.step_counter = 0
        self.reward_counter = 0

        if self.just_started:
            self.class_specific_init()
            self.vizdoom_setup(self.wad)
            if self.just_started_seed:
                self.game.set_seed(self.just_started_seed)
            self.just_started = False

        else:
            self.class_specific_reset()

        self.new_episode()

        if self.random_spawn_location:
            action = random.choice([MOVE_LEFT, MOVE_RIGHT])
            random_exec = random.randint(0, self.initial_skip)
            for _ in range(random_exec):
                self.game.make_action(action, REPEAT)
            for _ in range(self.initial_skip-random_exec):
                self.game.make_action(STAY_IDLE, REPEAT)

        else:
            for _ in range(self.initial_skip):
                self.stay_idle()

        state = self.get_state(done=False)
        return state

    def stay_idle(self):
        self.game.make_action(STAY_IDLE, REPEAT)

    def vizdoom_setup(self, wad, config=None):

        # print("Init vizdoom instance in this process {}".format(BasicGoalGame.num_instance))
        # BasicGoalGame.num_instance += 1
        game = DoomGame()
        config = DEFAULT_CONFIG if config is None else config
        game.load_config(config)
        game.set_doom_scenario_path(wad)

        # waiting = random.uniform(0, 5)
        # time.sleep(waiting)

        game.init()
        self.game = game
        self.time_out = self.game.get_episode_timeout()

    def make_action(self, action_index):

        reward = self.game.make_action(ACTIONS_LIST[action_index], REPEAT)
        self.step_counter += 1
        self.update_status()
        return reward

    def get_state(self, done):

        frame = self.get_frame(done)
        objective, sentence_length = self.get_objective()

        observation = OrderedDict()
        observation['state'] = frame
        observation['objective'] = objective
        observation['sentence_length'] = sentence_length

        return observation

    def get_frame(self, done):
          return real_get_frame_rgb(self.game) if not done else np.zeros(self.objective_shape, dtype=np.uint8)

    def update_status(self):
          pass

    @abstractmethod
    def reward_shaping(self, reward):
        pass

    @abstractmethod
    def class_specific_init(self):
        pass

    @abstractmethod
    def class_specific_reset(self):
        pass

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def new_episode(self):
        pass

    @abstractmethod
    def get_objective(self):
        pass

    @abstractmethod
    def init_space(self):
        pass