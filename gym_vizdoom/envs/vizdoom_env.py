import numpy as np

import gym
from gym import error, spaces, utils

from gym_vizdoom.envs.constants import ACTION_CLASSES
from gym_vizdoom.envs.register_games import GAMES

from collections import OrderedDict

class VizdoomEnv(gym.Env):
    def __init__(self, game_name):

        self.game = GAMES[game_name]
        self.action_space = spaces.Discrete(ACTION_CLASSES)

        # if using embedding instead of one-hot vector
        if len(self.game.objective_shape) == 1:
            objective_space = spaces.Box(low=0,
                                         high=self.game.objective_generator.voc_size,
                                         shape=self.game.objective_shape,
                                         dtype=np.int8)
        else:
            objective_space = spaces.Box(low=0,
                                         high=1,
                                         shape=self.game.objective_shape,
                                         dtype=np.int8)

        space = OrderedDict({
            "state": spaces.Box(0, 255, shape=self.game.observation_shape, dtype=np.int8),
            "objective": objective_space
        })

        dict_space = spaces.Dict(space)
        self.observation_space = dict_space
        self.seed()

        self.metadata = {'render.modes': ['rgb_array']}
        self.last_frame = np.zeros(shape=self.game.observation_shape)

    def seed(self, seed=None):
        return self.game.seed(seed)

    def step(self, action):
        observation, reward, done, info = self.game.step(action)
        self.last_frame = observation["state"]
        return observation, reward, done, info

    def reset(self):
        return self.game.reset()

    def render(self, mode='rgb_array'):
        return self.last_frame.astype(np.uint8)
