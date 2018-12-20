import numpy as np

import gym
from gym import error, utils

from gym_vizdoom.envs.constants import ACTION_CLASSES
from gym_vizdoom.envs.register_games import GAMES

from gym import spaces

class VizdoomEnv(gym.Env):
    def __init__(self, game_name):

        self.game = GAMES[game_name]
        self.action_space = spaces.Discrete(ACTION_CLASSES)

        self.observation_space = spaces.Dict(self.game.init_space())
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
        frame = self.last_frame.astype(np.uint8)
        # if frame is (H, W, 3) okay else convert H,W,1 to H,W,3
        return frame if frame.shape[2] == 3 else np.tile(frame, (1,1,3))

