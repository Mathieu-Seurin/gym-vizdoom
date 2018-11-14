from abc import ABC, abstractmethod
from os import path as osp
import numpy as np

from gym.utils import seeding

from vizdoom import DoomGame

from gym_vizdoom.envs.constants import (DEFAULT_CONFIG,
                                        ACTIONS_LIST,
                                        REPEAT,
                                        DATA_PATH,
                                        STAY_IDLE,
                                        STATE_SIZE,
                                        STATE_AFTER_GAME_END)

from gym_vizdoom.envs.util import real_get_frame

class NoGoalGame(ABC):
  def __init__(self,
               dir,
               wad,
               initial_skip=0):

    self.initial_skip = initial_skip // REPEAT
    self.observation_shape = STATE_SIZE

    self.wad = osp.join(osp.dirname(__file__), DATA_PATH, dir, wad)
    self.just_started = True
    self.just_started_seed = None

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
    return state, self.reward_shaping(reward), done, info

  def reset(self):
    self.step_counter = 0

    if self.just_started:
      self.class_specific_init()
      self.vizdoom_setup(self.wad)
      if self.just_started_seed:
        self.game.set_seed(self.just_started_seed)
      self.just_started = False
    else:
      self.class_specific_reset()
    self.new_episode()
    for _ in range(self.initial_skip):
      self.stay_idle()
    state = self.get_state(done=False)
    return state

  def stay_idle(self):
    self.game.make_action(STAY_IDLE, REPEAT)

  def vizdoom_setup(self, wad, config=None):
    game = DoomGame()
    config = DEFAULT_CONFIG if config is None else config
    game.load_config(config)
    game.set_doom_scenario_path(wad)
    game.init()
    self.game = game

  def make_action(self, action_index):

    reward = self.game.make_action(ACTIONS_LIST[action_index], REPEAT)
    self.step_counter += 1
    self.update_status()
    return reward

  def get_state(self, done):
    frame = self.get_frame(done)
    return frame

  def get_frame(self, done):
      return real_get_frame(self.game) if not done else STATE_AFTER_GAME_END

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
