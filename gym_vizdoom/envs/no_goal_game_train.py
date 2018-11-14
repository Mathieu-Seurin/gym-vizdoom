from os import path as osp

# from gym_vizdoom.envs.constants import (DATA_PATH,
#                                         STATE_AFTER_GAME_END,
#                                         MAP_NAME_TEMPLATE,
#                                         MIN_RANDOM_TEXTURE_MAP_INDEX,
#                                         MAX_RANDOM_TEXTURE_MAP_INDEX)
# from gym_vizdoom.envs.util import real_get_frame

from gym_vizdoom.envs.no_goal_game import NoGoalGame


class NoGoalGameTrain(NoGoalGame):
  def __init__(self,
               dir,
               wad,
               initial_skip=0):
    super(NoGoalGameTrain, self).__init__(dir=dir, wad=wad, initial_skip=initial_skip)

  def class_specific_init(self):

    self.maps = ['map01']
    self.vizdoom_setup(self.wad)

    for map in self.maps:
      self.game.set_doom_map(map)
      self.game.new_episode()

    self.map_index = self.np_random.randint(0, len(self.maps))

  def class_specific_reset(self):
    self.map_index = self.np_random.randint(0, len(self.maps))

  def is_done(self):
    return self.game.is_episode_finished()

  def reward_shaping(self, reward):
    return reward

  def new_episode(self):
    self.game.set_doom_map(self.maps[self.map_index])
    self.game.new_episode()
