import numpy as np
import warnings

from vizdoom import GameVariable

from gym_vizdoom.envs.abstract_basic_game import BasicGoalGame
from gym_vizdoom.envs.text_utils import TextObjectiveGenerator
from gym_vizdoom.envs.constants import INITIAL_SKIP

from gym import spaces

from collections import OrderedDict
from gym_vizdoom.envs.util import real_get_frame_gray


class NoGoalBasicGameTrain(BasicGoalGame):

    def __init__(self):
        self.dir = "Basic"
        self.wad = "basic.wad"

        super(NoGoalBasicGameTrain, self).__init__(dir=self.dir, wad=self.wad, initial_skip=INITIAL_SKIP)

        # Override cus you don't need color
        self.observation_shape = (128,128,1)

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
        self.reward_counter += reward

        if self.reward_counter < self.min_reward:
            warnings.warn(
                "Rewards are wrong \n Total rewards are {} and min is {}".format(self.reward_counter,
                                                                                 self.min_reward),
                stacklevel=4)

        return reward

    def new_episode(self):
         self.game.set_doom_map(self.maps[self.map_index])
         self.game.new_episode()

    def get_frame(self, done):
        return real_get_frame_gray(self.game) if not done else np.zeros(self.observation_shape, dtype=np.uint8)

    def get_objective(self):
         return 0, 0

    def init_space(self):

        return OrderedDict({
            "state": spaces.Box(0, 255, shape=self.observation_shape, dtype=np.int8),
            "objective" : spaces.Box(0, 1, shape=tuple([1]), dtype=np.int8),
            "sentence_length" : spaces.Box(0, 1, shape=tuple([1]), dtype=np.int8)
        })

class ColorBasicGameTrain(BasicGoalGame):
    def __init__(self, mode="simple"):
        self.dir = "Basic"
        self.wad = "basic_color.wad"
        super(ColorBasicGameTrain, self).__init__(dir=self.dir, wad=self.wad, initial_skip=INITIAL_SKIP)

        # Because you cannot retrieve string from Vizdoom, need to be set here, SIC.
        self.color_map = ["Blue", "Yellow", "Green", "Red"]

        self.objective_generator = TextObjectiveGenerator(env_specific_vocab=self.color_map)

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
        return reward/100

    def new_episode(self):
        self.game.set_doom_map(self.maps[self.map_index])
        self.game.new_episode()

        # color_pos is a shuffled version of color map, done in vizdoom.
        # Fixed per episode
        # self.color_pos[0] -> color of monster at extreme left
        # self.color_pos[3] -> color of monster at extreme right
        self.color_pos = [self.color_map[int(self.game.get_game_variable(var))]
                          for var in (GameVariable.USER2, GameVariable.USER3,
                                      GameVariable.USER4, GameVariable.USER5)]
        # Impossible to do differently because Doom doesn't deal well with list.

        self.current_objective = None
        self.sentence_length = None

    def get_objective(self):

        if self.current_objective is None:
            index = int(self.game.get_game_variable(GameVariable.USER1))
            color = self.color_pos[index]

            self.current_objective, self.sentence_length = self.objective_generator.sample(color, index, self.color_pos)

        return self.current_objective, self.sentence_length

    def init_space(self):

        objective_space = spaces.Box(low=0,
                                     high=self.objective_generator.voc_size,
                                     shape=tuple([self.objective_generator.max_sentence_length]),
                                     dtype=np.int8)

        space = OrderedDict({
            "state": spaces.Box(0, 255, shape=self.observation_shape, dtype=np.int8),
            "objective": objective_space,
            "sentence_length" : spaces.Box(low=0,
                                           high=100,
                                           shape=tuple([1]),
                                           dtype=np.int8)
        })

        return space

class SimplestColorBasicGameTrain(BasicGoalGame):
    def __init__(self):
        self.dir = "Basic"
        self.wad = "basic_color.wad"
        super(SimplestColorBasicGameTrain, self).__init__(dir=self.dir, wad=self.wad, initial_skip=INITIAL_SKIP)

        # Because you cannot retrieve string from Vizdoom, need to be set here, SIC.
        self.color_map = ["Blue", "Yellow"]

    def class_specific_init(self):

        self.maps = ['map02']
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
        return reward/100

    def new_episode(self):
        self.game.set_doom_map(self.maps[self.map_index])
        self.game.new_episode()

        self.color_pos = [self.color_map[int(self.game.get_game_variable(var))]
                          for var in (GameVariable.USER2, GameVariable.USER3)]

        self.current_objective = None
        self.sentence_length = None

    def get_objective(self):

        if self.current_objective is None:
            index = int(self.game.get_game_variable(GameVariable.USER1))

        sentence_length = 1
        return self.current_objective, sentence_length

    def init_space(self):

        return OrderedDict({
            "state": spaces.Box(0, 255, shape=self.observation_shape, dtype=np.int8),
            "objective": spaces.Box(low=0,
                                    high=1,
                                    shape=tuple([1]),
                                    dtype=np.int8),
            "sentence_length" : spaces.Box(0, 1, shape=tuple([1]), dtype=np.int8)
        })
