from gym_vizdoom.envs.abstract_basic_game import BasicGoalGame
import numpy as np
from gym_vizdoom.envs.text_utils import TextObjectiveGenerator
from vizdoom import GameVariable

import warnings

class NoGoalBasicGameTrain(BasicGoalGame):

    def __init__(self, initial_skip=14):
        self.dir = "Basic"
        self.wad = "basic.wad"
        super(NoGoalBasicGameTrain, self).__init__(dir=self.dir, wad=self.wad, initial_skip=initial_skip)

        self.objective_shape = (2,2)

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

    def new_episode(self):
         self.game.set_doom_map(self.maps[self.map_index])
         self.game.new_episode()

    def get_objective(self):
         return np.random.random(self.objective_shape)


class ColorBasicGameTrain(BasicGoalGame):
    def __init__(self, initial_skip=14, mode="simple"):
        self.dir = "Basic"
        self.wad = "basic_color.wad"
        super(ColorBasicGameTrain, self).__init__(dir=self.dir, wad=self.wad, initial_skip=initial_skip)

        self.color_map = ["Blue", "Yellow", "Green", "Red"]
        self.objective_generator = TextObjectiveGenerator(env_specific_vocab=self.color_map)
        self.objective_shape = (self.objective_generator.voc_size, self.objective_generator.max_sentence_length)

        # Because you cannot retrieve string from Vizdoom, need to be set here, SIC.


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

        # color_pos is a shuffled version of color map, done in vizdoom.
        # Fixed per episode
        # self.color_pos[0] -> color of monster at extreme left
        # self.color_pos[3] -> color of monster at extreme right
        self.color_pos = [self.color_map[int(self.game.get_game_variable(var))]
                          for var in (GameVariable.USER2, GameVariable.USER3,
                                      GameVariable.USER4, GameVariable.USER5)]
        # Impossible to do differently because Doom doesn't deal well with list.

        self.current_objective = None

    def get_objective(self):

        if self.current_objective is None:
            index = int(self.game.get_game_variable(GameVariable.USER1))
            color = self.color_pos[index]

            self.current_objective = self.objective_generator.sample(color, index, self.color_pos)

        else:
            return self.current_objective
