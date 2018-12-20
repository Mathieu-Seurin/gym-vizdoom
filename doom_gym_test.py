import random
import time
import argparse

from os import path


import gym
import gym_vizdoom
import gym.wrappers

import numpy as np

import logging

game = gym.make("{}-v0".format("VizdoomBasic"))
game.render(True)
episodes = 100

game = gym.wrappers.Monitor(game, "test_out", resume=False, force=True)

mean = 0
rew_list = []

for i in range(1, episodes):

    state = game.reset()
    done = False
    j = 0
    reward_total = 0
    while not done:
        action = random.randint(0,1) #int(input())
        observation, reward, done, info = game.step(action=action)
        j = j+1
        reward_total += reward
        if done:
            print(reward_total)
            rew_list.append(reward_total)

            if i == 0 :
                mean = reward_total
            else:
                mean = mean + reward_total/episodes
            break


print(mean)
print(np.mean(rew_list))