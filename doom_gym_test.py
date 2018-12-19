import random
import time
import argparse

from os import path


import gym
import gym_vizdoom
import gym.wrappers


import logging

game = gym.make("{}-v0".format("VizdoomBasicColoredSimplest"))
episodes = 10

game = gym.wrappers.Monitor(game, "test_out", resume=False, force=True)

for i in range(episodes):

    state = game.reset()
    done = False
    j = 0
    reward_total = 0
    while not done:
        action = random.randint(0,3) #int(input())
        observation, reward, done, info = game.step(action=action)
        j = j+1
        reward_total += reward
        #uprint(reward_total)
        if done:
            print(observation["objective"])
            print(observation["sentence_length"])
            break