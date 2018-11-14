import gym
from gym_vizdoom import LIST_OF_ENVS
from gym_vizdoom.logging.navigation_video_writer import SingleGoalVideoWriter

def split_current_goal(observation):
  c = observation.shape[2] // 2
  current = observation[..., :c]
  goal = observation[..., c:]
  return current, goal

def test(env, video_writer, number_of_episodes):
  for _ in range(number_of_episodes):
    observation = env.reset()
    video_writer.write(observation)
    step = 0
    while True:
      step += 1
      #action = env.action_space.sample()
      action = 0
      observation, reward, done, info = env.step(action)
      video_writer.write(observation)
      print('step:', step)
      print('reward:', reward)
      # print('Goal reached?', reward == GOAL_REACHING_REWARD)
      # print('status:', 'exploration' if (goal == EXPLORATION_GOAL_FRAME).all() else 'navigation')
      if done:
        print('Episode finished!')
        break

def main():
  print('All possible env names:', LIST_OF_ENVS)
  just_started = True
  for env_name in LIST_OF_ENVS:
    print('Testing env: {}'.format(env_name))
    env = gym.make(env_name)
    if just_started:
      video_writer = SingleGoalVideoWriter('output.mov',
                                           env.observation_space.shape)
      just_started = False
    test(env, video_writer, 1)

if __name__ == '__main__':
  main()
