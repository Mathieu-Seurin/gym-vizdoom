from gym_vizdoom.envs.constants import VIZDOOM_TO_TF
from PIL import Image
import numpy as np

def real_get_frame_rgb(game):
    raw_state = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
    processed_frame = np.array(Image.fromarray(raw_state).resize((128, 128)))
    return processed_frame

def real_get_frame_gray(game):
    raw_state = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
    processed_frame = np.array(Image.fromarray(raw_state).convert('L').resize((128, 128)))
    return np.expand_dims(processed_frame, axis=2)

def get_coordinates(game):
    return game.get_state().game_variables[:2]

def load_frames_from_lmp(game, lmp, skip):
    game.replay_episode(lmp)
    frames = []
    counter = 0
    while not game.is_episode_finished():
      frame = real_get_frame_rgb(game)
      if counter % skip == 0:
        frames.append(frame)
      game.advance_action()
      counter += 1
    return frames

def load_goal_frame_from_lmp(game, lmp):
    frames = load_frames_from_lmp(game, lmp, 1)
    return frames[-1]
