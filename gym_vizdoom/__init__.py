from gym.envs.registration import register
from gym_vizdoom.envs import GAME_NAMES

template = """
register(
  id='{}-v0',
  entry_point='gym_vizdoom.envs:{}',
)
"""

for game_name in GAME_NAMES:
  exec(template.format(game_name, game_name))

LIST_OF_ENVS = [name + '-v0' for name in GAME_NAMES]