from gym_vizdoom.envs.basic_game_train import NoGoalBasicGameTrain, ColorBasicGameTrain, SimplestColorBasicGameTrain

GAMES = {}

GAMES['VizdoomBasic'] = \
    NoGoalBasicGameTrain()

# GAMES['VizdoomBasicColoredSimple'] = \
#     ColorBasicGameTrain(mode="simple")
#
# GAMES['VizdoomBasicColoredMedium'] = \
#     ColorBasicGameTrain(mode="medium")
#
# GAMES['VizdoomBasicColoredSimplest'] = \
#     SimplestColorBasicGameTrain()
