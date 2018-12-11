from gym_vizdoom.envs.basic_game_train import NoGoalBasicGameTrain, ColorBasicGameTrain

GAMES = {}

GAMES['VizdoomBasic'] = \
    NoGoalBasicGameTrain(initial_skip=14) # todo : to check

GAMES['VizdoomBasicColoredSimple'] = \
    ColorBasicGameTrain(initial_skip=14, mode="simple") # todo : to check

GAMES['VizdoomBasicColoredMedium'] = \
    ColorBasicGameTrain(initial_skip=14, mode="medium") # todo : to check