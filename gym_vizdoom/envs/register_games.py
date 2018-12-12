from gym_vizdoom.envs.basic_game_train import NoGoalBasicGameTrain, ColorBasicGameTrain

GAMES = {}

GAMES['VizdoomBasic'] = \
    NoGoalBasicGameTrain()

GAMES['VizdoomBasicColoredSimple'] = \
    ColorBasicGameTrain(mode="simple")

GAMES['VizdoomBasicColoredSimpleOneHot'] = \
    ColorBasicGameTrain(mode="simple", onehot=True)

GAMES['VizdoomBasicColoredMedium'] = \
    ColorBasicGameTrain(mode="medium")

GAMES['VizdoomBasicColoredMediumOneHot'] = \
    ColorBasicGameTrain(mode="medium", onehot=True)
