from enum import Enum


class ClickedCaseType(str, Enum):
    NOTHING = "nothing"
    SOME_LOSS = "some_loss"
    LOSE_THE_GAME = "lose_the_game"
    LOSE_THE_GAME_AND_SOME_LOSS = "lose_the_game_and_some_loss"
