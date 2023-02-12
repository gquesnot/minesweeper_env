from enum import Enum


class PlayerType(str, Enum):
    HUMAN = "human"
    AI = "ai"
    RANDOM = "random"
