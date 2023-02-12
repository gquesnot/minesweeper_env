from enum import Enum


class Reward(float, Enum):
    LOSE = -1
    WIN = 1
    PROGRESS = 0.9
    NO_PROGRESS = -0.3
    YOLO = -0.3
