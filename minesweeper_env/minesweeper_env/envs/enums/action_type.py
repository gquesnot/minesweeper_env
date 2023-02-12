from enum import Enum


class ActionType(str, Enum):
    DISCRETE = "discrete"
    MULTI_DISCRETE = "multi_discrete"
