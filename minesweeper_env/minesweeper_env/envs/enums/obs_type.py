from enum import Enum


class ObsType(str, Enum):
    DISCRETE = "discrete"
    MULTI_DISCRETE = "multi_discrete"
    DICT = "dict"
