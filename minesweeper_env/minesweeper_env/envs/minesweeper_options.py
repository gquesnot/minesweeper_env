from dataclasses import dataclass

from minesweeper_env.envs.enums import *


@dataclass
class MinesweeperOptions:
    action: ActionType = ActionType.MULTI_DISCRETE
    obs: ObsType = ObsType.MULTI_DISCRETE
    rule_clicked_case: ClickedCaseType = ClickedCaseType.SOME_LOSS

    def to_json(self):
        return {
            "action": self.action,
            "obs": self.obs,
            "rule_clicked_case": self.rule_clicked_case
        }

    @classmethod
    def from_json(cls, json):
        return cls(**json)
