import sys
from functools import cache
from time import sleep
from typing import Tuple, Optional, Union, List, Any

import numpy as np
import pygame
from gym import Env, spaces
from gym.error import DependencyNotInstalled

from minesweeper_env.envs.minesweeper_options import MinesweeperOptions
from minesweeper_env.envs.enums import PlayerType, ActionType, ObsType, ClickedCaseType, BoxType, Reward


def auto_assign_attributes(self, **kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)


def _import_pygame():
    try:
        import pygame
    except ImportError:
        raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gym[toy_text]`"
        )


class MinesweeperEnv(Env):
    board: np.array
    mines: np.array
    action_mask: Optional[np.array] = None
    rng: np.random.Generator = np.random.default_rng()

    screen: Optional[pygame.Surface]

    pygame_init: bool = False
    completion: float = 0
    is_first_step: bool = True
    is_playing: bool = False
    max_episode_steps: int = 100
    flags: List[Tuple[int, int]] = []
    width: Optional[int]
    height: Optional[int]
    num_mines: Optional[int]
    options: MinesweeperOptions

    player_type: PlayerType = PlayerType.AI

    CELL_LENGTH: int = 40
    CELL_MID_FLOOR_LENGTH: int = 20

    def __init__(self, *args, **kwargs):
        self.options = MinesweeperOptions()
        auto_assign_attributes(self, **kwargs)
        if not hasattr(self, "width"):
            self.width = 8
        if not hasattr(self, "height"):
            self.height = 8
        if not hasattr(self, "num_mines"):
            self.num_mines = 10

        # print(
        #     "init env Minesweeper",
        #     f"width: {self.width}",
        #     f"height: {self.height}",
        #     f"num_mines: {self.num_mines}",
        #     f"num_cells: {self.get_nb_cells()}",
        #     f"action_type: {self.options.action}",
        #     f"obs_type: {self.options.obs}",
        #     f"case_type: {self.options.rule_clicked_case}",
        # )

        if self.options.action == ActionType.MULTI_DISCRETE:
            # x, y
            shape = (self.width, self.height)
            self.action_space = spaces.MultiDiscrete(shape)
        else:
            # index
            shape = (self.get_nb_cells(),)
            self.action_space = spaces.Discrete(self.get_nb_cells())

        board = spaces.Box(
            low=np.full(shape, fill_value=BoxType.MINE, dtype=int),
            high=np.full(shape, fill_value=9, dtype=int),
            shape=shape,
            dtype=int
        )
        if self.options.obs == ObsType.DISCRETE or self.options.obs == ObsType.MULTI_DISCRETE:
            self.observation_space = board
        elif self.options.obs == ObsType.DICT:
            self.observation_space = spaces.Dict({
                "board": board,
                "valid_actions": spaces.Box(
                    low=np.full(shape, fill_value=False, dtype=bool),
                    high=np.full(shape, fill_value=True, dtype=bool),
                    shape=shape,
                    dtype=bool
                )
            })

        self.screen = None

    # BASE gym.Env METHODS
    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        x, y = self._parse_action(action)
        terminated = False
        # first step is always safe
        cell_type = self._get_cell_type(x, y)
        reward = 0
        if self.is_first_step:
            if self._is_mine(x, y):
                self._reset_mines(x, y)
        if self._is_mine(x, y):
            self._set_case(x, y, BoxType.MINE)
            reward += Reward.LOSE
            terminated = True
        elif cell_type == BoxType.NOT_REVEALED:
            reward += Reward.YOLO if self._is_yolo(x, y) else Reward.PROGRESS
            adjacent_mines = self._set_adjacent_mines(x, y)
            if adjacent_mines == BoxType.EMPTY:
                self._reveal_neighbours(x, y)
            if self.has_won():
                reward += Reward.WIN
                terminated = True
        else:
            if self.options.rule_clicked_case == ClickedCaseType.SOME_LOSS or \
                    self.options.rule_clicked_case == ClickedCaseType.LOSE_THE_GAME_AND_SOME_LOSS:
                reward += Reward.NO_PROGRESS
            if self.options.rule_clicked_case == ClickedCaseType.LOSE_THE_GAME or \
                    self.options.rule_clicked_case == ClickedCaseType.LOSE_THE_GAME_AND_SOME_LOSS:
                terminated = True
        if terminated:
            self._reveal_all_mines()
        if self.is_first_step:
            self.is_first_step = False
        self.action_mask = self._get_valid_actions()

        return self._get_obs(), reward, terminated, self._get_info(terminated, reward)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        self.board = np.full(self._get_shape(), fill_value=BoxType.NOT_REVEALED, dtype=int)
        self.mines = self.rng.choice(self.get_nb_cells(), self.num_mines, replace=False)
        if self.options.obs == ObsType.DICT:
            self.action_mask = np.full(self._get_shape(), fill_value=True, dtype=bool)
        self._reset_mines()
        self.completion = 0
        self.is_first_step = True
        self.screen = None
        self.flags = []

        return self._get_obs()

    def render(self, mode="human"):
        if mode == "human":
            self._check_or_init_pygame()
            self._check_or_start_screen()
            self.screen.fill((255, 255, 255))
            for i in range(self.width):
                for j in range(self.height):
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 0),
                        (i * self.CELL_LENGTH, j * self.CELL_LENGTH, self.CELL_LENGTH, self.CELL_LENGTH),
                        1,
                    )
                    # check flag
                    if (i, j) in self.flags:
                        self._draw_flag(i, j)
                        continue
                    case = self._get_cell_type(i, j)
                    if case >= BoxType.EMPTY:
                        self._write_pygame_text(
                            str(case),
                            (0, 0, 0),
                            (i * self.CELL_LENGTH + self.CELL_MID_FLOOR_LENGTH,
                             j * self.CELL_LENGTH + self.CELL_MID_FLOOR_LENGTH),
                        )
                    elif case == BoxType.MINE:
                        pygame.draw.circle(
                            self.screen, (255, 0, 0), (i * self.CELL_LENGTH + self.CELL_MID_FLOOR_LENGTH,
                                                       j * self.CELL_LENGTH + self.CELL_MID_FLOOR_LENGTH), 8
                        )
            pygame.display.flip()
        else:
            return self.board

    def _get_info(self, terminated, reward) -> dict:
        if terminated:
            if reward == Reward.WIN:
                return {"win": True}
            elif reward == Reward.LOSE:
                return {"win": False}
        return {}

    def _get_obs(self) -> np.array:
        if self.options.obs == ObsType.DISCRETE or self.options.obs == ObsType.MULTI_DISCRETE:
            return self.board
        elif self.options.obs == ObsType.DICT:
            return {
                "board": self.board,
                "valid_actions": self.action_mask
            }
        else:
            raise ValueError("Invalid observation type")

    # UTILS for Gym.Env

    @cache
    def get_nb_cells(self) -> int:
        return self.width * self.height

    def _reset_mines(self, x: Optional[int] = None, y: Optional[int] = None):
        possible_indices = np.arange(self.get_nb_cells())
        if x is not None and y is not None:
            possible_indices = np.delete(possible_indices, self._get_cell_index(x, y))
        self.mines = self.rng.choice(possible_indices, self.num_mines, replace=False)

    def _get_coordinates(self, action: int) -> Tuple[int, int]:
        return action % self.width, action // self.width

    def _get_cell_index(self, x: int, y: int) -> int:
        return y * self.width + x

    def _get_shape(self) -> Union[Tuple[int, int], Tuple[int]]:
        return (self.width, self.height) if self.options.action == ActionType.MULTI_DISCRETE else (
            self.get_nb_cells(),)

    def _is_mine(self, x: int, y: int) -> bool:
        return self._get_cell_index(x, y) in self.mines

    def _is_yolo(self, x: int, y: int) -> bool:
        adjacent_indices = self._get_adjacent_cells_index(x, y)
        adjacent_cases_revealed = [
            1 for index in adjacent_indices if
            self._get_cell_type(*self._get_coordinates(index)) != BoxType.NOT_REVEALED
        ]
        return len(adjacent_cases_revealed) == 0

    def _parse_action(self, action: Any) -> tuple[int, int]:
        if self.options.action == ActionType.MULTI_DISCRETE:
            return int(action[0]), int(action[1])
        else:
            return self._get_coordinates(int(action))

    def _reveal_neighbours(self, x: int, y: int):
        reward = 0
        for index in self._get_adjacent_cells_index(x, y):
            x, y = self._get_coordinates(index)
            if self._get_cell_type(x, y) == BoxType.NOT_REVEALED:
                reward += 1
                adjacent_mines = self._set_adjacent_mines(x, y)
                if adjacent_mines == BoxType.EMPTY:
                    self._reveal_neighbours(x, y)

    def _set_adjacent_mines(self, x: int, y: int) -> int:

        adjacent_cells = self._get_adjacent_cells_index(x, y)
        adjacent_mines = int(np.sum(np.isin(adjacent_cells, self.mines)))
        self._set_case(x, y, adjacent_mines)
        return adjacent_mines

    def _toggle_flag(self, x: int, y: int):
        if (x, y) in self.flags:
            self.flags.remove((x, y))
        else:
            self.flags.append((x, y))

    def _get_adjacent_cells_index(self, x: int, y: int) -> List[int]:
        adjacent_coor = []
        for i in range(max(0, x - 1), min(self.width, x + 2)):
            for j in range(max(0, y - 1), min(self.height, y + 2)):
                if i != x or j != y:
                    adjacent_coor.append(self._get_cell_index(i, j))
        return adjacent_coor

    def _set_case(self, x: int, y: int, type_: int):
        if self.player_type == PlayerType.HUMAN:
            if (x, y) in self.flags:
                self.flags.remove((x, y))
        if self.options.action == ActionType.MULTI_DISCRETE:
            self.board[x, y] = type_
        else:
            self.board[self._get_cell_index(x, y)] = type_

    def _get_valid_actions(self) -> np.array:
        return np.isin(self.board, BoxType.NOT_REVEALED)

    def _get_cell_type(self, x: int, y: int) -> int:
        if self.options.action == ActionType.MULTI_DISCRETE:
            return self.board[x, y]
        else:
            return self.board[self._get_cell_index(x, y)]

    def _reveal_all_mines(self):
        if self.options.action == ActionType.MULTI_DISCRETE:
            for index in self.mines:
                x, y = self._get_coordinates(index)
                self._set_case(x, y, BoxType.MINE)
        else:
            self.board[self.mines] = BoxType.MINE

    def has_won(self) -> bool:
        if self.options.action == ActionType.MULTI_DISCRETE:
            return np.count_nonzero(self.board == BoxType.NOT_REVEALED) == self.num_mines
        else:
            nb_not_reveled = np.count_nonzero(self.board == BoxType.NOT_REVEALED)
        return nb_not_reveled == self.num_mines

    def close(self):
        if self.screen is not None or self.pygame_init:
            pygame.quit()

    def _update_completion(self):
        self.completion = np.sum(self.board >= BoxType.EMPTY) / (self.get_nb_cells())

    # PLAY GAME FUNCTION
    def set_player_type(self, player_type: PlayerType):
        self.player_type = player_type
        self.is_playing = True

    def play_has_human(self):
        self.reset()
        self._check_or_init_pygame()
        self._check_or_start_screen()
        self.render()
        self._pygame_loop()

    # PYGAME UTILS
    def _check_or_init_pygame(self):
        if not self.pygame_init:
            _import_pygame()
            self.pygame_init = True
            pygame.init()

    def _check_or_start_screen(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode(
                (self.width * self.CELL_LENGTH, self.height * self.CELL_LENGTH)
            )
            pygame.display.set_caption("Minesweeper")

    def _write_pygame_text(self, text: str, color: Tuple[int, int, int], position: Tuple[int, int], size: int = 20):
        text = pygame.font.SysFont("Arial", size).render(text, True, color)
        self.screen.blit(
            text,
            (
                position[0] - text.get_width() // 2,
                position[1] - text.get_height() // 2,
            ),
        )

    def _draw_flag(self, x: int, y: int):
        pygame.draw.rect(self.screen, (0, 0, 0), (
            x * self.CELL_LENGTH + self.CELL_LENGTH // 2,
            y * self.CELL_LENGTH + self.CELL_LENGTH // 6,
            self.CELL_LENGTH // 6,
            2 * self.CELL_LENGTH // 3
        ))
        # write base
        pygame.draw.rect(self.screen, (0, 0, 0), (
            x * self.CELL_LENGTH + self.CELL_LENGTH // 2 - self.CELL_LENGTH // 6,
            y * self.CELL_LENGTH + self.CELL_LENGTH // 6 + 2 * self.CELL_LENGTH // 3,
            1 * self.CELL_LENGTH // 2,
            self.CELL_LENGTH // 9
        ))
        pygame.draw.polygon(self.screen, (255, 0, 0), [
            (x * self.CELL_LENGTH + self.CELL_LENGTH // 6, y * self.CELL_LENGTH + self.CELL_LENGTH // 3),
            (x * self.CELL_LENGTH + self.CELL_LENGTH // 2, y * self.CELL_LENGTH + self.CELL_LENGTH // 6),
            (x * self.CELL_LENGTH + self.CELL_LENGTH // 2, y * self.CELL_LENGTH + self.CELL_LENGTH // 2),
        ])

    def _reset_pygame_after_done(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.reset()
                    self.render()
                    return

    def _pygame_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left click to reveal
                        x, y = event.pos
                        x //= self.CELL_LENGTH
                        y //= self.CELL_LENGTH
                        if self.options.action == ActionType.MULTI_DISCRETE:
                            action = x, y
                        else:
                            action = self._get_cell_index(x, y)
                        obs, rewards, done, info = self.step(action)
                        self.render()
                        if done:
                            sleep(1)
                            self.screen.fill((255, 0, 0) if not self.has_won() else (0, 255, 0))
                            self._write_pygame_text(
                                ("You won!" if rewards == Reward.WIN else "You lost!") + " Click to restart",
                                (255, 255, 255),
                                (self.width * self.CELL_LENGTH // 2, self.height * self.CELL_LENGTH // 2),
                            )
                            pygame.display.flip()
                            self._reset_pygame_after_done()
                    elif event.button == 3:
                        # right click => flag
                        x, y = event.pos
                        x //= self.CELL_LENGTH
                        y //= self.CELL_LENGTH
                        if self._get_cell_type(x, y) == BoxType.NOT_REVEALED:
                            self._toggle_flag(x, y)
                            self.render()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
