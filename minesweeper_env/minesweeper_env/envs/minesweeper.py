import sys
from enum import IntEnum, auto, Enum
from functools import cache
from threading import Thread
from time import sleep
from typing import Tuple, Optional, Union, List, Set, Any

import numpy as np
import pygame
from gym import Env, spaces
from gym.error import DependencyNotInstalled
from utils.attributes_helpers import auto_assign_attributes


class BoxType(IntEnum):
    MINE = -2
    NOT_REVEALED = -1
    EMPTY = 0


class PlayerType(str, Enum):
    HUMAN = "human"
    AI = "ai"
    RANDOM = "random"


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
    screen: Optional[pygame.Surface]
    pygame_init: bool = False

    width: Optional[int]
    height: Optional[int]
    num_mines: Optional[int]
    completion: float = 0
    is_first_step: bool = True
    player_type: PlayerType = PlayerType.AI

    CELL_LENGTH: int = 40
    CELL_MID_FLOOR_LENGTH: int = 20

    def __init__(self, *args, **kwargs):
        auto_assign_attributes(self, **kwargs)
        if not hasattr(self, "width"):
            self.width = 10
        if not hasattr(self, "height"):
            self.height = 10
        if not hasattr(self, "num_mines"):
            self.num_mines = 16

        print(
            "init env Minesweeper\n",
            f"width: {self.width}\n",
            f"height: {self.height}\n",
            f"num_mines: {self.num_mines}\n",
            f"num_cells: {self.width * self.height}\n",

        )
        self.observation_space = spaces.Box(
            low=BoxType.MINE, high=9, shape=(self.width * self.height,), dtype=int
        )
        # MultiDiscrete is not supported by DQN Stable Baselines3
        # self.action_space = spaces.MultiDiscrete([self.width, self.height])
        self.action_space = spaces.Discrete(self.width * self.height)
        self.screen = None

    # BASE gym.Env METHODS
    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        # for multi discrete
        # x, y = self._parse_action(action)
        x, y = self._get_coordinates(action)
        index = self._get_cell_index(x, y)

        reward = 0
        terminated = False

        # first step is always safe
        if self.is_first_step:
            if np.isin(index, self.mines):
                self.mines = np.random.choice(
                    np.delete(np.arange(self.width * self.height), index),
                    self.num_mines,
                    replace=False
                )
            self.is_first_step = False

        if np.isin(index, self.mines):
            self.board[self.mines] = BoxType.MINE
            reward -= 1.0
            terminated = True
        elif self.board[index] == BoxType.NOT_REVEALED:
            reward += 0.1
            self.board[index] = self._get_adjacent_mines(x, y)
            if self.board[index] == BoxType.EMPTY:
                reward += round(self._reveal_neighbours(x, y), 1)
            if self.has_won():
                reward += 1.0
                terminated = True
        else:
            # reward -= 0.1
            # or
            # pass
            # or
            if self.player_type == PlayerType.AI:
                terminated = True
        # self._update_completion()
        return self._get_obs(), reward, terminated, self._get_info()

    def reset(self):
        self.board = np.full((self.height * self.width,), fill_value=BoxType.NOT_REVEALED, dtype=int)
        self.mines = np.random.choice(self.width * self.height, self.num_mines, replace=False)
        self.completion = 0
        self.is_first_step = True
        # if self.pygame_init and self.screen:
        #     pygame.display.quit()

        self.screen = None
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
                    case = self.board[self._get_cell_index(i, j)]
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

    def _get_info(self) -> dict:
        return {
            # "completion": self.completion
        }

    def _get_obs(self) -> np.array:
        return self.board

    # UTILS for Gym.Env
    @cache
    def _get_coordinates(self, action: int) -> Tuple[int, int]:
        return action % self.width, action // self.width

    @cache
    def _get_cell_index(self, x: int, y: int) -> int:
        return y * self.width + x

    def _parse_action(self, action: Any) -> Tuple[int, int]:
        return int(action[0]), int(action[1])

    def _reveal_neighbours(self, x: int, y: int) -> float:
        reward = 0
        for cell in self._get_adjacent_cells(x, y):
            if self.board[cell] == BoxType.NOT_REVEALED:
                reward += 0.1
                cell_x, cell_y = self._get_coordinates(cell)
                self.board[cell] = self._get_adjacent_mines(cell_x, cell_y)
                if self.board[cell] == BoxType.EMPTY:
                    reward += self._reveal_neighbours(cell_x, cell_y)
        return reward

    def _get_adjacent_mines(self, x: int, y: int) -> int:
        adjacent_cells = self._get_adjacent_cells(x, y)
        return int(np.sum(np.isin(adjacent_cells, self.mines)))

    def _get_adjacent_cells(self, x: int, y: int) -> List[int]:
        adjacent_indices = []
        for i in range(max(0, x - 1), min(self.width, x + 2)):
            for j in range(max(0, y - 1), min(self.height, y + 2)):
                if i != x or j != y:
                    adjacent_indices.append(self._get_cell_index(i, j))
        return adjacent_indices

    def has_won(self) -> bool:
        return np.sum(self.board == BoxType.NOT_REVEALED) == self.num_mines

    def close(self):
        if self.screen is not None or self.pygame_init:
            pygame.quit()

    def _update_completion(self):
        self.completion = np.sum(self.board >= BoxType.EMPTY) / (self.width * self.height)

    # PLAY GAME FUNCTION
    def set_player_type(self, player_type: PlayerType):
        self.player_type = player_type
        if player_type == PlayerType.HUMAN:
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
                        x, y = event.pos
                        x //= self.CELL_LENGTH
                        y //= self.CELL_LENGTH
                        obs, rewards, done, info = self.step(self._get_cell_index(x, y))
                        self.render()
                        if done:
                            sleep(1)
                            self.screen.fill((255, 0, 0) if not self.has_won() else (0, 255, 0))
                            self._write_pygame_text(
                                ("You won!" if self.has_won() else "You lost!") + " Click to restart",
                                (255, 255, 255),
                                (self.width * self.CELL_LENGTH // 2, self.height * self.CELL_LENGTH // 2),
                            )
                            pygame.display.flip()
                            self._reset_pygame_after_done()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
