from enum import IntEnum
from functools import cache
from typing import Tuple, Optional, Union, List

import gym
import numpy as np
import pygame
from gym.error import DependencyNotInstalled


class BoxType(IntEnum):
    MINE = -2
    NOT_REVEALED = -1
    EMPTY = 0


class MinesweeperEnv(gym.Env):
    board: np.array
    mines: List[Tuple[int, int]]
    screen: Optional[pygame.Surface]

    def __init__(self, width: int = 10, height: int = 10, num_mines: int = 10):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        print(
            "init env Minesweeper\n",
            f"width: {self.width}\n",
            f"height: {self.height}\n",
            f"num_mines: {self.num_mines}\n",

        )
        self.observation_space = gym.spaces.Box(
            low=np.full((self.height * self.width,), fill_value=BoxType.MINE, dtype=int),
            high=np.full((self.height * self.width,), fill_value=9, dtype=int),
            shape=(width * height,),
            dtype=int
        )
        self.action_space = gym.spaces.Discrete(width * height)
        self.screen = None

    def reset(self):
        self._set_board()
        self._set_mines()
        return self._get_obs()

    def _set_board(self) -> np.array:
        self.board = np.full((self.height * self.width,), fill_value=BoxType.NOT_REVEALED, dtype=int)

    def _set_mines(self):
        self.mines = []
        temp_num_mines = self.num_mines
        while temp_num_mines > 0:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            identifier = y * self.width + x
            if id not in self.mines:
                self.mines.append(identifier)
                temp_num_mines -= 1

    def render(self, mode="human"):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )
        if mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.width * 25, self.height * 25)
                )
            self.screen.fill((255, 255, 255))
            for i in range(self.width):
                for j in range(self.height):
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 0),
                        (i * 25, j * 25, 25, 25),
                        1,
                    )
                    if self.board[j * self.width + i] >= BoxType.EMPTY:
                        text = pygame.font.SysFont("Arial", 20).render(
                            str(self.board[j * self.width + i]), True, (0, 0, 0)
                        )
                        self.screen.blit(
                            text,
                            (
                                i * 25 + 12 - text.get_width() // 2,
                                j * 25 + 12 - text.get_height() // 2,
                            ),
                        )
                    elif self.board[j * self.width + i] == BoxType.MINE:
                        pygame.draw.circle(
                            self.screen, (0, 0, 0), (i * 25 + 12, j * 25 + 12), 5
                        )
            pygame.display.flip()
        else:
            return self.board

    def step(self, action):
        info = {}
        reward = 0
        # convert action (np.array of shape (1,)) to int
        if isinstance(action, np.ndarray):
            action = action.item()
        if action in self.mines:
            reward -= 1.0
            terminated = True
        elif self.board[action] >= BoxType.EMPTY:
            terminated = False
            reward = 0
        else:
            reward += 0.1
            self.board[action] = self._get_num_mines_around(action)
            if self.board[action] == BoxType.EMPTY:
                reward += self._reveal_neighbours(action)
            terminated = self.has_won()
        return self._get_obs(), reward, terminated, info

    def _get_obs(self):
        return self.board

    @cache
    def _get_coordinates(self, action):
        x = action % self.width
        y = action // self.width
        return x, y

    @cache
    def _get_identifier(self, x, y):
        return y * self.width + x

    def _reveal_neighbours(self, action: int):
        x, y = self._get_coordinates(action)
        reward = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                identifier = self._get_identifier(x + i, y + j)
                if self._is_valid_identifier(identifier) and self.board[identifier] == BoxType.NOT_REVEALED:
                    reward += 0.1
                    self.board[identifier] = self._get_num_mines_around(identifier)
                    if self.board[identifier] == BoxType.EMPTY:
                        reward += self._reveal_neighbours(identifier)
        return reward

    def _get_num_mines_around(self, action: int) -> int:
        x, y = self._get_coordinates(action)
        return sum([1 for i in range(-1, 2) for j in range(-1, 2) if
                    self._get_identifier(x + i, y + j) in self.mines])

    def _is_valid_identifier(self, identifier: int) -> bool:
        return 0 <= identifier < self.width * self.height

    def has_won(self) -> bool:
        return np.sum(self.board == BoxType.NOT_REVEALED) == self.num_mines

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
