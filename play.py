import argparse
from enum import Enum, auto
from time import sleep

import gym
from stable_baselines3 import PPO

import minesweeper_env
from minesweeper_env.envs import PlayerType


class Difficulty(str, Enum):
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()

    def get_num_mines(self, width: int, height: int) -> int:
        match self:
            case Difficulty.EASY:
                return int(width * height // 8.1)
            case Difficulty.MEDIUM:
                return int(width * height // 6.4)
            case Difficulty.HARD:
                return int(width * height // 4.8)
            case _:
                raise ValueError(f"Invalid difficulty {self}")


def main():
    parser = argparse.ArgumentParser(
        prog='MineSweeper',
        description='play minesweeper yourself or watch an AI play it',
        epilog='Enjoy the game!')
    parser.add_argument('--type', type=PlayerType, default=PlayerType.HUMAN,
                        help='type of player: human, ai, random')
    parser.add_argument('--model', type=str, default='best_models/best_model',
                        help='path to model to load')
    parser.add_argument('--width', type=int, default=10,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=10,
                        help='height of the board')
    parser.add_argument('--difficulty', type=Difficulty, default=Difficulty.EASY,
                        help='difficulty of the board: easy, medium, hard')

    args = parser.parse_args()
    print(args)
    if args.type != PlayerType.AI and args.type != PlayerType.RANDOM and args.type != PlayerType.HUMAN:
        raise ValueError('type must be human or ai')

    env = gym.make(
        'minesweeper_env/Minesweeper-v0',
        width=args.width,
        height=args.height,
        num_mines=args.difficulty.get_num_mines(args.width, args.height)
    )
    obs = env.set_player_type(args.type)
    if args.type == PlayerType.AI or args.type == PlayerType.RANDOM:
        model = None
        obs = env.reset()
        if args.type == PlayerType.AI:
            model = PPO.load(args.model)
        try:
            while True:
                if args.type == PlayerType.AI:
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                obs, rewards, done, info = env.step(action)
                env.render()
                if done:
                    sleep(2)
                    print("done")
                    env.reset()
                    env.render()

                sleep(0.5)
        except KeyboardInterrupt:
            env.close()


if __name__ == '__main__':
    main()
