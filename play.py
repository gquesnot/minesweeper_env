import argparse
from enum import Enum, auto
from time import sleep
import gym

from minesweeper_env.envs import PlayerType, MinesweeperOptions, ObsType, ActionType, ClickedCaseType, Reward


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    def get_num_mines(self, width: int, height: int) -> int:
        if self == Difficulty.EASY:
            return int(width * height * 0.1)
        elif self == Difficulty.MEDIUM:
            return int(width * height * 0.2)
        elif self == Difficulty.HARD:
            return int(width * height * 0.3)
        else:
            raise ValueError(f"Unknown difficulty {self}")


def main():
    parser = argparse.ArgumentParser(
        prog='MineSweeper',
        description='play minesweeper yourself or watch an AI play it',
        epilog='Enjoy the game!')
    parser.add_argument('--type', type=PlayerType, default=PlayerType.HUMAN,
                        help='type of player: human, random')
    # parser.add_argument('--model', type=str, default='best_models/best_model',
    #                     help='path to model to load, ai disabled')
    parser.add_argument('--width', type=int, default=8,
                        help='width of the board (default: 8)')
    parser.add_argument('--height', type=int, default=8,
                        help='height of the board (default: 8)')
    parser.add_argument('--mines', type=int, default=10,
                        help='number of mines (default: 10)')

    args = parser.parse_args()
    if args.type != PlayerType.AI and args.type != PlayerType.RANDOM and args.type != PlayerType.HUMAN:
        raise ValueError('type must be human or ai')

    env = gym.make(
        'minesweeper_env/Minesweeper-v0',
        width=args.width,
        height=args.height,
        num_mines=args.mines,
    )
    obs = env.set_player_type(args.type)
    if args.type == PlayerType.HUMAN:
        env.play_has_human()
    if args.type == PlayerType.AI or args.type == PlayerType.RANDOM:
        model = None
        obs = env.reset()
        if args.type == PlayerType.AI:
            # model = PPO.load(args.model)
            pass
        try:
            while True:
                # if args.type == PlayerType.AI:
                #     action, _states = model.predict(obs)
                # else:
                #     action = env.action_space.sample()
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    sleep(2)
                    print("You lose " if not reward == Reward.WIN else "You win!")
                    env.reset()
                    env.render()
                sleep(0.5)
        except KeyboardInterrupt:
            env.close()


if __name__ == '__main__':
    main()
