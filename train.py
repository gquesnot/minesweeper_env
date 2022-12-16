import os

import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed, update_learning_rate

import minesweeper_env
from utils.dir_helpers import get_unique_name
from utils.rl_helpers import linear_schedule


def train_with_config(config: dict[str | float], max_steps: int, num_envs: int = 1):
    config_str = "_".join([f"{k}_{str(v)}" for k, v in config.items()])
    best_log_dir = os.path.join("./best_logs", config_str)
    best_models_dir = os.path.join("./best_models", config_str)
    log_dir = os.path.join("./logs", config_str)
    last_model_dir = os.path.join("./models", config_str)

    eval_env = gym.make('minesweeper_env/Minesweeper-v0')
    monitor = Monitor(eval_env, best_log_dir)
    eval_callback = EvalCallback(monitor, best_model_save_path=best_models_dir,
                                 log_path=best_log_dir, eval_freq=config['n_steps'] * 5,
                                 n_eval_episodes=20,
                                 deterministic=True, render=False)

    env = make_vec_env('minesweeper_env/Minesweeper-v0', n_envs=num_envs,
                       env_kwargs={"width": 10, "height": 10, "num_mines": 10})
    name = "minesweeper_ppo"
    model = PPO(
        "MlpPolicy",
        env,
        **config,
        learning_rate=linear_schedule(3e-4),
        verbose=1,
        seed=set_random_seed(4),
        tensorboard_log=log_dir,

    )
    model.learn(total_timesteps=max_steps, callback=eval_callback)
    model.save(last_model_dir)


def main():
    # concat config to name
    # n_steps = [256, 512, 1024, 2048, 4096]
    # batch_sizes = [32, 64, 128, 256, 512]
    # n_epochs = [10, 7, 4]
    # #for n_epoch in n_epochs:
    # for n_step in n_steps:
    #     for batch_size in batch_sizes:
    #         train_with_config(
    #             {
    #                 "n_steps": n_step,
    #                 "batch_size": batch_size,
    #                 # 'n_epochs': n_epoch
    #             },
    #             max_steps=int(5e5),
    #             num_envs=20
    #         )
    train_with_config(
        {
            "n_steps": 256,
            "batch_size": 128,
            # 'n_epochs': n_epoch
        },
        max_steps=int(2.5e6),
        num_envs=20
    )



if __name__ == '__main__':
    main()
