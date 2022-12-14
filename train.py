from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import minesweeper_env

def main():
    env = make_vec_env('minesweeper_env/Minesweeper-v0', n_envs=10,
                       env_kwargs={"width": 10, "height": 10, "num_mines": 10})
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/",
        n_steps=2048,
        n_epochs=10,

    )
    model.learn(total_timesteps=1000000)
    model.save("minesweeper_ppo")


if __name__ == '__main__':
    main()
