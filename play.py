from time import sleep

import gym
from stable_baselines3 import PPO

import minesweeper_env

def main():
    env = gym.make('minesweeper_env/Minesweeper-v0', width=10, height=10, num_mines=10)
    model = PPO.load("minesweeper_ppo")
    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            print("Won"if env.has_won() else "Lost")
            env.reset()
        env.render("human")
        sleep(0.2)


if __name__ == '__main__':
    main()
