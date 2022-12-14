import gym
import minesweeper_env

def main():
    env = gym.make('minesweeper_env/Minesweeper-v0',render_mode="human", width= 10, height= 10, num_mines= 10)
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == '__main__':
    main()
