from gym.envs.registration import register

register(
    id='minesweeper_env/Minesweeper-v0',
    entry_point='minesweeper_env.envs:MinesweeperEnv',
    max_episode_steps=1000,
)
