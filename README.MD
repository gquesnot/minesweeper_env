## Minesweeper Env

Minesweeper Env is a gym environment for the game of minesweeper. The goal of the game is to clear a rectangular board containing hidden "mines" or bombs without detonating any of them, with help from clues about the number of neighboring mines in each field.  

You can play the game or train an agent to play the game using the gym environment.



### Installation

Python 3.10
```bash
pip install -r requirements.txt && 
cd minesweeper_env && 
pip install -e .
```

### Usage

```bash
python play.py --width 8 --height 8 --mines 1 --type human

python play.py --help # for help
```

