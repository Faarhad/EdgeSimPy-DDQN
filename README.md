## EdgeSimPy + DDQN environment
### Author: Farhad Kazemipour
### Contact: evanescencefkz@gmail.com

This repository contains:
- A baseline EdgeSimPy scenario (`scenario_1cloud3edge.py`)
- A reinforcement-learning wrapper environment (`edgesim_env.py`)
- A DDQN implementation on CartPole, (a tiny hello-world for those who are new to RL) (`ddqn_cartpole.py`)
- A simple test script for the RL environment (`test_edgesim_env.py`)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # on Windows PowerShell
pip install -r requirements.txt


Running

Test RL env: python test_edgesim_env.py

Run baseline scenario: python scenario_1cloud3edge.py

Train DDQN on CartPole: python ddqn_cartpole.py