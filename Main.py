from DQN import DQN
from SnakeGame import SnakeGame
from CatchGame import CatchGame

## Train dqn for snake game

dqn = DQN()
snake_game = SnakeGame()
catch_game = CatchGame()

#dqn.train_model(catch_game)
dqn.test_model(catch_game, 'rl-network-screenshot-catch-400')