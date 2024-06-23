import pygame as pg
import random
import torch
from torch import Tensor
from common import Global
from collections import deque
from game import Game,Direction,GameState
from model import LinearQNet
from trainer import Trainer
from utils import plot
from enum import Enum

class Action(Enum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2

class Agent:
    
    def __init__(self, 
        game : Game,
        max_memory : int = 100_000,
        batch_size : int = 1000,
        learning_rate : float = 0.001,
        max_games : int = 100,
        max_epsilon : float = 0.9,
        min_epsilon : float = 0.1,
        gamma : float = 0.9,
        training : bool = True     
    ) -> None:
        
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_games = max_games
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.training = training

        self.n_games = 0
        self.model = LinearQNet(11, 3, 256)
        self.trainer = Trainer(self.model, self.learning_rate, self.gamma)
        self.reward = 0
        self.memory = deque(maxlen=self.max_memory)
        self.game = game
        self.current_score = self.game.score.value

    def game_step(self):
        
        ### Update the reward if a change in the game's score is detected
        self.reward += (self.game.score.value - self.current_score)
        self.current_score = self.game.score.value
            
        ### The necessary logic to restart the game immediatly
        ### if our snake hits the borders
        if self.game.state == GameState.GAME_OVER:

            self.reward -= 10

            if self.n_games < self.max_games:
                self.n_games += 1
                self.game.reset()
            else:
                self.game.quit()

        if self.game.state == GameState.STOPPED:
            self.game.start()

        ### Event Handling
        for event in pg.event.get():

            match event.type:

                case self.game.SNAKE_UPDATE:
                    self.game.update()

                case pg.QUIT:
                    self.game.quit()

        ### Drawing
        self.game.draw()

        ### Update the screen
        pg.display.update()
        self.game.clock.tick(Global.FRAMES_PER_SECOND)

    def is_danger(self, action : Action) -> bool:
        direction = self.get_direction_from_action(action)
        return self.game.is_game_over(direction.value)

    def get_state(self) -> torch.Tensor:

        snake_position = self.game.snake.position()
        food_position = self.game.food.position

        return torch.tensor([
            ### Danger
            self.is_danger(Action.STRAIGHT),
            self.is_danger(Action.RIGHT),
            self.is_danger(Action.LEFT),

            ### Direction
            self.game.snake.direction == Direction.UP,
            self.game.snake.direction == Direction.DOWN,
            self.game.snake.direction == Direction.LEFT,
            self.game.snake.direction == Direction.RIGHT,

            ### Food
            snake_position.x < food_position.x,
            snake_position.x > food_position.x,
            snake_position.y < food_position.y,
            snake_position.y > food_position.y
        ]).long()

    def remember(self, state : Tensor, action, reward : int, next_state : Tensor, game_over : bool):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_lm(self):

        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        game_overs = torch.tensor(game_overs)

        self.trainer.train_step((states, actions, rewards, next_states, game_overs))

    def train_sm(self, state : Tensor, action : int, reward : int, next_state : Tensor, game_over : bool):

        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        game_over = torch.tensor([game_over])
        
        self.trainer.train_step((state, action, reward, next_state, game_over))

    def run(self):

        best_score = 0
        total = 0
        scores = []
        mean_scores = []

        while True:

            ### Get the old state
            old_state = self.get_state()

            ### Make a prediction
            action : Action = self.get_action(old_state)
            self.game.direction = self.get_direction_from_action(action)

            ### Get the reward,game state and the current score
            self.game_step()
            new_state = self.get_state()

            reward = self.reward
            game_over = self.game.is_game_over()
            score = self.game.score.value

            ### train the short memory
            input = old_state, action.value, reward, new_state, game_over
            self.train_sm(input)

            ### store this in memory
            self.memory.append(input)

            if game_over:

                ### train the long memory
                ### plotting
                if score > best_score:
                    best_score = score
                    self.model.save(f"models/{best_score}.pt")
                    # TODO : save the model

                total += score
                mean_score = total / self.n_games
                scores.append(score)
                mean_scores.append(mean_score)

                plot(scores, mean_scores)

    def get_action(self) -> Action:

        p = random.random()

        epsilon = (self.max_epsilon - self.min_epsilon) * (1 - self.n_games / self.max_games) + self.min_epsilon

        if p < epsilon and self.training:
            action = random.choice(list(Action))
        else:
            x = self.get_state().unsqueeze(0)
            y = self.model.predict(x).squeeze().item()
            action = Action(y)

        return action

    def get_direction_from_action(self, action : Action) -> Direction:
        
        if action == Action.STRAIGHT:
            return self.game.snake.direction
    
        if action == Action.RIGHT:

            if self.game.snake.direction in [Direction.UP,Direction.DOWN]:
                return Direction.RIGHT
            else:
                return Direction.UP

        if action == Action.LEFT:

            if self.game.snake.direction in [Direction.UP,Direction.DOWN]:
                return Direction.LEFT
            else:
                return Direction.DOWN

if __name__ == '__main__':
    game : Game = Game()
    agent = Agent(game=game)
    agent.run()
        