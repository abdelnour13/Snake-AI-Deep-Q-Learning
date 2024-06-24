import pygame as pg
import random
import torch
from torch import Tensor
from common import Global
from collections import deque
from game import Game,Direction,GameState
from model import LinearQNet
from trainer import Trainer
from plotter import Plotter
from enums import Action

class Agent:

    SNAKE_UPDATE = pg.USEREVENT
    
    def __init__(self, 
        game : Game,
        max_memory : int = 100_000,
        batch_size : int = 1024,
        learning_rate : float = 0.001,
        max_games : int = 200,
        max_epsilon : float = 0.4,
        min_epsilon : float = 0.0,
        gamma : float = 0.9,
        training : bool = True,
        device : torch.device | None = None     
    ) -> None:
        
        self.clock = pg.time.Clock()
        self.direction = None
        pg.time.set_timer(self.SNAKE_UPDATE, int(1000 / Global.SNAKE_SPEED))
        
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_games = max_games
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.training = training

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_games = 0
        self.best_score = 0
        self.model = LinearQNet(input_size=11,hidden_size=256, output_size=3).to(self.device)
        self.trainer = Trainer(self.model, self.learning_rate, self.gamma, device=self.device)
        self.plotter = Plotter()
        self.memory = deque(maxlen=self.max_memory)
        self.game = game

        if not self.training:
            self.model.load()
            self.model.eval()

    def is_danger(self, action : list[Action]) -> bool:
        direction = self.get_direction_from_action(action)
        return self.game.is_game_over(direction.value)

    def get_state(self) -> Tensor:

        snake_position = self.game.snake.position()
        food_position = self.game.food.position

        return torch.tensor([
            ### Danger
            self.is_danger(Action.STRAIGHT),
            self.is_danger(Action.LEFT),
            self.is_danger(Action.RIGHT),

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
        ]).float()
    

    def remember(self, state : Tensor, action, reward : int, next_state : Tensor, game_over : bool):
        self.memory.append((state, action, reward, next_state, game_over))

    def get_action(self, state : Tensor) -> Action:

        p = random.random()

        epsilon = (self.max_epsilon - self.min_epsilon) * (1 - 3 * self.n_games / self.max_games) + self.min_epsilon

        if p < epsilon and self.training:
            action = random.choice(list(Action))
        else:
            with torch.inference_mode(mode=not self.training):
                x = state.unsqueeze(0).to(self.device)
                y = self.model.predict(x)
                y = y.squeeze().item()

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

    def train_lm(self) -> Tensor:

        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        game_overs = torch.tensor(game_overs).to(self.device)

        return self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_sm(self, state : Tensor, action : int, reward : int, next_state : Tensor, game_over : bool) -> None:

        state = state.unsqueeze(0).to(self.device)
        next_state = next_state.unsqueeze(0).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        game_over = torch.tensor([game_over]).to(self.device)
        
        return self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def game_step(self, direction : Direction) -> tuple[int, bool]:
        
        ### Start the game if it is stopped
        if self.game.state == GameState.STOPPED:
            self.game.start()

        ### Event Handling
        for event in pg.event.get():

            match event.type:

                case pg.QUIT:
                    self.game.quit()

                case pg.MOUSEBUTTONDOWN:
                    if self.game.state == GameState.GAME_OVER and not self.training:
                        self.game.game_over_button.update()

        ### Update the game
        food,is_game_over = self.game.update(direction)

        ### Reward
        reward = 0

        if is_game_over:
            reward = -10
            self.n_games += 1
        
        if food:
            reward = 10

        ### Drawing
        self.game.draw()

        ### Update the screen
        pg.display.update()
        self.clock.tick(Global.FRAMES_PER_SECOND)

        return reward,is_game_over
    
    def step(self) -> None:

        ### Get the old state
        old_state = self.get_state()

        ### Make a prediction or a random move
        action : Action = self.get_action(old_state)
        direction : Direction = self.get_direction_from_action(action)

        ### Play a step in the game
        reward,game_over = self.game_step(direction)

        if self.training:
            ### Get the new state, and the score
            new_state = self.get_state()
            score = self.game.score()

            ### train the short memory
            self.train_sm(old_state, action.value, reward, new_state, game_over)

            ### store this in memory
            self.remember(old_state, action.value, reward, new_state, game_over)

        if game_over:

            if self.training:
                ### train the long memory
                self.train_lm()

                ### plotting
                if score > self.best_score:
                    self.best_score = score
                    self.model.save(f"models_{self.best_score}.pt")

                self.plotter.add_score(score)
                self.plotter.plot()

                ### The necessary logic to restart the game immediatly
                ### if our snake hits the borders
                if self.game.state == GameState.GAME_OVER:
                    if self.n_games < self.max_games:
                        self.game.reset()
                    else:
                        self.game.quit()

    def run(self) -> None:
        while True:
            self.step()
        