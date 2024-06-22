import pygame as pg
import random
from common import Global
from collections import deque
from game import Game,Mode,Direction,GameState

class Agent:
    
    def __init__(self, 
        game : Game,
        max_memory : int = 100_000,
        batch_size : int = 1000,
        learning_rate : float = 0.001,
        max_games : int = 100,
        epsilon : float = 0.0,
        gamma : float = 0.0,
        training : bool = True     
    ) -> None:
        
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_games = max_games
        self.epsilon = epsilon
        self.gamma = gamma
        self.training = training

        self.n_games = 0
        self.model = None
        self.trainer = None
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

    def get_state(self):
        return [
            
        ]

    def remember(self, state, action, reward, next_state, game_over):
        pass

    def train_lm(self, state, action, reward, next_state, game_over):
        pass

    def train_sm(self):
        pass

    def run(self):

        best_score = 0

        while True:

            ### Get the old state
            old_state = self.get_state()

            ### Make a prediction
            self.game.direction = self.get_action(old_state)

            ### Get the reward,game state and the current score
            self.game_step()
            new_state = self.get_state()
            reward = self.reward
            game_over = self.game.state == GameState.GAME_OVER
            score = self.game.score.value

            ### train the short memory
            input = old_state, self.game.direction, reward, new_state, game_over
            self.train_sm(input)

            ### store this in memory
            self.memory.append(input)

            if game_over:
                ### train the long memory
                ### plotting
                if score > best_score:
                    best_score = score
                    # TODO : save the model

    def get_action(self) -> Direction:
        
        action = int(random.random() * 3)

        if action == 1:

            if self.game.snake.direction in [Direction.UP,Direction.DOWN]:
                return Direction.RIGHT
            else:
                return Direction.UP

        if action == 2:
            if self.game.snake.direction in [Direction.UP,Direction.DOWN]:
                return Direction.LEFT
            else:
                return Direction.DOWN

def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = Game(Mode.AGENT_CONTROLLED)

    while True:

        state = agent.get_state()
        move = agent.get_action()
        reward,done,score = game.step(move)

        new_state = agent.get_state()

        agent.train_sm()
        agent.remember()

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_lm()

            if score > best_score:
                best_score = score
                # TODO : save the model

            # TODO : plot


if __name__ == '__main__':
    game : Game = Game()
    agent = Agent(game=game)
    agent.run()
        