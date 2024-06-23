import sys
import random
import pygame as pg
from pygame import Vector2
from food import Food
from snake import Snake,Direction
from game_over import GameOver
from enum import Enum
from score import Score
from playground import Playgound
from common import Global

pg.init()

class Mode(Enum):
    HUMAN_CONTROLLED = 0
    AGENT_CONTROLLED = 1

class GameState(Enum):
    STOPPED = 0
    RUNNING = 1
    GAME_OVER = 2

class Game:

    SNAKE_UPDATE = pg.USEREVENT

    def __init__(self) -> None:

        self.screen = pg.display.set_mode((
            Global.CELLS_X * Global.CELL_WIDTH + 2 * Global.OFFSET,
            Global.CELLS_Y * Global.CELL_WIDTH + 2 * Global.OFFSET
        ))

        self.clock = pg.time.Clock()
        self.playground = Playgound(self.screen)

        self.reset()

        pg.time.set_timer(Game.SNAKE_UPDATE, int(1000 / Global.SNAKE_SPEED))
        pg.display.set_caption("Snake AI")

    def create_snake(self) -> Snake:

        x = int(Global.CELLS_X / 2)
        y = int(Global.CELLS_Y / 2)

        return Snake(
            self.screen, 
            initial_position=(x,y), 
            initial_length=3, 
            initial_direction=Direction.RIGHT
        )

    def reset(self):
        self.snake = self.create_snake()
        self.food = self.spwan_food()
        self.score = Score(self.screen)
        self.game_over_button = GameOver(self.screen,on_quit_click=self.quit,on_restart_click=self.reset)
        self.state = GameState.STOPPED
        self.direction = None

    def spwan_food(self):

        food = Food(self.screen, (0,0))

        x = int(random.random() * Global.CELLS_X)
        y = int(random.random() * Global.CELLS_Y)

        food.position = Vector2(x, y)

        while food.position in self.snake.body:

            x = int(random.random() * Global.CELLS_X)
            y = int(random.random() * Global.CELLS_Y)

            food.position = Vector2(x, y)

        return food
   
    def draw(self):

        self.playground.draw()
        self.food.draw()
        self.snake.draw()
        self.score.draw()

        if self.state == GameState.GAME_OVER:
            self.game_over_button.draw()

    def update(self):

        if self.state != GameState.RUNNING:
            return

        if self.direction is not None:
            self.snake.set_direction(self.direction)
            self.direction = None

        self.snake.update()

        if self.is_game_over():
            self.game_over()

        if self.food.position == self.snake.position():
            self.snake.grow()
            self.food = self.spwan_food()
            self.score.update()

    def check_for_collision_with_barriers(self, translation : Vector2 | None = None):

        if translation is None:
            translation = Vector2(0,0)

        snake_position =  self.snake.position() + translation
        result = snake_position.x < 0
        result = result or snake_position.x > Global.CELLS_X - 1
        result = result or snake_position.y < 0
        result = result or snake_position.y > Global.CELLS_Y - 1
        return result

    def is_game_over(self, translation : Vector2 | None = None):
        # TODO : Check the influence of this condition : self.frame_iteration >= 100 * len(self.snake.body)
        b = self.state == GameState.GAME_OVER
        b = b or self.check_for_collision_with_barriers(translation)
        b = b or self.snake.check_for_collision(translation)
        return b
    
    def game_over(self):
        if self.state != GameState.GAME_OVER:
            self.state = GameState.GAME_OVER
            self.snake.wall_sound.play()
            
    def quit(self):
        pg.quit()
        sys.exit()

    def start(self):
        self.state = GameState.RUNNING

    def step(self):
                    
        ### Event Handling
        for event in pg.event.get():

            match event.type:

                case self.SNAKE_UPDATE:
                    self.update()

                case pg.QUIT:
                    self.quit()

                case pg.KEYDOWN:

                    if self.state == GameState.STOPPED:
                        self.start()

                    match event.key:
                        case pg.K_UP:
                            self.direction = Direction.UP
                        case pg.K_DOWN:
                            self.direction = Direction.DOWN
                        case pg.K_LEFT:
                            self.direction = Direction.LEFT
                        case pg.K_RIGHT:
                            self.direction = Direction.RIGHT

                case pg.MOUSEBUTTONDOWN:
                    if self.state == GameState.GAME_OVER:
                        self.game_over_button.update()
            ### Drawing
            self.draw()

            ### Update the screen
            pg.display.update()
            self.clock.tick(Global.FRAMES_PER_SECOND)

    def run(self) -> None:
        while True:
            self.step()