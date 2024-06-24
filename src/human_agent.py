import pygame as pg
from game import Game
from enums import Direction, GameState
from common import Global

class HumanControlledAgent:

    SNAKE_UPDATE = pg.USEREVENT

    def __init__(self, game : Game) -> None:
        self.game = game
        self.clock = pg.time.Clock()
        self.direction = None
        pg.time.set_timer(self.SNAKE_UPDATE, int(1000 / Global.SNAKE_SPEED))

    def step(self) -> None:
            
        ### Event Handling
        for event in pg.event.get():

            match event.type:

                case self.SNAKE_UPDATE:
                    self.game.update(self.direction)
                    self.direction = None

                case pg.QUIT:
                    self.game.quit()

                case pg.KEYDOWN:

                    if self.game.state == GameState.STOPPED:
                        self.game.start()

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
                    if self.game.state == GameState.GAME_OVER:
                        self.game.game_over_button.update()
            ### Drawing
            self.game.draw()

            ### Update the screen
            pg.display.update()
            self.clock.tick(Global.FRAMES_PER_SECOND)

    def run(self) -> None:

        while True:
            self.step()
