import pygame as pg
import os
from pygame import Vector2,Surface
from common import Global

class Food:

    def __init__(self,
        screen : Surface,
        position : tuple[int,int],
    ) -> None:
        self.screen = screen
        self.position = Vector2(position)
        self.food_surface = pg.transform.scale(
            pg.image.load(os.path.join("assets", "food.png")), (Global.CELL_WIDTH,Global.CELL_WIDTH)
        )

    def draw(self):
    
        food_rect = pg.Rect(
            self.position.x * Global.CELL_WIDTH + Global.OFFSET, 
            self.position.y * Global.CELL_WIDTH + Global.OFFSET,
            Global.CELL_WIDTH,
            Global.CELL_WIDTH
        )

        self.screen.blit(self.food_surface, food_rect)