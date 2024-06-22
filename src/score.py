import pygame as pg
import os
from pygame import Surface
from common import Global

class Score:

    def __init__(self, screen : Surface) -> None:
        self.screen = screen
        self.value = 0
        self.food = pg.transform.scale(
            pg.image.load(os.path.join("assets", "food.png")), (Global.CELL_WIDTH * 1.25,Global.CELL_WIDTH * 1.25)
        )
        self.font = pg.font.Font(os.path.join("fonts", "Atop-R99O3.ttf"), 30)

    def draw(self):

        rect = (
            Global.OFFSET,
            Global.OFFSET - Global.CELL_WIDTH - 10,
        )

        surface = self.font.render(f"Score : {self.value}", True, Global.TEXT_COLOR_1)
        self.screen.blit(surface, rect)

        w = surface.get_width()

        rect = (
            Global.OFFSET + w,
            Global.OFFSET - Global.CELL_WIDTH - 12,
        )

        self.screen.blit(self.food, rect)

    def update(self):
        self.value += 1