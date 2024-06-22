import pygame as pg
from pygame import Surface
from common import Global

class Playgound:

    def __init__(self,
        screen : Surface             
    ) -> None:
        self.screen = screen

    def draw_background(self): 
        
        self.screen.fill(Global.N_SCREEN_COLOR)

        for x in range(Global.CELLS_X):
            for y in range(Global.CELLS_Y):

                rect = pg.Rect(
                    x * Global.CELL_WIDTH + Global.OFFSET, 
                    y * Global.CELL_WIDTH + Global.OFFSET,
                    Global.CELL_WIDTH,
                    Global.CELL_WIDTH
                )

                if (x + y) % 2 == 0:
                    pg.draw.rect(self.screen, Global.N_SCREEN_COLOR, rect)
                else:
                    pg.draw.rect(self.screen, Global.P_SCREEN_COLOR, rect)

    def draw_borders(self):

        offset = Global.OFFSET
        barrier_width = Global.BARRIER_WIDTH
        width = Global.CELLS_X
        height = Global.CELLS_Y
        cell_width = Global.CELL_WIDTH
        cell_height = Global.CELL_WIDTH

        pg.draw.lines(self.screen, Global.BARRIER_COLOR, True, [
            (offset - barrier_width, offset - barrier_width), 
            (offset + width * cell_width + barrier_width, offset - barrier_width), 
            (offset + width * cell_width + barrier_width, offset + height * cell_height + barrier_width), 
            (offset - barrier_width, offset + height * cell_height + barrier_width)
        ], barrier_width)

    def draw(self):
        self.draw_background()
        self.draw_borders()