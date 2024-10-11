import pygame as pg
import os
from common import Global,FONTS_DIR
from pygame import Surface

class GameOver:

    def __init__(self, screen : Surface,on_restart_click,on_quit_click) -> None:
        self.screen = screen
        self.height = 4
        self.fancy_font = pg.font.Font(os.path.join(FONTS_DIR, "Atop-R99O3.ttf"), 40)
        self.regular_font = pg.font.Font(None, 25)
        self.restart_rect = None
        self.quit_rect = None
        self.on_restart_click = on_restart_click
        self.on_quit_click = on_quit_click

    def draw(self) -> None:

        title_surface = self.fancy_font.render("Game Over", True, (255, 255, 255))
        restart_surface = self.regular_font.render("Restart", True, (255, 255, 255))
        quit_surface = self.regular_font.render("Quit", True, (255, 255, 255))

        title_rect = (
            Global.OFFSET + (Global.CELLS_X * Global.CELL_WIDTH - title_surface.get_width()) / 2,
            Global.OFFSET + (Global.CELLS_Y * Global.CELL_WIDTH - title_surface.get_height()) / 2,
        )

        bg_rect = (
            Global.OFFSET + (Global.CELLS_X * Global.CELL_WIDTH - title_surface.get_width() - Global.CELL_WIDTH) / 2,
            Global.OFFSET + (Global.CELLS_Y * Global.CELL_WIDTH - title_surface.get_height() - Global.CELL_WIDTH) / 2,
            title_surface.get_width() + Global.CELL_WIDTH,
            title_surface.get_height() + Global.CELL_WIDTH + 4 + quit_surface.get_height()
        )

        restart_rect = (
            Global.OFFSET + (Global.CELLS_X * Global.CELL_WIDTH - title_surface.get_width()) / 2 + 5,
            Global.OFFSET + (Global.CELLS_Y * Global.CELL_WIDTH + title_surface.get_height()) / 2 + 4,
        )

        quit_rect = (
            Global.OFFSET + (Global.CELLS_X * Global.CELL_WIDTH + title_surface.get_width()) / 2 - quit_surface.get_width() - 5,
            Global.OFFSET + (Global.CELLS_Y * Global.CELL_WIDTH + title_surface.get_height()) / 2 + 4,
        )

        pg.draw.rect(self.screen, Global.TEXT_COLOR_1, bg_rect, border_radius=12)
        self.screen.blit(title_surface, title_rect)
        self.restart_rect = self.screen.blit(restart_surface, restart_rect)
        self.quit_rect = self.screen.blit(quit_surface, quit_rect)

    def update(self):

        mouse_x, mouse_y = pg.mouse.get_pos()
                        
        if self.restart_rect.x <= mouse_x <= self.restart_rect.x + self.restart_rect.w and self.restart_rect.y <= mouse_y <= self.restart_rect.y + self.restart_rect.h:
            self.on_restart_click()
                        
        if self.quit_rect.x <= mouse_x <= self.quit_rect.x + self.quit_rect.w and self.quit_rect.y <= mouse_y <= self.quit_rect.y + self.quit_rect.h:
            self.on_quit_click()