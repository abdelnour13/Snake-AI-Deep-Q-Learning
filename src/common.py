import sys
sys.path.append('..')
from definitions import ROOT_DIR,ASSETS_DIR, CHECKPOINTS_DIR, FONTS_DIR, SOUNDS_DIR

class Global:

    CELLS_X = 24
    CELLS_Y = 24
    CELL_WIDTH = 30
    FRAMES_PER_SECOND = 60
    OFFSET = 75
    PADDING = 2
    BARRIER_WIDTH = 3
    SNAKE_SPEED = 5

    P_SCREEN_COLOR = (161,199,80)
    N_SCREEN_COLOR = (173,204,96)
    BARRIER_COLOR = (0,0,0)
    TEXT_COLOR_1 = (32, 46, 2)
    TEXT_COLOR_2 = (255, 255, 255)