from enum import Enum
from pygame.math import Vector2

class GameState(Enum):
    STOPPED = 0
    RUNNING = 1
    GAME_OVER = 2

class Direction(Enum):
    RIGHT = Vector2(1,0)
    LEFT = Vector2(-1,0)
    UP = Vector2(0,-1)
    DOWN = Vector2(0,1)

class Action(Enum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2