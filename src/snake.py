import pygame as pg
import os
from pygame import Vector2,Surface
from enum import Enum
from common import Global

class Direction(Enum):
    RIGHT = Vector2(1,0)
    LEFT = Vector2(-1,0)
    UP = Vector2(0,-1)
    DOWN = Vector2(0,1)

class Snake:

    def __init__(self, 
        screen : Surface,
        initial_position : tuple[int,int],
        initial_length : int,
        initial_direction : Direction
    ) -> None:
        
        x, y = initial_position

        self.screen = screen
        self.body = [Vector2(x + i - initial_length + 1, y) for i in range(initial_length)]
        self.direction = initial_direction
        self.assets = self.load_assets()
        self.eat_sound = pg.mixer.Sound(os.path.join('sounds', 'eat.mp3'))
        self.wall_sound = pg.mixer.Sound(os.path.join('sounds', 'wall.mp3'))

    def load_assets(self) -> dict[str,Surface]:

        result = {}

        for asset_name in os.listdir('assets'):
            name,_ = os.path.splitext(asset_name)
            result[name] = pg.transform.scale(
                pg.image.load(os.path.join('assets', asset_name)), (Global.CELL_WIDTH,Global.CELL_WIDTH)
            )

        return result

    def position(self):
        return self.body[-1]
    
    def get_segments(self) -> list[int]:

        corners = [0]
        tmp = self.body[0]

        for i in range(1, len(self.body)):
            if self.body[i].x != tmp.x and self.body[i].y != tmp.y:
                corners.append(i-1)
                tmp = self.body[i-1]

        corners.append(len(self.body) - 1)

        return corners
    
    def draw_tail(self):

        cell = self.body[0]
                
        rect = pg.Rect(
            cell.x * Global.CELL_WIDTH + Global.OFFSET,
            cell.y * Global.CELL_WIDTH + Global.OFFSET,
            Global.CELL_WIDTH,
            Global.CELL_WIDTH
        )

        direction = self.body[1] - self.body[0]

        match direction:
            case Direction.UP.value:
                self.screen.blit(self.assets['tail_down'], rect)
            case Direction.DOWN.value:
                self.screen.blit(self.assets['tail_up'], rect)
            case Direction.LEFT.value:
                self.screen.blit(self.assets['tail_right'], rect)
            case Direction.RIGHT.value:
                self.screen.blit(self.assets['tail_left'], rect)

    def draw_head(self):

        cell = self.body[-1]
                
        rect = pg.Rect(
            cell.x * Global.CELL_WIDTH + Global.OFFSET,
            cell.y * Global.CELL_WIDTH + Global.OFFSET,
            Global.CELL_WIDTH,
            Global.CELL_WIDTH
        )

        match self.direction:
            case Direction.UP:
                self.screen.blit(self.assets['head_up'], rect)
            case Direction.DOWN:
                self.screen.blit(self.assets['head_down'], rect)
            case Direction.LEFT:
                self.screen.blit(self.assets['head_left'], rect)
            case Direction.RIGHT:
                self.screen.blit(self.assets['head_right'], rect)

    def draw_corners(self, corners : list[int]):

        for corner in corners:
            
            cell = self.body[corner]
                
            rect = pg.Rect(
                cell.x * Global.CELL_WIDTH + Global.OFFSET,
                cell.y * Global.CELL_WIDTH + Global.OFFSET,
                Global.CELL_WIDTH,
                Global.CELL_WIDTH
            )

            direction_1 = self.body[corner] - self.body[corner-1]
            direction_2 = self.body[corner+1] - self.body[corner]

            if direction_1 == Direction.RIGHT.value and direction_2 == Direction.UP.value:
                self.screen.blit(self.assets['body_topleft'], rect)

            if direction_1 == Direction.RIGHT.value and direction_2 == Direction.DOWN.value:
                self.screen.blit(self.assets['body_bottomleft'], rect)

            if direction_1 == Direction.LEFT.value and direction_2 == Direction.UP.value:
                self.screen.blit(self.assets['body_topright'], rect)
                
            if direction_1 == Direction.LEFT.value and direction_2 == Direction.DOWN.value:
                self.screen.blit(self.assets['body_bottomright'], rect)

            if direction_1 == Direction.UP.value and direction_2 == Direction.RIGHT.value:
                self.screen.blit(self.assets['body_bottomright'], rect)

            if direction_1 == Direction.UP.value and direction_2 == Direction.LEFT.value:
                self.screen.blit(self.assets['body_bottomleft'], rect)

            if direction_1 == Direction.DOWN.value and direction_2 == Direction.RIGHT.value:
                self.screen.blit(self.assets['body_topright'], rect)

            if direction_1 == Direction.DOWN.value and direction_2 == Direction.LEFT.value:
                self.screen.blit(self.assets['body_topleft'], rect)

    def draw_body_parts(self, corners : list[int]):

        for start,end in zip(corners[:-1],corners[1:]):

            segment_direction = self.body[start+1] - self.body[start]

            for i in range(start + 1,end):

                cell = self.body[i]
                
                rect = pg.Rect(
                    cell.x * Global.CELL_WIDTH + Global.OFFSET,
                    cell.y * Global.CELL_WIDTH + Global.OFFSET,
                    Global.CELL_WIDTH,
                    Global.CELL_WIDTH
                )

                match segment_direction:
                    case Direction.UP.value:
                        self.screen.blit(self.assets['body_vertical'], rect)
                    case Direction.DOWN.value:
                        self.screen.blit(self.assets['body_vertical'], rect)
                    case Direction.LEFT.value:
                        self.screen.blit(self.assets['body_horizontal'], rect)
                    case Direction.RIGHT.value:
                        self.screen.blit(self.assets['body_horizontal'], rect)

    def draw(self):
        corners = self.get_segments()
        self.draw_body_parts(corners)
        self.draw_corners(corners[1:-1])
        self.draw_tail()
        self.draw_head()

    def update(self):
        head = self.body[-1].copy()
        head += self.direction.value
        self.body.pop(0)
        self.body.append(head)

    def set_direction(self, direction : Direction):

        if self.direction == Direction.RIGHT and direction == Direction.LEFT:
            return False
        
        if self.direction == Direction.LEFT and direction == Direction.RIGHT:
            return False
        
        if self.direction == Direction.UP and direction == Direction.DOWN:
            return False
        
        if self.direction == Direction.DOWN and direction == Direction.UP:
            return False
        
        self.direction = direction

        return True
    
    def grow(self):
        tail = self.body[0].copy()
        direction = self.body[1] - self.body[0]
        tail -= direction
        self.body.insert(0, tail)
        self.eat_sound.play()

    def check_for_collision(self, translation : Vector2 | None = None) -> bool:

        if translation is None:
            translation = Vector2(0,0)

        head = self.body[-1] + translation
        return head in self.body[:-1]