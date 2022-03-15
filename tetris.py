import time
from typing import List
from numpy import character
import pygame
# import pygame.freetype
from pygame.math import Vector2 as vec
from math import ceil
from copy import deepcopy
import random

# TODO List:
# - score
# - hold display
# - upcoming display
# - fix instant lose bug (@Piece.align_to_top())
# - fix full drop
# - finish typing


FPS = 60

BOARD_DIM = (10, 20)

keys = {32: 'SPACE', 13: 'RETURN', 113: 'Q', 119: 'W', 101: 'E', 114: 'R', 116: 'T', 121: 'Y', 117: 'U', 105: 'I', 111: 'O', 112: 'P', 97: 'A', 115: 'S', 100: 'D', 1073742053: 'RSHIFT', 102: 'F', 103: 'G', 104: 'H', 106: 'J', 107: 'K', 108: 'L', 122: 'Z', 120: 'X', 99: 'C', 118: 'V', 98: 'B', 110: 'N', 109: 'M', 49: '1', 50: '2', 51: '3', 52: '4', 53: '5', 54: '6', 55: '7', 56: '8', 57: '9', 48: '0', 44: ',', 46: '.', 59: ';', 47: '/', 39: "'", 91: '[', 93: ']', 92: '\\', 45: '-', 61: '=', 9: 'TAB', 8: 'BACKSPACE', 1073742049: 'LSHIFT', 96: '`', 1073741906: 'UP', 1073741905: 'DOWN', 1073741903: 'RIGHT', 1073741904: 'LEFT', 27: 'ESCAPE'}

def nums_to_keys(nums):
    return [ keys[num] for num in nums if num in keys]


def max_board_size():
    '''Returns board scale factor'''
    win_size = tuple(x * 0.9 for x in pygame.display.get_window_size())

    if win_size[0] / BOARD_DIM[0] >= win_size[1] / BOARD_DIM[1]: # y is limiting
        return win_size[1] / BOARD_DIM[1]
    else:
        return win_size[0] / BOARD_DIM[0]

def board_center():
    return vec(*(x/2 for x in pygame.display.get_window_size()),)

def rect_center(x, y, w, h):
    x -= w / 2
    y -= h / 2
    return pygame.Rect(x, y, w, h)


class Piece:
    shapes = { # relative distance from center
        'I': {'points': [ vec(-1, 0), vec(0, 0), vec(1, 0), vec(2, 0) ], 'center': vec(0, 0), 'color': '#00BFFF' },
        'J': {'points': [ vec(-1, -1), vec(0, -1), vec(0, 0), vec(0, 1) ], 'center': vec(0, 0), 'color': '#0000FF' },
        'L': {'points': [ vec(-1, -1), vec(-1, 0), vec(0, 0), vec(1, 0) ], 'center': vec(0, 0), 'color': '#FF7700' },
        'O': {'points': [ vec(0, 0), vec(0, 1), vec(1, 0), vec(1, 1) ], 'center': vec(0.5, 0.5), 'color': '#FFFF00' },
        'S': {'points': [ vec(-1, -1), vec(0, -1), vec(0, 0), vec(1, 0) ], 'center': vec(0, 0), 'color': '#00FF00' }, 
        'T': {'points': [ vec(-1, 0), vec(0, 0), vec(0, 1), vec(1, 0) ], 'center': vec(0, 0), 'color': '#9900FF' },
        'Z': {'points': [ vec(-1, -1), vec(-1, 0), vec(0, 0), vec(0, 1) ], 'center': vec(0, 0), 'color': '#FF0000' }
    }

    @staticmethod
    def new_bag() -> List:
        pieces = list(Piece.shapes.keys())
        random.shuffle(pieces)
        bag = [ Piece(vec(BOARD_DIM[0] >> 1, 5), type) for type in pieces ]
        for item in bag: item.rotate(random.randint(0,3))
        for item in bag: item.align_to_top()

        return bag

    def __init__(self, pos: vec, type: character) -> None:
        self.pos = pos
        self.shape = Piece.shapes[type]['points']
        self.center = Piece.shapes[type]['center']
        self.color = Piece.shapes[type]['color']
        self.type = type

    def __repr__(self) -> str:
        return "Piece: points at {points} with color {color}".format(points=[tuple(v) for v in self.adj_shape()], color=self.color)

    def __str__(self) -> str:
        return f"{self.type} piece at {tuple(self.pos)}"

    def adj_shape(self) -> List[vec]:
        return [ square + self.pos for square in self.shape ]

    def draw(self, win: pygame.Surface, coords: vec) -> None:
        size = max_board_size()
        for square in self.shape:
            pos = (self.pos + square) * size + coords

            pygame.draw.rect(win, self.color, (*pos, size+1, size+1))
    

    def rotate(self, dir: int) -> None:
        for i, p in enumerate(self.shape):
            p -= self.center
            p = p.rotate(90 * dir)
            p += self.center
            self.shape[i] = p

    def check_collision(self, pieces: List) -> bool:
        for square in self.adj_shape():
            if square[0] not in range(BOARD_DIM[0]) or square[1] not in range(BOARD_DIM[1]):
                return True
        if any( pieces[int(me.y)][int(me.x)] is not None for me in self.adj_shape() ):
            return True
        return False

    def align_to_top(self) -> None:
        print(repr(self))
        highest_y = min( p.y for p in self.adj_shape() )
        print(highest_y)

        self.pos.y -= highest_y

        print(min( p.y for p in self.adj_shape() ))

        if min( p.y for p in self.adj_shape() ) < 0:
            raise Exception

        print(self.pos)
        # self.pos = vec(BOARD_DIM[0] >> 1, self.pos.y - min( p.y for p in self.adj_shape() ))




class TetrisGame:
    def __init__(self) -> None:
        self.size = BOARD_DIM
        self.points = 0
        self.set_pieces = [ [ None ] * BOARD_DIM[0] for _ in range(BOARD_DIM[1]) ] # 2d array of colors
        self.falling_piece: Piece = None
        self.queued_pieces = []
        self.held_piece: Piece = None
        self.speed = 20 # big num = slower
        self.since_fall = 0
        self.paused = False
        self.can_hold = True


        self.queued_pieces.extend(Piece.new_bag())
        self.falling_piece = self.queued_pieces.pop(0)

    def draw(self, win: pygame.Surface) -> None:
        board_dims = (*board_center(), *(x * max_board_size() for x in BOARD_DIM))
        centered_dims = rect_center(*board_dims)

        self.falling_piece.draw(win, vec(centered_dims[:2]))

        for y, l in enumerate(self.set_pieces):
            for x, color in enumerate(l):
                if color is not None:
                    pos = vec(x, y) * max_board_size() + vec(centered_dims[:2])
                    pygame.draw.rect(win, color, (*pos, ceil(max_board_size()), ceil(max_board_size())))

        pygame.draw.rect(win, 'Dark Gray', centered_dims, 3)
    
    def step(self):
        if self.paused: return
        if self.since_fall % self.speed < 1:
            if not self.fall():    # piece lands
                for p in self.falling_piece.adj_shape():
                    self.set_pieces[int(p.y)][int(p.x)] = self.falling_piece.color # store colors in 2d array
                self.falling_piece = self.queued_pieces.pop(0)
                self.can_hold = True
                if len(self.queued_pieces) < 3:
                    self.queued_pieces.extend(Piece.new_bag())
                if self.falling_piece.check_collision(self.set_pieces):
                    print('Lose', self.falling_piece, self.falling_piece.adj_shape())
                    self.__init__()

                self.line_clear()
        self.since_fall += 1

    def move_falling(self, dir, move='TRANSLATE'):
        test_piece = deepcopy(self.falling_piece)

        if move == 'TRANSLATE':
            test_piece.pos += dir
        elif move == 'ROTATE':
            test_piece.rotate(dir)
        
        if test_piece.check_collision(self.set_pieces):
            return False

        self.falling_piece = test_piece
        return True
    
    def fall(self):
        return self.move_falling( vec(0, 1) )

    def line_clear(self):
        for i, row in enumerate(self.set_pieces):
            if all(row): # bool(None) == False, bool("color") == True
                self.set_pieces[i] = [ None ] * BOARD_DIM[0]
                for r in range(i, 0, -1): # shift lines down bottom to top
                    self.set_pieces[r], self.set_pieces[r-1] = self.set_pieces[r-1], self.set_pieces[r] # swap each

    def handle_inputs(self, inputs: List[str]) -> None:
        if 'ESCAPE' in inputs:
            self.paused = not self.paused
        if self.paused: return
        for inp in inputs:
            if inp in ['C', 'LSHIFT', 'RSHIFT']:
                self.hold()
            if inp in ['D', 'RIGHT']:
                self.move_falling( vec(1, 0) )
            if inp in ['A', 'LEFT']:
                self.move_falling( vec(-1, 0) )
            if inp == 'Z':
                self.move_falling(-1, 'ROTATE')
            if inp in ['X', 'UP']:
                self.move_falling(1, 'ROTATE')
            if inp in ['S', 'DOWN']:
                self.since_fall = 0
            if inp == 'SPACE':
                while self.fall():
                    pass

    def hold(self) -> None:
        print(self.held_piece)
        if not self.can_hold: return
        if self.held_piece is not None:
            self.falling_piece, self.held_piece = self.held_piece, self.falling_piece
            self.falling_piece.align_to_top()
        else:
            self.held_piece = deepcopy(self.falling_piece)
            self.falling_piece = self.queued_pieces.pop(0)

        self.can_hold = False

                



'''MAIN'''

pygame.init()
pygame.display.init()
# pygame.font.init()

win = pygame.display.set_mode((1000, 1000), pygame.RESIZABLE)
pygame.display.set_caption('Tetris')

clock = pygame.time.Clock()

game = TetrisGame()

game.draw(win)
pygame.display.update()

clock.tick(1)


running = True
while running:
    clock.tick(FPS)

    inputs = []
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            inp = event.key
            if inp is not None:
                inputs.append(inp)


    game.handle_inputs(nums_to_keys(inputs))

    win.fill('#000000')

    game.step()
    game.draw(win)

    pygame.display.update()


pygame.display.quit()
pygame.quit()