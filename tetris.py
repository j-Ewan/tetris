import pygame
from pygame.math import Vector2 as vec
from math import ceil
from copy import deepcopy
import random
import os

# TODO List:
# - lose screen
# - highscore
# - controls
# - t-spin detection
# - tidy up piece display 
# - main screen?


FPS = 60

BOARD_DIM = (10, 20)
PIECE_DISP_SCALE = 0.75

KEYS = {32: 'SPACE', 13: 'RETURN', 113: 'Q', 119: 'W', 101: 'E', 114: 'R', 116: 'T', 121: 'Y', 117: 'U', 105: 'I', 111: 'O', 112: 'P', 97: 'A', 115: 'S', 100: 'D', 1073742053: 'RSHIFT', 102: 'F', 103: 'G', 104: 'H', 106: 'J', 107: 'K', 108: 'L', 122: 'Z', 120: 'X', 99: 'C', 118: 'V', 98: 'B', 110: 'N', 109: 'M', 49: '1', 50: '2', 51: '3', 52: '4', 53: '5', 54: '6', 55: '7', 56: '8', 57: '9', 48: '0', 44: ',', 46: '.', 59: ';', 47: '/', 39: "'", 91: '[', 93: ']', 92: '\\', 45: '-', 61: '=', 9: 'TAB', 8: 'BACKSPACE', 1073742049: 'LSHIFT', 96: '`', 1073741906: 'UP', 1073741905: 'DOWN', 1073741903: 'RIGHT', 1073741904: 'LEFT', 27: 'ESCAPE'}

FONT_FILE = os.path.join(os.path.dirname(__file__), 'arcadepix', 'ARCADEPI.TTF')

def nums_to_keys(nums):
    return [ KEYS[num] for num in nums if num in KEYS ]


def scale_factor():
    '''Returns board scale factor'''
    win_size = tuple(x * 0.9 for x in pygame.display.get_window_size())

    return min(win_size[0] / (BOARD_DIM[0] + 12 * PIECE_DISP_SCALE),
               win_size[1] / (BOARD_DIM[1] + 2))

def board_center():
    return vec(*(x/2 for x in pygame.display.get_window_size()),)


class Piece:
    shapes = { # relative distance from center
        'I': {'points': [ vec(-1, 0), vec(0, 0), vec(1, 0), vec(2, 0) ], 'center': vec(0.5, 0.5), 'color': '#00BFFF' },
        'J': {'points': [ vec(-1, -1), vec(-1, 0), vec(0, 0), vec(1, 0) ], 'center': vec(0, 0), 'color': '#0000FF' },
        'L': {'points': [ vec(-1, 0), vec(0, 0), vec(1, 0), vec(1, -1) ], 'center': vec(0, 0), 'color': '#FF7700' },
        'O': {'points': [ vec(0, 0), vec(0, 1), vec(1, 0), vec(1, 1) ], 'center': vec(0.5, 0.5), 'color': '#FFFF00' },
        'S': {'points': [ vec(-1, 0), vec(0, 0), vec(0, -1), vec(1, -1) ], 'center': vec(0, 0), 'color': '#00FF00' }, 
        'T': {'points': [ vec(-1, 0), vec(0, 0), vec(0, -1), vec(1, 0) ], 'center': vec(0, 0), 'color': '#9900FF' },
        'Z': {'points': [ vec(-1, -1), vec(0, -1), vec(0, 0), vec(1, 0) ], 'center': vec(0, 0), 'color': '#FF0000' }
    }

    @staticmethod
    def new_bag():
        pieces = list(Piece.shapes.keys())
        random.shuffle(pieces)
        bag = deepcopy([ Piece(vec(0, 0), type) for type in pieces ])
        # for item in bag: item.rotate(random.randint(0,3))
        for item in bag: item.align_to_top()

        return bag

    def __init__(self, pos: vec, type: str) -> None:
        self.pos = pos
        self.shape = Piece.shapes[type]['points']
        self.center = Piece.shapes[type]['center']
        self.color = Piece.shapes[type]['color']
        self.type = type
        self.rotation = 0

    def __repr__(self) -> str:
        return "Piece: points at {points} with color {color}".format(points=[tuple(v) for v in self.adj_shape()], color=self.color)

    def __str__(self) -> str:
        return f"{self.type} piece at {tuple(self.pos)} rotated {self.rotation}"

    def adj_shape(self):
        return [ square + self.pos for square in self.shape ]

    def draw(self, win: pygame.Surface, coords: vec) -> None:
        # draw on board
        size = scale_factor()
        for square in self.shape:
            pos = (self.pos + square) * size + coords

            pygame.draw.rect(win, self.color, (*pos, size+1, size+1))

    def display(self, win: pygame.Surface, x: float, y: float, w: float, h: float) -> None:
        # display for side
        # x, y in top left of center square
        x, y, w, h = map(ceil, (x, y, w, h))
        for square in self.shape:
            posx = x + square[0] * w
            posy = y + square[1] * h
            pygame.draw.rect(win, self.color, (posx, posy, w, h))

    

    def rotate(self, dir: int) -> None:
        self.rotation += dir
        self.rotation %= 4
        for i, p in enumerate(self.shape):
            p -= self.center
            p = p.rotate(90 * dir)
            p += self.center
            self.shape[i] = p

    def check_collision(self, pieces) -> bool:
        for square in self.adj_shape():
            if square[0] not in range(BOARD_DIM[0]) or square[1] not in range(BOARD_DIM[1]):
                return True
        if any( pieces[int(me.y)][int(me.x)] is not None for me in self.adj_shape() ):
            return True
        return False

    def align_to_top(self) -> None:

        self.pos.x = (BOARD_DIM[0] >> 1) - 1

        highest_y = min( p.y for p in self.adj_shape() )

        self.pos.y -= highest_y



class TetrisGame:
    def __init__(self) -> None:
        self.size = BOARD_DIM
        self.set_pieces = [ [ None ] * BOARD_DIM[0] for _ in range(BOARD_DIM[1]) ] # 2d array of colors
        self.falling_piece: Piece = None
        self.queued_pieces = []
        self.held_piece: Piece = None
        self.since_fall = 0
        self.paused = False
        self.can_hold = True
        self.level = 0
        self.score = 0
        self.lines_to_lvl = 0


        self.queued_pieces.extend(Piece.new_bag())
        self.falling_piece = self.queued_pieces.pop(0)

    def draw(self, win: pygame.Surface) -> None:
        size = scale_factor()
        center = board_center()
        board_dims = pygame.Rect(*center, *(x * size for x in BOARD_DIM))
        board_dims.center = center

        self.falling_piece.draw(win, vec(board_dims[:2]))

        for y, l in enumerate(self.set_pieces):
            for x, color in enumerate(l):
                if color is not None:
                    pos = vec(x, y) * size + vec(board_dims[:2])
                    pygame.draw.rect(win, color, (*pos, ceil(size), ceil(size)))

        pygame.draw.rect(win, 'Dark Gray', board_dims, 3)

        if self.held_piece is not None:
            x = center[0] - size * (BOARD_DIM[0]/2 + 3)
            y = center[1] - size * (BOARD_DIM[0]/2 + 2)
            self.held_piece.display(win, x, y, size * PIECE_DISP_SCALE, size * PIECE_DISP_SCALE)

        for ind, piece in enumerate(self.queued_pieces[:3]):
            x = center[0] + size * (BOARD_DIM[0]/2 + 3)
            y = center[1] - size * (BOARD_DIM[0]/2 + 2 - ind * 4)
            piece.display(win, x, y, size * PIECE_DISP_SCALE, size * PIECE_DISP_SCALE)

        font_size = round( size, -1 ) if size > 5 else 1
        font = pygame.freetype.Font(FONT_FILE, font_size)
        score_rect = font.get_rect(str(self.score), size = font_size)
        
        x = ceil(center[0] - size * (BOARD_DIM[0]/2))
        y = ceil(center[1] - size * (BOARD_DIM[0]) + 0.5)
        score_rect.bottomleft = (x, y)

        font.render_to(win, score_rect, str(self.score), "#FFFFFF")

        lvl_rect = font.get_rect(str(self.level), size = font_size)
        
        x = ceil(center[0] + size * (BOARD_DIM[0]/2))
        y = ceil(center[1] - size * (BOARD_DIM[0]) + 0.5)
        lvl_rect.bottomright = (x, y)

        font.render_to(win, lvl_rect, str(self.level), "#FFFFFF")
        
    
    def step(self):
        if self.paused: return
        speed = 0.03 + 0.01 * self.level
        if self.since_fall % (1/speed) < 1:
            if not self.fall():    # piece lands
                for p in self.falling_piece.adj_shape():
                    self.set_pieces[int(p.y)][int(p.x)] = self.falling_piece.color # store colors in 2d array
                self.falling_piece = self.queued_pieces.pop(0)
                self.can_hold = True
                if len(self.queued_pieces) < 3:
                    self.queued_pieces.extend(Piece.new_bag())
                if self.falling_piece.check_collision(self.set_pieces):
                    print('Lose', self.falling_piece, self.falling_piece.adj_shape())
                    print(self.score, 'points')
                    self.__init__()

                self.line_clear()
        self.since_fall += 1

    def move_falling(self, dir):
        test_piece = deepcopy(self.falling_piece)
        test_piece.pos += dir
        
        if test_piece.check_collision(self.set_pieces):
            return False

        self.falling_piece = test_piece
        return True

    def rotate_falling(self, rot):
        tables = {
            (0, 1): [vec(0, 0), vec(-1, 0), vec(-1, -1), vec(0, 2), vec(-1, 2)],
            (1, -1): [vec(0, 0), vec(1, 0), vec(1, 1), vec(0, -2), vec(1, -2)],
            (1, 1): [vec(0, 0), vec(1, 0), vec(1, 1), vec(0, -2), vec(1, -2)],
            (2, -1): [vec(0, 0), vec(-1, 0), vec(-1, -1), vec(0, 2), vec(-1, 2)],
            (2, 1): [vec(0, 0), vec(1, 0), vec(1, -1), vec(0, 2), vec(1, 2)],
            (3, -1): [vec(0, 0), vec(-1, 0), vec(-1, 1), vec(0, -2), vec(-1, -2)],
            (3, 1): [vec(0, 0), vec(-1, 0), vec(-1, 1), vec(0, -2), vec(-1, -2)],
            (0, -1): [vec(0, 0), vec(1, 0), vec(1, -1), vec(0, 2), vec(1, 2)]
        }

        i_tables = {
            (0, 1): [vec(0, 0), vec(-2, 0), vec(1, 0), vec(-2, 1), vec(1, -2)], #
            (1, -1): [vec(0, 0), vec(2, 0), vec(-1, 0), vec(2, -1), vec(-1, 2)], #
            (1, 1): [vec(0, 0), vec(-1, 0), vec(2, 0), vec(-1, -2), vec(2, 1)], #
            (2, -1): [vec(0, 0), vec(1, 0), vec(-2, 0), vec(1, 2), vec(-2, -1)], #
            (2, 1): [vec(0, 0), vec(2, 0), vec(-1, 0), vec(2, -1), vec(-1, 2)], # 
            (3, -1): [vec(0, 0), vec(-2, 0), vec(1, 0), vec(-2, 1), vec(1, -2)],
            (3, 1): [vec(0, 0), vec(1, 0), vec(-2, 0), vec(1, 2), vec(-2, -1)],
            (0, -1): [vec(0, 0), vec(-1, 0), vec(2, 0), vec(-1, -2), vec(2, 1)]
        }

        if self.falling_piece.type == 'I':
            tests = i_tables[(self.falling_piece.rotation, rot)]
        else:
            tests = tables[(self.falling_piece.rotation, rot)]

        test_piece = deepcopy(self.falling_piece)
        test_piece.rotate(rot)

        for pos in tests:
            test_piece.pos = self.falling_piece.pos + pos
            if not test_piece.check_collision(self.set_pieces):
                self.falling_piece = deepcopy(test_piece)
                return True
        else:
            print('Couldn\'t spin %s towards %s' % (self.falling_piece, rot))
        return False
    
    def fall(self):
        return self.move_falling( vec(0, 1) )

    def line_clear(self):
        lines = 0
        for i, row in enumerate(self.set_pieces):
            if all(row): # bool(None) == False, bool("color") == True
                lines += 1
                self.lines_to_lvl += 1
                self.set_pieces.pop(i)
                self.set_pieces.insert(0, [ None ] * BOARD_DIM[0])
        if lines == 0:
            return
        if lines == 1:
            self.score += 100 * (self.level + 1)
        elif lines == 2:
            self.score += 300 * (self.level + 1)
        elif lines == 3:
            self.score += 500 * (self.level + 1)
        elif lines == 4:
            self.score += 800 * (self.level + 1)
        if self.lines_to_lvl >= 10:
            self.level += 1
            self.lines_to_lvl -= 10
        

    def handle_inputs(self, inputs) -> None:
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
                self.rotate_falling(-1)
            if inp in ['X', 'UP']:
                self.rotate_falling(1)
            if inp in ['S', 'DOWN']:
                self.since_fall = 0
                self.score += 1
            if inp == 'SPACE':
                while self.fall():
                    self.score += 2
                self.since_fall = 0
            if inp == 'R':
                self.__init__()
            if inp == 'P':
                print(self.falling_piece)

    def hold(self) -> None:
        if not self.can_hold: return
        self.can_hold = False

        if self.held_piece is None:
            self.held_piece = deepcopy(self.falling_piece)
            self.falling_piece = self.queued_pieces.pop(0)
            return

        self.falling_piece, self.held_piece = self.held_piece, self.falling_piece
        self.falling_piece.align_to_top()

        

                



'''----------------MAIN----------------'''

pygame.init()
pygame.display.init()
pygame.font.init()

win = pygame.display.set_mode((1000, 1000), pygame.RESIZABLE, vsync=1)
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
