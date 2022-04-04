import pygame
from pygame.math import Vector2 as vec
from math import ceil
from copy import deepcopy
import random
import os

# TODO List:
# - DAS
# - settings.txt

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.txt')
with open(SETTINGS_FILE) as f:
    settings = eval(f.read())

FPS = 60

BOARD_DIM = (10, 20)
PIECE_DISP_SCALE = 0.75


highscore = None

FONT_FILE = os.path.join(os.path.dirname(__file__), 'arcadepix', 'ARCADEPI.TTF')


def scale_factor():
    '''Returns board scale factor'''
    win_size = tuple(x * 0.9 for x in pygame.display.get_window_size())

    return min(win_size[0] / (BOARD_DIM[0] + 12 * PIECE_DISP_SCALE),
               win_size[1] / (BOARD_DIM[1] + 2))

def board_center():
    return vec(*(x/2 for x in pygame.display.get_window_size()),)


class Piece:
    shapes = { # relative distance from center
        'I': {'points': [ vec(-1, 0), vec(0, 0), vec(1, 0), vec(2, 0) ], 'color': '#00BFFF' },
        'J': {'points': [ vec(-1, -1), vec(-1, 0), vec(0, 0), vec(1, 0) ], 'color': '#0000FF' },
        'L': {'points': [ vec(-1, 0), vec(0, 0), vec(1, 0), vec(1, -1) ], 'color': '#FF7700' },
        'O': {'points': [ vec(0, 0), vec(0, -1), vec(1, 0), vec(1, -1) ], 'color': '#FFFF00' },
        'S': {'points': [ vec(-1, 0), vec(0, 0), vec(0, -1), vec(1, -1) ], 'color': '#00FF00' }, 
        'T': {'points': [ vec(-1, 0), vec(0, 0), vec(0, -1), vec(1, 0) ], 'color': '#9900FF' },
        'Z': {'points': [ vec(-1, -1), vec(0, -1), vec(0, 0), vec(1, 0) ], 'color': '#FF0000' }
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
        self.color = Piece.shapes[type]['color']
        self.type = type
        self.rotation = 0

    def __repr__(self) -> str:
        return "Piece: points at {points} with color {color}".format(points=[tuple(v) for v in self.adj_shape()], color=self.color)

    def __str__(self) -> str:
        return f"{self.type} piece at {tuple(self.pos)} rotated {self.rotation}"

    def adj_shape(self):
        return [ square + self.pos for square in self.shape ]

    def draw(self, win, coords):
        size = scale_factor()
        for square in self.shape:
            pos = (self.pos + square) * size + coords

            pygame.draw.rect(win, self.color, (*pos, size+1, size+1))

    def ghost_draw(self, win, coords):
        size = scale_factor()
        for square in self.shape:
            pos = (self.pos + square) * size + coords

            pygame.draw.rect(win, self.color, (*pos, size+1, size+1), 1)

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
            p = p.rotate(90 * dir)
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
        self.lock_delay = 0
        self.ld_max = 3


        self.queued_pieces.extend(Piece.new_bag())
        self.falling_piece = self.queued_pieces.pop(0)

    def draw(self, win: pygame.Surface) -> None:
        size = scale_factor()
        center = board_center()
        board_dims = pygame.Rect(*center, *(x * size for x in BOARD_DIM))
        board_dims.center = center
        
        self.draw_ghost_falling(win)

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

        for ind, piece in enumerate(self.queued_pieces[:5]):
            x = center[0] + size * (BOARD_DIM[0]/2 + 3)
            y = center[1] - size * (BOARD_DIM[0]/2 + 2 - ind * 3) 
            piece.display(win, x, y, size * PIECE_DISP_SCALE, size * PIECE_DISP_SCALE)

        font_size = round( size, -1 ) if size > 5 else 1
        font = pygame.freetype.Font(FONT_FILE, font_size)
        score_rect = font.get_rect(str(self.score), size = font_size)
        
        x = ceil(center[0] - size * (BOARD_DIM[0]/2))
        y = ceil(center[1] - size * (BOARD_DIM[0] + 0.1))
        score_rect.bottomleft = (x, y)

        font.render_to(win, score_rect, str(self.score), "#FFFFFF")

        if highscore is not None:
            highscore_rect = font.get_rect('HI: ' + str(highscore), size = font_size)
        
            x = ceil(center[0] - size * (BOARD_DIM[0]/2))
            y = ceil(center[1] - size * (BOARD_DIM[0] + 1.2))
            highscore_rect.bottomleft = (x, y)

            font.render_to(win, highscore_rect, 'HI: ' + str(highscore), "#FFFFFF")


        lvl_rect = font.get_rect(str(self.level), size = font_size)
        
        x = ceil(center[0] + size * (BOARD_DIM[0]/2))
        y = ceil(center[1] - size * (BOARD_DIM[0] + 0.1))
        lvl_rect.bottomright = (x, y)

        font.render_to(win, lvl_rect, str(self.level), "#FFFFFF")
        
    
    def step(self):
        if self.paused: return
        speed = 0.03 + 0.01 * self.level
        if self.since_fall % (1/speed) < 1:
            if not self.fall():    # piece lands
                self.lock_delay += 1
            else:
                self.lock_delay = 0
            if self.lock_delay >= self.ld_max:
                tspin = False
                if self.falling_piece.type == 'T':
                    tspin = self.check_spin()
                for p in self.falling_piece.adj_shape():
                    self.set_pieces[int(p.y)][int(p.x)] = self.falling_piece.color # store colors in 2d array
                
                self.line_clear(tspin=tspin)

                self.falling_piece = self.queued_pieces.pop(0)
                self.can_hold = True
                if len(self.queued_pieces) < 6:
                    self.queued_pieces.extend(Piece.new_bag())
                if self.falling_piece.check_collision(self.set_pieces):
                    print('Lose')
                    print(self.score, 'points')
                    global highscore
                    if highscore is None or self.score > highscore:
                        highscore = self.score
                    self.__init__()
        self.since_fall += 1

    def draw_ghost_falling(self, win):
        ghost_piece = deepcopy(self.falling_piece)
        while not ghost_piece.check_collision(self.set_pieces):
            ghost_piece.pos += vec(0, 1)

        ghost_piece.pos += vec(0, -1)

        size = scale_factor()
        center = board_center()
        board_dims = pygame.Rect(*center, *(x * size for x in BOARD_DIM))
        board_dims.center = center

        ghost_piece.ghost_draw(win, vec(board_dims[:2]))

    def move_falling(self, dir):
        test_piece = deepcopy(self.falling_piece)
        test_piece.pos += dir
        
        if test_piece.check_collision(self.set_pieces):
            return False

        self.falling_piece = test_piece
        return True

    def rotate_falling(self, rot):
        tables = {
            0: [vec(0,0)]*5, 
            1: [vec(0,0), vec(1,0), vec(1,1), vec(0,-2), vec(1,-2)],
            2: [vec(0,0)]*5,
            3: [vec(0,0), vec(-1,0), vec(-1,1), vec(0,-2), vec(-1,-2)],
        }

        tables_i = {
            0: [vec(0,0), vec(-1,0), vec(2,0), vec(-1,0), vec(2,0)],
            1: [vec(-1,0), vec(0,0), vec(0,0), vec(0,-1), vec(0,2)],
            2: [vec(-1,-1), vec(1,-1), vec(-2, -1), vec(1,0), vec(-2,0)],
            3: [vec(0,-1), vec(0,-1), vec(0, -1), vec(0,1), vec(0,-2)],
        }

        tables_o = {
            0: [vec(0, 0)],
            1: [vec(0, 1)],
            2: [vec(-1, 1)],
            3: [vec(-1, 0)],
        }
        

        current_rot = self.falling_piece.rotation
        use_table = None
        if self.falling_piece.type == 'I':
            use_table = tables_i
        elif self.falling_piece.type == 'O':
            use_table = tables_o
        else:
            use_table = tables

        tests = [ v1 - v2 
                 for v1, v2 in 
                  zip(use_table[current_rot],
                      use_table[(current_rot + rot) % 4])
                ]



        test_piece = deepcopy(self.falling_piece)
        test_piece.rotate(rot)

        for pos in tests:
            test_piece.pos = self.falling_piece.pos + pos
            if not test_piece.check_collision(self.set_pieces):
                self.falling_piece = deepcopy(test_piece)
                return True
        else:
            pass
        return False
    
    def fall(self):
        return self.move_falling( vec(0, 1) )

    def line_clear(self, tspin = False):
        num_lines = 0
        for i, row in enumerate(self.set_pieces):
            if all(row): # bool(None) == False, bool("color") == True
                num_lines += 1
                self.lines_to_lvl += 1
                self.set_pieces.pop(i)
                self.set_pieces.insert(0, [ None ] * BOARD_DIM[0])
        if num_lines == 0:
            return
        self.score += self.calc_score(num_lines, tspin)
        if self.lines_to_lvl >= 10:
            self.level += 1
            self.lines_to_lvl -= 10


    def calc_score(self, lines, tspin):
        spin_reward = 4 if tspin else 1
        if lines == 1:
            return 100 * (self.level + 1) * spin_reward
        elif lines == 2:
            return 300 * (self.level + 1) * spin_reward
        elif lines == 3:
            return 500 * (self.level + 1) * spin_reward
        elif lines == 4:
            return 800 * (self.level + 1) * spin_reward

    def check_spin(self):
        test_piece = deepcopy(self.falling_piece)
        spin = True
        for shift in [vec(-1, 0), vec(0, -1), vec(1, 0)]:
            test_piece.pos = deepcopy(self.falling_piece.pos)
            test_piece.pos += shift
            if not test_piece.check_collision(self.set_pieces):
                spin = False
        return spin

    def handle_inputs(self, inputs) -> None:
        if pygame.K_ESCAPE in inputs:
            self.paused = not self.paused
        if self.paused: return
        for inp in inputs:
            if inp in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                self.hold()
            if inp in [pygame.K_d, pygame.K_RIGHT]:
                self.move_falling( vec(1, 0) )
            if inp in [pygame.K_a, pygame.K_LEFT]:
                self.move_falling( vec(-1, 0) )
            if inp == pygame.K_z:
                self.rotate_falling(-1)
            if inp in [pygame.K_x, pygame.K_UP]:
                self.rotate_falling(1)
            if inp == pygame.K_c:
                self.rotate_falling(2)
            if inp in [pygame.K_s, pygame.K_DOWN]:
                if self.fall():
                    self.since_fall = 1
                    self.score += 1
            if inp == pygame.K_SPACE:
                while self.fall():
                    self.score += 2
                self.lock_delay = self.ld_max
                self.since_fall = 0
                self.step()
            if inp == pygame.K_r:
                self.__init__()
            if inp == pygame.K_p:
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

logo_file = os.path.join(os.path.dirname(__file__), 'dt-logo.png')
logo = pygame.image.load(logo_file)
pygame.display.set_icon(logo)

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


    game.handle_inputs(inputs)

    win.fill('#000000')

    game.step()
    game.draw(win)

    pygame.display.update()

pygame.display.quit()
pygame.quit()
