import pygame
import config
import time

TS = 100 # timestep in ms
CONTINUOUS = True # continues past render height

s = config.CELLSIZE
h = config.HEIGHT

colors = [
    [(0,0,0), (255,255,255)],
    [(200,0,0), (255,200,200)],
    [(0,200,0), (200,255,200)],
    [(0,0,200), (200,200,255)],
    [(0,255,255), (200,255,255)],
]

class EcaVisualizer(object):
    def __init__(self, reservoir = None, colour = False):
        self.w = config.WIDTH
        self.colour = colour

        if reservoir is not None:
            self.reservoir = reservoir
            self.w = reservoir.width


        pygame.init()
        pygame.display.set_caption("ECA Reservoir")
        self.screen = pygame.display.set_mode((self.w * s + 1, h * s + 1))
        self.screen.fill((255, 255, 255))
        self.clock = pygame.time.Clock()

    def draw(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.save_image()
        
        # if (CONTINUOUS and (len(self.reservoir.rows) >= h)):
        self.screen.fill((255, 255, 255))

        dif = 0
        if self.reservoir.generation > config.HEIGHT:
            if self.reservoir.iterations > 0:
                dif = (self.reservoir.generation - config.HEIGHT) % self.reservoir.iterations
                dif = (self.reservoir.iterations - 1) - dif

        for j, row in enumerate(self.reservoir.rows):
            for i, cell in enumerate(row):
                c = self.get_color(i, j, cell, dif)
                if s > 1:
                    pygame.draw.rect(self.screen, c, [i*s, j*s, s, s])
                else:
                    self.screen.set_at((i, j), c)

        '''
        If drawing extreamly large reservoirs, uncomment below code (and line 45) to only draw one row at a time
        '''
        # else:
        #     self.hv_lines()
        #     for i, cell in enumerate(self.reservoir.cells):
        #         if cell == 1:
        #             c = self.get_color(i)
        #             if s > 1:
        #                 pygame.draw.rect(self.screen, c, [i*s, self.reservoir.generation*s, s, s])
        #             else:
        #                 self.screen.set_at((i, self.reservoir.generation), c)
        #     pygame.display.update([0, s*self.reservoir.generation, self.w*s, h*s])

        # if (CONTINUOUS and (len(self.reservoir.rows) >= h)):
        self.hv_lines()
        pygame.display.flip()

        self.clock.tick(1000/TS)

    def save_image(self):
        '''
        saves a image of the current screen as an image
        '''
        pygame.image.save(self.screen, f"screenshots/rule{int(self.reservoir.rule)}_{int(time.time())}.png")

    def hv_lines(self):
        if s > 1:
            # Horizontal lines
            for i in range(h + 1):
                y = i * s
                pygame.draw.line(self.screen, (120,120,120), (0, y), (self.w * s, y))

            # Verticle lines
            for i in range(self.w + 1):
                x = i * s
                pygame.draw.line(self.screen, (120,120,120), (x, 0), (x, y * s))

    def get_color(self, x, y, cell, dif):
        if self.colour:
            if self.reservoir.iterations == 0 or y % self.reservoir.iterations == dif:
                if (x in self.reservoir.obs_mappings):
                    index = self.reservoir.obs_mappings.index(x)
                    index = int(index / self.reservoir.acc_per_obs)
                    c = colors[index + 1]
                    c = c[0] if cell == 1 else c[1]
                    return c
        
        c = colors[0][0] if cell == 1 else colors[0][1]
        return c

            
    def reset(self):
        self.screen.fill((255, 255, 255))
