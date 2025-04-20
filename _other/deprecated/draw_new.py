import pygame as p

def draw_text(text, font_size, color, x, y):
    # font = p.font.Font(None, font_size)
    font = p.font.SysFont("Arial", font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    screen.blit(text_surface, text_rect)


def draw_button(text, font_size, color, rect, x_pos=0, y_pos=0):
    rect.center = (x_pos, y_pos)
    p.draw.rect(screen, WHITE, rect)
    p.draw.rect(screen, GREY, rect, width=3)
    draw_text(text, font_size, color, rect.centerx, rect.centery)

def show_home_screen(win_height, win_length):
     p.draw.line(screen, WHITE,(0, (0.15 * win_height)), (win_length, (0.15 * win_height)))
     p.draw.line(screen, WHITE,(0, (0.75 * win_height)), (win_length, (0.75 * win_height)))

     learn_more_collider = p.Rect(0, win_height - (win_length/10) - 5, win_length * 0.85, win_length/10)
     try_it_collider = p.Rect(0, win_height - (win_length/10) - 5, win_length / 3, win_length/10)
     draw_button("Learn more about our neural network", 20, BLACK, learn_more_collider, win_length//2, win_height * .80)
     draw_button("Try it now", 20, BLACK, try_it_collider, win_length//2, win_height * .90)

     draw_text('Made by: Christian Host-Madsen (chost-madsen25@punahou.edu) Mason Morales (mmorales25@punahou.edu)', 10, WHITE, win_length / 2, win_height * 0.965)
     draw_text('Isaac Verbrugge (isaacverbrugge@gmail.com) Derek Yee (dyee25@punahou.edu)', 10, WHITE, win_length / 2, win_height * 0.98)
     

# initialize pygame
p.init()

# background color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
# window scale
win_scale = 20
win_height = 28 * win_scale + (28 * 0.4 * win_scale)
win_length = 28 * win_scale
screen = p.display.set_mode((win_length, win_height))
running = True

while running:
    show_home_screen(win_height, win_length)
    p.display.flip()

    for event in p.event.get():
            if event.type == p.QUIT:
                running = False