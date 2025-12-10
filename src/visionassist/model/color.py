import random

def get_random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))