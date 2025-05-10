import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 2.1 Map Abstraction ---
def abstract_map(image_path, target_size):
    image = Image.open(image_path)
    binary = np.array(image) < 128
    height, width = binary.shape
    height_new, width_new = target_size
    block_height, block_width = height // height_new, width // width_new
    abstracted = np.zeros((height_new, width_new), dtype=int)

    for i in range(height_new):
        for j in range(width_new):
            block = binary[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            abstracted[i, j] = 1 if np.any(block) else 0
    return abstracted

# --- 2.2 Environment Class ---
class GridEnvironment:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.state = start
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dx, dy = self.actions[action]
        x, y = self.state
        nx, ny = x + dx, y + dy
        height, width = self.grid.shape

        if 0 <= nx < width and 0 <= ny < height:
            if self.grid[nx, ny] == 0:
                self.state = (nx, ny)
                if self.state == self.goal:
                    return self.state, 100
                return self.state, -1
        return self.state, -10
    
    