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
                dist = abs(self.goal[0] - nx) + abs(self.goal[1] - ny)
                return self.state, -1 - 0.1 * dist
        return self.state, -10
    
# --- 2.3 Q-Learning Agent ---
class QAgent:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions
        height, width = env.grid.shape
        self.q_table = np.random.uniform(low=-0.5, high=0.5, size=(height, width, len(self.actions)))

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, alpha, gamma):
        self.q_table[state][action] += alpha * (reward + gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])




