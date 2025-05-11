import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


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
    
# --- 2.3 Q-Learning Agent ---
class QAgent:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions
        height, width = env.grid.shape
        self.q_table = np.random.uniform(low=-0.5, high=0.5, size=(height, width, len(self.actions)))

    def choose_action(self, state, exploration_rate):
        if random.random() < exploration_rate:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, learning_rate, discount_factor):
        self.q_table[state][action] += learning_rate * (reward + discount_factor * np.max(self.q_table[next_state]) - self.q_table[state][action])

# --- 2.4 Q-learning Process ---
def q_learning(env, agent, episodes, max_steps, learning_rate, discount_factor, exploration_rate):
    start_time = time.time()
    for episode in tqdm(range(episodes)):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.choose_action(state, exploration_rate)
            next_state, reward = env.step(action)
            agent.update(state, action, reward, next_state, learning_rate, discount_factor)
            state = next_state
            if reward == 100:
                break
    training_time = time.time() - start_time
    return training_time

# --- 2.5 Evaluation Function ---
def model_evaluation(env, agent, episodes=100):
    state = env.reset()
    total_reward = 0
    for _ in range(episodes):
        action = np.argmax(agent.q_table[state])
        state, reward = env.step(action)
        total_reward += reward
        if reward == 100:
            return total_reward, True
    return total_reward, False

# --- 2.6 Run Experiments ---
def run(image_path, start, goal):
    grid = abstract_map(image_path, (50, 50))

    parameters = [
        (0.01, 0.9, 2000, 100),
        (0.01, 0.99, 2000, 100),
        (0.05, 0.9, 5000, 200),
        (0.05, 0.99, 5000, 200),
    ]

    total_rewards_all = []
    success_flags_all = []
    training_times_all = []
    labels = []

    for learning_rate, discount_factor, episodes, max_steps in parameters:
        label = f"lr={learning_rate}, df={discount_factor}"
        print(
            f"\nLearning Rate: {learning_rate}, Discount Factor: {discount_factor}, Episodes: {episodes}, Max Steps: {max_steps}")

        env = GridEnvironment(grid, start, goal)
        agent = QAgent(env)
        training_time = q_learning(env, agent, episodes, max_steps, learning_rate, discount_factor,
                                   exploration_rate=0.1)
        total_reward, reached_goal = model_evaluation(env, agent)
        print(f"Reached Goal: {reached_goal}, Total Reward: {total_reward:.2f}, Training Time: {training_time:.2f}s")

        total_rewards_all.append(total_reward)
        success_flags_all.append(int(reached_goal))
        training_times_all.append(training_time)
        labels.append(label)


    # Total Reward
    plt.figure(figsize=(10, 5))
    plt.bar(labels, total_rewards_all, color='skyblue')
    plt.title("Total Reward by Learning Parameters")
    plt.ylabel("Total Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Success Rate
    plt.figure(figsize=(10, 5))
    plt.bar(labels, success_flags_all, color='lightgreen')
    plt.title("Success (1=Reached Goal) by Learning Parameters")
    plt.ylabel("Success Flag")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Training Time
    plt.figure(figsize=(10, 5))
    plt.bar(labels, training_times_all, color='salmon')
    plt.title("Training Time by Learning Parameters")
    plt.ylabel("Training Time (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



run('maps/map1.bmp', start=(0, 0), goal=(49, 49))



