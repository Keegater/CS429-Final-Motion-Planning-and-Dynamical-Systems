import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

class CruiseControlEnv:
    def __init__(self, x_range=(-5, 5), v_range=(-5, 5), n_x=21, n_v=21,
                 control_inputs=np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001,
                                          0, 0.0001, 0.001, 0.01, 0.1, 1, 5]),
                 delta=0.1):

        self.X = np.linspace(x_range[0], x_range[1], n_x)
        self.V = np.linspace(v_range[0], v_range[1], n_v)
        self.n_x, self.n_v = n_x, n_v
        self.u_list = control_inputs
        self.n_u = len(control_inputs)

        self.delta = delta
        self._t_span = np.linspace(0, self.delta, 2)  # just start & end

    def _dynamics(self, s, t, u):
        x, v = s
        return [v, u]

    def step(self, s, u):
        traj = odeint(self._dynamics, s, self._t_span, args=(u,))
        s_next = traj[-1]
        # clip bounds
        s_next[0] = np.clip(s_next[0], self.X[0], self.X[-1])
        s_next[1] = np.clip(s_next[1], self.V[0], self.V[-1])
        return s_next

    def discretize(self, s):
        # find nearest grid point
        i = np.argmin(np.abs(self.X - s[0]))
        j = np.argmin(np.abs(self.V - s[1]))
        return i, j

    def reward(self, s, u, s_next, alpha=1.0, beta=1.0, gamma=0.01, R_goal=100.0, tol=0.05):
        # step penalty - punishes more steps (rewards quicker parking)
        r = -1.0
        # state‐shaping penalty - punishes distance from origin (rewards progress towards origin)
        x_next, v_next = s_next
        r -= alpha * abs(x_next) + beta * abs(v_next)

        # terminal bonus - Large reward if next state is very close to origin
        if abs(x_next) < tol and abs(v_next) < tol:
            r += R_goal

        return r

    def simulate_trajectory(self, s0, Q, max_steps=500, tol=0.05):
        traj = [s0.copy()]
        s = s0.copy()
        for _ in range(max_steps):
            i, j = self.discretize(s)
            # pick action with highest Q‐value
            u_idx = np.argmax(Q[i, j, :])
            u = self.u_list[u_idx]
            s = self.step(s, u)
            traj.append(s.copy())

            if np.hypot(s[0], s[1]) < tol:
                break
        return np.array(traj)




class QLearningAgent:
    def __init__(self, n_x: int, n_v: int, n_u: int,
                 alpha: float = 0.1, # learning rate
                 gamma: float = 0.99, # discount
                 epsilon: float = 0.1): # exploration prob
        self.Q = np.zeros((n_x, n_v, n_u))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[2])
        i, j = state_idx
        return int(np.argmax(self.Q[i, j, :]))

    def update(self, state_idx, action_idx, reward, next_state_idx):
        i, j = state_idx
        i2, j2 = next_state_idx
        q_pred = self.Q[i, j, action_idx]
        q_target = reward + self.gamma * np.max(self.Q[i2, j2, :])
        self.Q[i, j, action_idx] += self.alpha * (q_target - q_pred)


def train_q_learning(env,
                     agent: QLearningAgent,
                     num_episodes: int = 5000,
                     max_steps: int = 500,
                     init_states: np.ndarray = None,
                     tol: float = 0.05):

    if init_states is None:
        init_states = [np.array([5.0, 0.0])]

    for ep in range(num_episodes):

        s = init_states[np.random.randint(len(init_states))].copy()

        for t in range(max_steps):
            # discretize
            idx = env.discretize(s)

            a_idx = agent.choose_action(idx)
            u = env.u_list[a_idx]
            s_next = env.step(s, u)

            r = env.reward(s, u, s_next)
            idx_next = env.discretize(s_next)

            agent.update(idx, a_idx, r, idx_next)

            s = s_next

            # stop if parked
            if np.hypot(s[0], s[1]) < tol:
                break

    return agent


def evaluate_q_table(env,
                     Q: np.ndarray,
                     init_states: np.ndarray,
                     num_episodes: int = 100,
                     max_steps: int = 500,
                     tol: float = 0.05):

    successes = 0
    returns = []
    lengths = []

    for k in range(num_episodes):

        s = init_states[k % len(init_states)].copy()
        total_r = 0.0

        for t in range(1, max_steps + 1):
            # discretize
            i, j = env.discretize(s)
            a_idx = int(np.argmax(Q[i, j, :]))
            u = env.u_list[a_idx]

            s_next = env.step(s, u)
            r = env.reward(s, u, s_next)
            total_r += r
            s = s_next

            if abs(s[0]) < tol and abs(s[1]) < tol:
                successes += 1
                lengths.append(t)
                returns.append(total_r)
                break
        else:
            lengths.append(max_steps)
            returns.append(total_r)

    success_rate = successes / num_episodes
    avg_return = np.mean(returns)
    avg_length = np.mean(lengths)
    return success_rate, avg_return, avg_length



def run_experiment(args):
    n, alpha, gamma, epsilon, num_eps, max_steps, init_states, save_dir = args

    # rebuild env
    env = CruiseControlEnv(
        n_x=n, n_v=n,
        x_range=(-5, 5), v_range=(-5, 5))

    agent = QLearningAgent(
        n_x=n, n_v=n, n_u=env.n_u,
        alpha=alpha, gamma=gamma, epsilon=epsilon
    )

    # train
    trained = train_q_learning(
        env, agent,
        num_episodes=num_eps,
        max_steps=max_steps
    )

    # evaluate
    sr, ar, al = evaluate_q_table(
        env, trained.Q,
        init_states=init_states,
        num_episodes=100,
        max_steps=500
    )

    dt = env.delta
    traj = env.simulate_trajectory(np.array([5, 0]), trained.Q)
    t2 = np.linspace(0, dt * len(traj), len(traj))
    plt.figure(figsize=(8, 4))
    plt.plot(t2, traj[:, 0], 'r-', label='position $x(t)$')
    plt.plot(t2, traj[:, 1], 'b-', label='velocity $v(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel('States')
    plt.title(f"agent_bins{n}_a{alpha}_g{gamma}_e{epsilon}_ep{num_eps}_ms{max_steps}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save data
    fname = (f"agent_bins{n}_a{alpha}_g{gamma}_e{epsilon}_ep{num_eps}_ms{max_steps}.pkl")
    path = os.path.join(save_dir, fname)
    with open(path, "wb") as f:
        pickle.dump(trained, f)

    return {
        "n_bins": n,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "episodes": num_eps,
        "max_steps": max_steps,
        "success_rate": sr,
        "avg_return": ar,
        "avg_length": al
    }


if __name__ == "__main__":

    np.random.seed(42)

    discretizations = [21, 51, 101]
    alphas = [0.1, 0.5]
    gammas = [0.9, 0.99]
    epsilons = [0.05, 0.1, 0.2]
    training_settings = [ (1000, 200), (5000, 500), (10000, 500)]
    init_states = np.random.uniform(
        low=[-5.0, -5.0],
        high=[5.0, 5.0],
        size=(100, 2)
    )
    # make folder to save models
    save_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(save_dir, exist_ok=True)

    configs = []

    for (n, alpha, gamma, (num_eps, max_steps), epsilon) in itertools.product( discretizations, alphas, gammas, training_settings, epsilons):
        configs.append((n, alpha, gamma, epsilon, num_eps, max_steps,
                        init_states, save_dir))

    results = []
    # run in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, cfg) for cfg in configs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running experiments"):
            results.append(future.result())

    df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)
    dfpath = os.path.join(save_dir, "dataframe.pkl")
    df.to_pickle(dfpath)
    print("\nAvg success rate by discretization:")
    print(df.groupby("n_bins").success_rate.mean())

