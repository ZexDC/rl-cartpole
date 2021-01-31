import os
import numpy as np
import matplotlib.pyplot as plt

def plot_curve(episode_r_history, file):
    running_avg = np.zeros(len(episode_r_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(episode_r_history[max(0, i-100):(i+1)])
    plt.grid(1)
    plt.plot(running_avg)
    plt.title('Reward over episodes')
    plt.savefig(file)

def random(env):
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)