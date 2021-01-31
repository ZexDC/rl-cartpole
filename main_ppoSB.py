import os
import gym
import numpy as np

import torch
from stable_baselines3 import PPO
from utils import plot_curve

from cartpole_env1c import CartPoleEnv as env1c
from cartpole_env3c import CartPoleEnv as env3c

# Environment
env_name = "Env3cont"
env = env3c() # change custom env here

# Development settings
if not os.path.exists('plots'):
        os.makedirs('plots')
if not os.path.exists('models'):
        os.makedirs('models')
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

# Progress parameters
episodes = 300
reward_sum = 0
reward_history = []
dones=0

model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=10000)

obs = env.reset()
#for i in range(25000):
while(dones<episodes):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    reward_sum += reward
    
    if done:
      dones=dones+1
      print("reward: ", reward_sum)
      reward_history.append(reward_sum)
      reward_sum = 0
      obs = env.reset()
      

env.close()
plot_curve(reward_history, "plots/cartpole_{}_SB_test.png".format(env_name))
