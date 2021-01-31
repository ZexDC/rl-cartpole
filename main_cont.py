# Code adapted from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from utils import plot_curve
from ppo_cont import PPO, Memory

import gym
from stable_baselines3.common.env_checker import check_env
from cartpole_env1c import CartPoleEnv as env1c
from cartpole_env2c import CartPoleEnv as env2c
from cartpole_env3c import CartPoleEnv as env3c


if __name__ == '__main__':

    train = False
    test = True
    render = True

    # Environment
    env_name = "Env3cont"
    env = env3c() # change custom env here
    #check_env(env) # check custom environment and output additional warnings if needed
    n_actions = env.action_space.shape[0] # continuous
    n_states = env.observation_space.shape[0]

    # Development settings
    if not os.path.exists('plots'):
            os.makedirs('plots')
    if not os.path.exists('models'):
            os.makedirs('models')
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # Agent hyperparameters
    gamma=0.99 # discount
    alpha=0.0004 # adam optimizer stepsize
    gae_lambda=0.95
    policy_clip=0.2
    batch_size=5
    n_epochs=10
    action_std = 0.5 # constant std for action distribution (Multivariate Normal)
    betas = (0.9, 0.999)

    # Agent initialization
    agent = PPO(n_states, n_actions, action_std, alpha, betas, gamma, n_epochs, policy_clip)
    memory = Memory()

    # Reward functions
    reward_f = 1

    # Progress parameters
    n_episodes = 5000
    n_steps = 1500
    n_eval = 4000 # make the agent learn again after n steps
    reward_sum = 0
    reward_avg = 0
    reward_threshold = 5000 # solved episode beyond this reward
    consecutive_solved = 0
    n_solved = 100 # solved episodes required to finish
    reward_history = []
    time_step = 0

    if train:
        print("Starting training...")
        for i in range(n_episodes):
            observation = env.reset()
            done = False
            reward_sum = 0
            for k in range(n_steps):
                time_step += 1
                action = agent.select_action(observation, memory)
                observation_, reward, done, info = env.step(action)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                # update if its time
                if time_step % n_eval == 0:
                    agent.update(memory)
                    memory.clear_memory()
                    time_step = 0
                reward_sum += reward
                observation = observation_
                if render:
                    env.render()
                if done:
                    break
            reward_history.append(reward_sum)
            reward_avg = np.mean(reward_history[-100:])
            if reward_avg >= reward_threshold:
                consecutive_solved += 1
            if consecutive_solved == n_solved:
                reward_best = reward_avg
                torch.save(agent.policy.state_dict(),
                       './models/PPO_continuous_solved_{}'.format(i))
                print("Solved at episode ", i, "with Avg reward %.1f" % reward_avg)
                break
            print('Episode', i, 'Reward %.1f' % reward_sum, 'Avg reward %.1f' % reward_avg)
        
        # Plot the results of training
        axis_x = [episode+1 for episode in range(len(reward_history))]
        plot_curve(reward_history, "plots/cartpole_cont_{}_train.png".format(env_name))


    if test: 
        print("Starting testing...")
        n_episodes = 300
        max_steps = 200
        reward_sum = 0
        reward_avg = 0
        reward_best = 0
        reward_history = []

        #filename = "./models/env1/0003-64/PPO_continuous_solved_1037" # env1 0003-64
        #filename = "./models/env1/0004-5/PPO_continuous_solved_919" # env1 0004-5
        #filename = "./models/env2/0003-64/PPO_continuous_solved_1699" # env2 0003-64
        #filename = "./models/env2/0004-5/PPO_continuous_solved_1521" # env2 0004-5
        filename = "./models/env3/0003-64/PPO_continuous_solved_1395" # env3 0003-64
        #filename = "./models/env3/0004-5/PPO_continuous_solved_1260" # env3 0004-5
        agent.policy_old.load_state_dict(torch.load(filename))

        for i in range(n_episodes):
            print("Episode", i+1, "/", n_episodes)
            observation = env.reset()
            done = False
            reward_sum = 0
            n_steps = 0
            t = 0 # timesteps so far
            ep_len = 0 # episodic length
            while not done:
                #t += 1
            #for k in range(n_steps): # uncomment this and below for early stop
                if render:
                    env.render()
                action = agent.select_action(observation, memory)
                observation_, reward, done, info = env.step(action)
                # Random agent
                # action = env.action_space.sample()
                # observation_, reward, done, info = env.step(action)
                reward_sum += reward
                observation = observation_
                n_steps += 1
                #if n_steps > max_steps:
                    #done = True
            ep_len = t
            reward_history.append(reward_sum)
            reward_avg = np.mean(reward_history[-100:])

            print('Episode', i, 'Reward %.1f' % reward_sum, 'Avg reward %.1f' % reward_avg)

        plot_curve(reward_history, "plots/cartpole_{}_test.png".format(env_name))
