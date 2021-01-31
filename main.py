# Discrete implementation

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from utils import plot_curve
from ppo import Agent

import gym
from stable_baselines3.common.env_checker import check_env
from cartpole_env1 import CartPoleEnv as env1
from cartpole_env2 import CartPoleEnv as env2
from cartpole_env3 import CartPoleEnv as env3


if __name__ == '__main__':

    train = True
    test = False
    render = False

    # Environment
    env_name = "Env2"
    env = env2() # change custom env here
    check_env(env) # check custom environment and output additional warnings if needed
    n_actions = env.action_space.n # discrete
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

    # Agent initialization
    agent = Agent(n_actions=n_actions, n_states=n_states, gamma=gamma, 
                alpha=alpha, gae_lambda=gae_lambda, policy_clip=policy_clip,
                batch_size=batch_size, n_epochs=n_epochs)

    # Reward functions
    reward_f = 1

    # Progress parameters
    n_episodes = 5000
    n_steps = 200
    n_eval = 20 # make the agent learn again after n steps
    reward_sum = 0
    reward_avg = 0
    reward_threshold = 195. # solved episode beyond this reward
    consecutive_solved = 0
    n_solved = 100 # solved episodes required to finish
    reward_history = []

    if train:
        print("Starting training...")
        for i in range(n_episodes):
            observation = env.reset()
            done = False
            reward_sum = 0
            for k in range(n_steps):
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                agent.remember(observation, action, prob, val, reward, done)
                if k % n_eval == 0:
                    agent.learn()
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
                agent.save_models()
                print("Solved at episode ", i, "with Avg reward %.1f" % reward_avg)
                break
            print('Episode', i, 'Reward %.1f' % reward_sum, 'Avg reward %.1f' % reward_avg)
        
        # Plot the results of training
        axis_x = [episode+1 for episode in range(len(reward_history))]
        plot_curve(reward_history, "plots/cartpole_{}_train.png".format(env_name))


    if test: 
        print("Starting testing...")
        n_episodes = 50
        max_steps = 200
        reward_sum = 0
        reward_avg = 0
        reward_best = 0
        reward_history = []

        agent.load_models()

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
                action, prob, val = agent.choose_action(observation)
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
