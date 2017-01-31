from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict

import gym
from gym import envs
from gym import wrappers


def epsilon_greedy_policy(Q, state, num_actions, epsilon):
    policy = np.ones(num_actions, dtype=float) * epsilon / num_actions
    best_action = np.argmax(Q[state])
    policy[best_action] += 1-epsilon
    return policy


def learn(
    env_n,\
    num_episodes,\
    outdir,\
    Q= defaultdict(lambda: np.zeros(env.action_space.n)),\
    discount_factor = 0.8,\
    folder_options = '',\
    delete_content=False,\
    monitor=False):
    
    #environment initialization
    envi = gym.make(env_n)
    outdir = outdir + env_n + folder_options + "/"
    global env
    env = (wrappers.Monitor(envi, outdir, video_callable=False, force=delete_content) if monitor else envi)

    er = np.zeros(num_episodes)
    el = np.zeors(num_episodes)

    for i_episode in range(1,num_episodes+1):
        #give response to the user
        if (i_episode%1000)==0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #initialize alpha and epsilon for each episode
        alpha = 0.5
        epsilon = (1-i_episode/num_episodes) ** 2

        state = env.reset()

        #calculate the e-greedy (behavior) policy given the Q function
        #and choose the action accordingly
        policy = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
        action = np.random.choice(np.array(len(policy)), p=policy)

        while True:
            next_state, reward, done, _= env.step(action)
            policy = epsilon_greedy_policy(Q, next_state, env.action_space.n, epsilon)
            next_action = np.random.choice(np.array(len(policy)), p=policy)

            # Update statistics
            er[i_episode-1] += reward
            el[i_episode-1] += 1

            #improvement of the Q function for the current (state, action) for the deterministic policy, following the e-greedy one
            delta = reward + discount_factor * np.max(Q[next_state]) - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * delta

            state = next_state; action = next_action;
            if done:
                break
    return Q, er, el


if __name__ == '__main__':
    Q, episode_reward, episode_length = learn(
        env_n='FrozenLake-v0',\
        num_episodes=1000,\
        outdir= "/Users/jacopo/openaigym/project/TD/results/SARSA_Lambda/",\
        replacing_traces=True,\
        folder_options = '78',\
        delete_content=True,\
        monitor=False)