

from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict

import gym
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
    lambda_factor = 0.9,\
    replacing_traces=True,\
    folder_options = '',\
    delete_content=False,\
    monitor=False):
    envi = gym.make(env_n)
    outdir = outdir + env_n + folder_options + "/"
    global env
    env = (wrappers.Monitor(envi, outdir, video_callable=False, force=delete_content) if monitor else envi)
    
    er = np.zeros(num_episodes)
    el = np.zeros(num_episodes)
    
    for i_episode in range(1,num_episodes+1):
        if (i_episode%1000)==0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #initialization: use a list of (state, action) pair to keep
        #track of the episode "history"
        episode = []
        E = defaultdict(lambda: np.zeros(env.action_space.n))
        alpha = 1 - i_episode/num_episodes
        epsilon = (1 - i_episode/num_episodes) ** 2
        #initial state
        state = env.reset()
        #calculate the e-greedy policy for the init state given the Q function
        #and choose the action accordingly
        policy = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
        action = np.random.choice(np.array(len(policy)), p=policy)
        while True:
            if ((state, action) not in episode):
                episode.append((state, action))
            next_state, reward, done, _= env.step(action)
            policy = epsilon_greedy_policy(Q, next_state, env.action_space.n, epsilon)
            next_action = np.random.choice(np.array(len(policy)), p=policy)

            r = -1 if (done and reward==0) else reward
            
            # Update statistics
            er[i_episode-1] += reward
            el[i_episode-1] += 1

            #updating the trace using the replacing or the eligibility
            if replacing_traces:
                E[state][action]  = 1
            else:
                E[state][action]  += 1

            #improvement of the Q value function for each state visited in the current episode
            delta = r + discount_factor * Q[next_state][next_action] - Q[state][action]
            for (states, actions) in episode:
                Q[states][actions] = Q[states][actions] + alpha * delta * E[states][actions]
                E[states][actions] = discount_factor * lambda_factor * E[states][actions]
                
            
            state = next_state; action = next_action;
            if done:
                break
    env.close()
    return Q, er, el


if __name__ == '__main__':
    Q, episode_reward, episode_length = learn(
        env_n='FrozenLake-v0',\
        num_episodes=1000,\
        outdir= "/Users/jacopo/openaigym/project/TD/results/SARSA_Lambda/",\
        replacing_traces=True,\
        folder_options = '',\
        delete_content=True,\
        monitor=False)
