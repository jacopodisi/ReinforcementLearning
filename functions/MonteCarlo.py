from __future__ import print_function
import numpy as np
import pprint
import sys
from collections import defaultdict

import gym
from gym import wrappers
from gym import spaces
from gym import envs

#behavior policy for on policy control
def epsilon_greedy_policy(s, nA, q, epsilon):
    A = np.ones(nA, dtype = float) * epsilon / nA
    best_action = np.argmax(q[s])
    A[best_action] += 1.0 - epsilon
    return A

#behavior policy for off policy control
def random_policy(s, nA):
    A = np.ones(nA, dtype=float) / nA
    return A

#target policy for off policy control
def greedy_policy(s, nA, q):
    A = np.zeros(nA, dtype = float)
    best_action = np.argmax(q[s])
    A[best_action] = 1.0
    return A

#learning funtion for the montecarlo on policy control
# given an episode = [(state, action, reward)] update the Q function
def on_policy_control(env, num_episodes, discount_factor = 1, epsilon = 0.05, returns_sum = defaultdict(float), returns_count = defaultdict(float), Q = defaultdict(lambda: np.zeros(env.action_space.n))):
    for i_episode in range(1,num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()

        while True:
            probs = epsilon_greedy_policy(state, env.action_space.n, Q, epsilon)
            action = np.random.choice(np.arange(len(probs)), p = probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        sa_in_episode = set((x[0], x[1]) for x in episode)
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
            G = sum([x[2] for x in episode[first_occurence_idx:]])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    return Q, returns_sum, returns_count

#learning funtion for the montecarlo off policy control
# given an episode = [(state, action, reward)] update the Q function
def off_policy_control(env, num_episodes, discount_factor = 1, Q = defaultdict(lambda: np.zeros(env.action_space.n)), C = defaultdict(lambda: np.zeros(env.action_space.n))):

    for i_episode in range(1,num_episodes + 1):
        print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        sys.stdout.flush()

        G = 0.0
        W = 1.0

        episode = []
        state = env.reset()

        while True:
            probs = random_policy(state, env.action_space.n)
            action = np.random.choice(np.arange(len(probs)), p = probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            if action != np.argmax(greedy_policy(state)):
                break
            W = W * 1./greedy_policy(state)[action]
    return Q, C


# In[16]:

if __name__ == '__main__':
    num_episodes = 50000
    env_name = 'FrozenLake-v0'
    env = envs.make(env_name)
    outdir = "/Users/jacopo/openaigym/project/MC/results/" + env_name
    env = wrappers.Monitor(env, outdir, force=True)
    env.seed(0)
    Q, ret_sum, ret_count = on_policy_control(env.unwrapped, 50000)

    
    state = env.unwrapped.reset()
    env.unwrapped.render()

    while True:
        probs = epsilon_greedy_policy(state, env.unwrapped.action_space.n, Q, 0.1)
        action = np.random.choice(np.arange(len(probs)), p = probs)
        next_state, reward, done, _ = env.unwrapped.step(action)
        env.unwrapped.render()
        if done:
            break
        state = next_state

    env.close()
