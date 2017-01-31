from __future__ import print_function
import numpy as np
import sys

import gym
from gym import wrappers


def policy_eval(env, policy, discount_factor, theta=0.000001):
    V = np.zeros(env.nS)
    c = 0
    while True:
        c += 1
        delta = 0
        for s in range(env.nS):
            v = 0
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += policy[s][a] * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v

        if delta < theta:
            break

    return V


def policy_iteration(env, discount_factor=1):
    policy = np.ones([env.nS, env.nA]) / env.nA
    c = 0
    while True:
        c += 1
        li = 0
        V = policy_eval(env, policy, discount_factor)
        policy_stable = True
        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            q_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    q_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(q_values)
            if chosen_a != best_a:
                policy_stable = False
                li += 1
            p = np.eye(env.nA)
            policy[s] = p[best_a]
        print("\rPolicy Improvement iteration {}. Number of instable choice: {}".format(c, li), end="") 
        sys.stdout.flush()
        if policy_stable:
            return policy, V


if __name__ == '__main__':
    env = gym.make('Taxi-v1')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/Users/jacopo/openaigym/project/DP/results'
    env = wrappers.Monitor(env, outdir, force=True)
    env.seed(0)
    policy, V = policy_iteration(env.unwrapped, discount_factor=0.9)

    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = np.argmax(policy[ob])
            env.render()
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
    #gym.upload(outdir, api_key='sk_v7ktbUr7SzC68vXvnrwLLQ')
