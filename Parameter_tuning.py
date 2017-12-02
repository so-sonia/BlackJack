from scipy.optimize import differential_evolution

import random
import gym

env = gym.make('Blackjack-v0')
policy_action = {}

def create_policy_arary():
    policy_action = {}
    for hand_p in range(2, 32):
        for hand_d in range(1, 11):
            policy_action[((hand_p, hand_d, True), 1)] = 0
            policy_action[((hand_p, hand_d, True), 0)] = 0
            policy_action[((hand_p, hand_d, False), 1)] = 0
            policy_action[((hand_p, hand_d, False), 0)] = 0
    return(policy_action)

def greedy_policy(state, epsilon=0):
    if random.random() < epsilon:
        return (random.randint(0, 1))
    else:
        return (max_action(state))


def max_action(state):
    if policy_action[(state, 0)] > policy_action[(state, 1)]:
        return (0)
    else:
        return (1)

def blackjack(x):
    alpha = x[0]
    gamma = x[1]
    epsilon = x[2]
    episode_count = 10000
    episode_reward = [0]*episode_count

    #needs to be global so max_action will have access to it
    global policy_action
    policy_action = create_policy_arary()

    for i in range(episode_count):
        state = env.reset()
        done = False
        while not done:
            action = greedy_policy(state, epsilon)
            state2, reward2, done, _ = env.step(action)
            present_value = policy_action[(state, action)]

            # Q-learning
            action2 = max_action(state2)
            policy_action[(state, action)] = \
                present_value + \
                alpha * (reward2 + gamma * policy_action[(state2, action2)] - present_value)

            # Sarsa
            # action2 = greedy_policy(state2, epsilon)
            # policy_action[(state, action)] = \
            #     present_value + \
            #     alpha*(reward2 + gamma * policy_action[(state2, action2)] - present_value)

            state = state2

        episode_reward[i] = reward2

    good = episode_reward.count(1.0) * 100 / episode_count
    return(-good)


bounds = [(0, 1), (0, 1), (0, 1)]
result = differential_evolution(blackjack, bounds, maxiter = 200)
print(result.x, result.fun)

