import random
import gym
# import numpy as np
# import matplotlib.pyplot as plt

from timeit import default_timer as timer

# performance measure
perf = [0]*10

for tt in range(10):
   start = timer()
env = gym.make('Blackjack-v0')
policy_action = {}

# initialize all state - action pairs
# it was checked if it's faster to initialize on the flow or at the beginning
# and initialization at the beginning was performing slightly faster
# performance averaged over 10 runs for 10 000 and 100 000 episodes
# mean time for creating on the flow 0.846010236708 8.1050000278
# mean time for intitializing at the beginning 0.830068203385, 7.71337580026

for hand_p in range(2, 32):
    for hand_d in range(1, 11):
        policy_action[((hand_p, hand_d, True), 1)] = 0
        policy_action[((hand_p, hand_d, True), 0)] = 0
        policy_action[((hand_p, hand_d, False), 1)] = 0
        policy_action[((hand_p, hand_d, False), 0)] = 0


def greedy_policy(state, epsilon=0):
    assert (epsilon < 1 and epsilon >= 0), "epsilon must be in a range [0, 1)"
    if random.random() < epsilon:
        # if policy_action.get((state, 0)) == None:
        #     policy_action[(state, 0)] = 0
        #     policy_action[(state, 1)] = 0
        return (random.randint(0, 1))
    else:
        return (max_action(state))


def max_action(state):
    ###print("dla stanu:" + str(state))
    # if policy_action.get((state, 0)) == None:
    #     policy_action[(state, 0)] = 0
    #     policy_action[(state, 1)] = 0
    #     # or maybe choose action randomly?
    #     return (1)
    # else:
    if policy_action[(state, 0)] > policy_action[(state, 1)]:
        return (0)
    else:
        return (1)


episode_count = 100000
alpha = 0.05
gamma = 0.9
epsilon = 0.1
episode_reward = [0] * episode_count

for i in range(episode_count):
    ###print(i)
    state = env.reset()
    done = False

    # uncomment below for Sarsa version
    # action = greedy_policy(state, epsilon)

    while not done:
        # comment this action for Sarsa version
        action = greedy_policy(state, epsilon)
        state2, reward2, done, _ = env.step(action)
        present_value = policy_action[(state, action)]

        Q-learning
        action2 = max_action(state2)
        policy_action[(state, action)] = \
            present_value + \
            alpha * (reward2 + gamma * policy_action[(state2, action2)] - present_value)

        ###print("dla stanu:" + str((state, action)) + " nagroda: " + str(policy_action[(state, action)]))

        # Sarsa
        # action2 = greedy_policy(state2, epsilon)
        # policy_action[(state, action)] = \
        #     present_value + \
        #     alpha*(reward2 + gamma * policy_action[(state2, action2)] - present_value)
        # action = action2

        ###print("dla stanu:" + str((state, action)) + " nagroda: " + str(policy_action[(state, action)]))
        state = state2

    episode_reward[i] = reward2

    perf[tt] = timer() - start

print("Elapsed time: " + str(sum(perf) / float(len(perf))))

# good = episode_reward.count(1.0) * 100 / episode_count
# bad = episode_reward.count(-1) * 100 / episode_count
# draw = episode_reward.count(0.0) * 100 / episode_count
# print("good: {}").format(good)
# print("bad: {}").format(bad)
# print("draw: {}").format(draw)
#
# episode_reward = np.array(episode_reward)
# episode_reward_mat = episode_reward.reshape(episode_count / 100, 100)
# reward100 = episode_reward_mat.sum(1)
# print(reward100)
#
# plt.plot(reward100)
# plt.show()
#
# for hand_p in range(18, 24):
#     for hand_d in range(1, 11):
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", True), 1): " + str(
#             policy_action[((hand_p, hand_d, True), 1)]))
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", True), 0)]): " + str(
#             policy_action[((hand_p, hand_d, True), 0)]))
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", False), 1)]): " + str(
#             policy_action[((hand_p, hand_d, False), 1)]))
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", False), 0)]): " + str(
#             policy_action[((hand_p, hand_d, False), 0)]))