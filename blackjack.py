import random
import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Blackjack-v0')
policy_action = {}

# initialize all state - action pairs
for hand_p in range(2, 32):
    for hand_d in range(1, 11):
        policy_action[((hand_p, hand_d, True), 1)] = 0
        policy_action[((hand_p, hand_d, True), 0)] = 0
        policy_action[((hand_p, hand_d, False), 1)] = 0
        policy_action[((hand_p, hand_d, False), 0)] = 0


def greedy_policy(state, epsilon=0):
    assert (epsilon < 1 and epsilon >= 0), "epsilon must be in a range [0, 1)"
    if random.random() < epsilon:
        return (random.randint(0, 1))
    else:
        return (max_action(state))


def max_action(state):
    if policy_action[(state, 0)] > policy_action[(state, 1)]:
        return (0)
    else:
        return (1)

episode_count = 100000

#parameters found by differential evolution algorithm
alpha = 0.18751748
gamma =  0.02787828
epsilon = 0.00291444

episode_reward = [0]*episode_count

for i in range(episode_count):
    state = env.reset()
    done = False

    #uncomment below for Sarsa version
    # action = greedy_policy(state, epsilon)

    while not done:
        #comment this action for Sarsa version
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
        # action = action2

        state = state2

    episode_reward[i] = reward2



good = episode_reward.count(1.0)*100/episode_count
bad = episode_reward.count(-1)*100/episode_count
draw = episode_reward.count(0.0)*100/episode_count
print("good: {}").format(good)
print("bad: {}").format(bad)
print("draw: {}").format(draw)

episode_reward = np.array(episode_reward)
episode_reward_mat = episode_reward.reshape(episode_count/100,100)
reward100 = episode_reward_mat.sum(1)
# print(reward100)

plt.plot(reward100)
plt.show()

# check weights assigned to state-value pairs:
# for hand_p in range(18, 24):
#     for hand_d in range(1, 11):
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", True), 1): " + str(policy_action[((hand_p, hand_d, True), 1)]))
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", True), 0)]): " + str(policy_action[((hand_p, hand_d, True), 0)]))
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", False), 1)]): " + str(policy_action[((hand_p, hand_d, False), 1)]))
#         print("dla stanu: ((" + str(hand_p) + ", " + str(hand_d) + ", False), 0)]): " + str(policy_action[((hand_p, hand_d, False), 0)]))