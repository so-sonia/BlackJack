import random
import gym
env = gym.make('Blackjack-v0')

policy_action = {}

#initialize all state - action pairs
# for i in range(1, 32):
#     for j in range(1, 11):
#         policy_action[(i, j, True), 1] = 0
#         policy_action[(i, j, True), 0] = 0
#         policy_action[(i, j, False), 1] = 0
#         policy_action[(i, j, False), 0] = 0

def greedy_policy(state, epsilon = 0):
    assert (epsilon<1 and epsilon>=0), "epsilon must be in a range [0, 1)"
    if random.random()<epsilon:
        if policy_action.get((state, 0)) == None:
            policy_action[(state, 0)] = 0
            policy_action[(state, 1)] = 0
        return(random.randint(0, 1))
    else:
        return(better_action(state))

def better_action(state):
    ###print("dla stanu:" + str(state))
    if policy_action.get((state, 0))==None :
        policy_action[(state, 0)] = 0
        policy_action[(state, 1)] = 0
        #or maybe choose action randomly?
        return(1)
    else:
        if policy_action[(state, 0)]> policy_action[(state, 1)]:
            return(0)
        else:
            return(1)

episode_count = 100
alpha = 0.5
gamma = 0.5
epsilon = 0.1

for i in range(episode_count):
    ###print(i)
    state = env.reset()

    #1.5 reward if blackjack?

    done = False
    while not done:
        action = greedy_policy(state, epsilon)
        state2, reward2, done, _ = env.step(action)
        present_value = policy_action[(state, action)]

        # Q-learning
        action2 = better_action(state2)
        policy_action[(state, action)] = \
            present_value + \
            alpha*(reward2 + gamma * policy_action[(state2, action2)] - present_value)

        ###print("dla stanu:" + str((state, action)) + " nagroda: " + str(policy_action[(state, action)]))

        # Sarsa
        # action2 = greedy_policy(state2, epsilon)
        # policy_action[(state, action)] = \
        #     present_value + \
        #     alpha*(reward2 + gamma * policy_action[(state2, action2)] - present_value)

        state = state2

    policy_action[(state, action)] = reward2

    ###print("dla stanu:" + str((state, action)) + " nagroda: " + str(policy_action[(state, action)]))


