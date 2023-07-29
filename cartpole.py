import gymnasium as gym
import numpy
import time

env = gym.make("CartPole-v1")

state_space = 4 # Each state is defined by 4 different values
action_space = 2 # I can only make two different actions, left or right

def QTable(state_space, action_space, bin_size=30):
    bins = [numpy.linspace(-4.8,4.8,bin_size),
            numpy.linspace(-4,4,bin_size),
            numpy.linspace(-0.418,0.418,bin_size),
            numpy.linspace(-4,4,bin_size)]
    
    q_table = numpy.random.uniform(low=-1, high=1, size=([bin_size] * state_space + [action_space]))

    return q_table, bins

def Discrete(state, bins):
    index = []

    for i in range(len(state)):
        index.append(numpy.digitize(state[i], bins[i]) - 1)

    return tuple(index)


def Q_learning(q_table, bins, episodes = 5000, gamma = 0.95, lr = 0.1, timestep = 100, epsilon = 0.2):
    rewards = 0
    steps = 0

    for episode in range(1,episodes+1):
        steps += 1 
        current_state = Discrete(env.reset()[0], bins)

        score = 0
        done = False

        while not done: 
            if episode % timestep == 0:
                pass #env.render()

            if numpy.random.random() < epsilon: # Randomly choose an action
                action = env.action_space.sample()
            else:
                action = numpy.argmax(q_table[current_state]) # Choose the best known action

            observation, reward, done, _, _ = env.step(action) # Make an action and get a reward
            next_state = Discrete(observation, bins)
            score += reward

            if not done:
                max_future_q = numpy.max(q_table[next_state]) # Take the max future Q value
                current_q = q_table[current_state+(action,)] # Take the current Q value
                new_q = (1-lr)*current_q + lr*(reward + gamma*max_future_q) # Compute the new Q value with the Q function
                q_table[current_state+(action,)] = new_q # Update it in the Q Table

            current_state = next_state
        
        else:
            rewards += score # If done, add the reward to the total rewards
            if score > 195 and steps >= 100:
                print(f"Episode: {episode} -> Solved")
            if episode % timestep == 0:
                print(f"Episode: {episode}")

q_table, bins = QTable(state_space, action_space, bin_size=20)
print(q_table.shape)
Q_learning(q_table, bins, episodes=5000)