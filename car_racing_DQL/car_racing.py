import numpy as np
from agent import DQNAgent
import gymnasium as gym
import time

if __name__ == '__main__':
    env = gym.make("CarRacing-v2", continuous=False)

    best_score = -np.inf
    load_checkpoint = True
    n_games = 1000
    force_train = True

    agent = DQNAgent(gamma=0.99, epsilon=0.9, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=5, mem_size=50000, eps_min=0.01,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='CarRacing-v2')
    
    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        start = time.time()
        done = False
        observation, info = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated
            score += reward

            if not load_checkpoint or force_train:
                agent.store_transition(observation, action,
                                            reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        time_taken = time.time() - start
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps, 'time %.2f' % time_taken)

        if avg_score > best_score:
            if not load_checkpoint or force_train:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
