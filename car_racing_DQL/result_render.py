import gymnasium as gym
from agent import DQNAgent
import time

if __name__ == "__main__":
    env = gym.make('CarRacing-v2', continuous=False, render_mode="human")

    agent = DQNAgent(gamma=0.99, epsilon=0.1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=5, mem_size=50000, eps_min=0.0,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='CarRacing-v2')
    
    agent.load_models()
    
    score = 0
    observation, info = env.reset()

    for i in range(1000):
        action = agent.choose_action(observation)
        print(action)
        observation_, reward, terminated, truncated, info = env.step(action)
        score += reward
        time.sleep(0.01)
    
    print(score)
