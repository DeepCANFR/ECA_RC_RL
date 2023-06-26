import gymnasium as gym
import os
import numpy as np

from reservoir import Reservoir
from agent import Agent



RUN_NAME = 'run_1'
NUM_TEST = 100
MAX_STEPS = 500


run_path = os.path.join('models', RUN_NAME)
data_paths = []

for f in os.listdir(run_path):
    model = os.path.join(run_path, f)
    for df in os.listdir(model):
        if df == 'done.pkl':
            data_paths.append(model)

data_paths.sort(key = lambda x: int(x.split('rule')[1].split('_')[0]))

results = {}
env = gym.make("CartPole-v1")
for path in data_paths:
    print(path)
    
    reservoir = Reservoir(render=False, load_path=os.path.join(path, 'reservoir.pkl'))
    agent = Agent(reservoir=reservoir, load_path=os.path.join(path, 'done.pkl'))

    ep_rewards = []
    for episode in range(NUM_TEST):
        episode_reward = 0

        observation, info = env.reset()
        reservoir.reset()
        reservoir.update(observation, env.observation_space)
        state = reservoir.read()

        terminated = False
        while not terminated:

            action = np.argmax(agent.get_actions(state).numpy())

            observation, reward, terminated, _, _ = env.step(action)
            reservoir.update(observation, env.observation_space)
            state = reservoir.read()

            episode_reward += reward

            if episode_reward >= MAX_STEPS:
                terminated = True

        ep_rewards.append(episode_reward)

    average_reward = sum(ep_rewards)/len(ep_rewards)
    min_reward = min(ep_rewards)
    max_reward = max(ep_rewards)

    rule = int(path.split('rule')[1].split('_')[0])
    results[rule] = average_reward
    print(f"rule {rule}", max_reward, average_reward, min_reward)

env.close()

results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

with open(f'runs/test_{RUN_NAME}.txt', 'a') as file:
    for res in results:
        test_result = f'{res:3d}, {results[res]:.2f}'
        file.writelines(test_result + '\n')