import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from reservoir import Reservoir
from agent import Agent


SHOW_RESERVOIR = True
AGGREGATE_STATS_EVERY = 10


''' Change path to model, and model name '''
BASE_PATH = 'models/run_6/rule38_w64_iter15_input15_acc16_1683382220'
MODEL_NAME = 'done.pkl'


reservoir = Reservoir(render=SHOW_RESERVOIR, load_path=f'{BASE_PATH}/reservoir.pkl')
agent = Agent(reservoir=reservoir, load_path=f'{BASE_PATH}/{MODEL_NAME}')
print(agent.main)


if SHOW_RESERVOIR:
    env = gym.make("CartPole-v1")
else:
    env = gym.make("CartPole-v1", render_mode="human")


ep_rewards = []
t_steps = 0
for episode in range(20):
    episode_reward = 0

    observation, info = env.reset()
    reservoir.reset()
    reservoir.update(observation, env.observation_space)
    state = reservoir.read()

    terminated = False
    ac0 = []
    ac1 = []
    while not terminated:

        qs = agent.get_actions(state)
        ac0.append(qs[0])
        ac1.append(qs[1])

        action = np.argmax(qs.numpy())

        observation, reward, terminated, truncated, info = env.step(action)
        reservoir.update(observation, env.observation_space)
        state = reservoir.read()

        episode_reward += reward
        t_steps += 1

    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        print(episode, episode_reward, max_reward, average_reward, min_reward)
    
    print('-------------------------', episode_reward)
    
    plt.title('Q-value over steps')
    plt.xlabel('Step')
    plt.ylabel('Q-value')
    plt.plot(ac0, label='Action 0')
    plt.plot(ac1, label='Action 1')
    plt.legend()
    plt.show()


env.close()