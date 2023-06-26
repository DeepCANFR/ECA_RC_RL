import os
if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('runs'):
    os.makedirs('runs')
if not os.path.isdir('screenshots'):
    os.makedirs('screenshots')

import gymnasium as gym
import config
import random
import numpy as np
import time
import pickle

import utils
from reservoir import Reservoir
from agent import Agent


EPSILON_DECAY = 0.9975 # 0.9997 for 20_000, 0.9995 for 10_000, 0.999 for 5_000, 0.9975 for 1_000
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 100  # in episodes, save checkpoint, and print stats

c = 0
e = 1
while e > MIN_EPSILON:
    e *= EPSILON_DECAY
    c += 1

print(f'Epsilon will reach {MIN_EPSILON} after {c} epochs')



env = gym.make("CartPole-v1")

FILE_NAME = f'runs/{config.RUN_NAME}.txt'
run_start_time = int(time.time())

for rule in utils.unique_eca_rules():

    epsilon = 1
    MODEL_PATH = f'models/{config.RUN_NAME}/rule{rule}_num{config.NUM}_w{config.WIDTH}_iter{config.ITERATIONS}_input{config.NUM_ROWS_INPUT}_acc{config.ACCURASY_PER_OBSERVATION}_{run_start_time}'
    reservoir = Reservoir(rule, render=False)
    reservoir.save(f'{MODEL_PATH}/reservoir.pkl')
    agent = Agent(reservoir=reservoir)
    ep_rewards = []


    s = time.time()
    t_steps = 0
    for episode in range(1, config.EPISODES):
        episode_reward = 0
        step = 1

        observation, info = env.reset()
        reservoir.reset()
        reservoir.update(observation, env.observation_space)
        state = reservoir.read()

        early_stop = False
        terminated = False
        while not terminated:

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.get_actions(state).numpy())

            observation, reward, done, truncated, info = env.step(action)
            if done: # when terminated, reward is still 1
                reward = 0
            
            reservoir.update(observation, env.observation_space)
            new_state = reservoir.read()

            episode_reward += reward

            agent.update_replay_memory((state, action, reward, new_state, done))
            agent.train(done, step)

            state = new_state

            if step >= 200 or done:
                terminated = True

            step += 1
            t_steps += 1



        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

            elaps_time_format = time.strftime('%H:%M:%S', time.gmtime(time.time()-s))
            print(elaps_time_format, t_steps, episode, episode_reward, max_reward, average_reward, min_reward)

            agent.save(f'{MODEL_PATH}/{episode}episode_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.pkl')

            # if episode > 500 and average_reward > 100:
            #     run_string = f'Total steps {t_steps:6d}, Rule {rule:3d} stopped after {episode:4d} episodes, finishing EARLY due to GOOD average {average_reward:.2f}, taking {elaps_time_format}'
            #     utils.write_run_file(FILE_NAME, run_string)
            #     early_stop = True
            #     break

            data = {'rule': rule, 'rewards': ep_rewards}
            with open(f'{MODEL_PATH}/data.pkl', 'wb') as f:
                pickle.dump(data, f)

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    if not early_stop:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        elaps_time_format = time.strftime('%H:%M:%S', time.gmtime(time.time()-s))
        run_string = f'Total steps {t_steps:6d}, Rule {rule:3d} stopped after {episode:4d} episodes with an average of average {average_reward:.2f}, taking {elaps_time_format}'
        utils.write_run_file(FILE_NAME, run_string)
    
    agent.save(f'{MODEL_PATH}/done.pkl')

    data = {'rule': rule, 'rewards': ep_rewards}
    with open(f'{MODEL_PATH}/data.pkl', 'wb') as f:
        pickle.dump(data, f)

    print('-----------------------')
    print('-----------------------')
    elaps_time = time.time()-s
    print(f'Rule {rule:3d}, took {elaps_time:.2f}s, total steps {t_steps}, averaging {t_steps/elaps_time:.2f} steps per second')
    print('-----------------------')
    print('-----------------------')

env.close()