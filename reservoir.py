import config
import random
import os
import numpy as np
import pickle
from eca import ECA
from ECAvisualizer import EcaVisualizer
from utils import clamp, xor

ENV_MAX = [0.5, 2, 0.1, 0.5] # observations normalised between

class Reservoir(object):
    def __init__(self, 
        rule: int = 0, 
        render: bool = False,
        load_path: str = '',
    ):
        self.rule = ECA(rule)
        
        if not load_path:
            self.width = config.WIDTH
            self.num_rows_input = config.NUM_ROWS_INPUT
            self.acc_per_obs = config.ACCURASY_PER_OBSERVATION
            self.iterations = config.ITERATIONS
            self.obs_mappings = self.create_obs_mappings(method='all_random')
            # self.obs_mappings = self.create_obs_mappings(method='local_random')
            # self.obs_mappings = self.create_obs_mappings(method='obs_place')
        else:
            self.load(load_path)

        self.render = render
        if render:
            self.visualizer = EcaVisualizer(self, colour=True)

        self.reset()

    def step(self):
        '''
        Applies the rule to the current cells
        '''

        self.rows.append(self.cells)
        self.cells = self.rule.iterate(self.cells)
        self.generation += 1

        if (len(self.rows) > config.HEIGHT and len(self.rows) > self.num_rows_input):
            self.rows.pop(0)

        if self.render:
            self.visualizer.draw()
        
    def update(self, observation, observation_space = None):
        '''
        takes inn a observation from an env and updates the reservoir

        compute accuracy for each observation
        closer to middle of lower and higher means as many 0's as 1's
        closer to lower limit is more 0's
        closer to higher limit is more 1's
        '''

        bit_obs = []

        for i in range(len(observation)):
            obs_norm = observation[i] / ENV_MAX[i]
            for i in np.arange(-1, 1+2/self.acc_per_obs, 2/(self.acc_per_obs-1)):
                if obs_norm < i:
                    bit_obs.append(0)
                else:
                    bit_obs.append(1)

        # map observations to cells based on obs_mappings
        bits_per_res = self.acc_per_obs * len(observation)
        for res_index in range(config.NUM):
            c = res_index * bits_per_res

            for i in range(bits_per_res):
                index = self.obs_mappings[i + c]

                # PLACE
                self.cells[index] = bit_obs[i]

                # XOR
                # self.cells[index] = xor(self.cells[index], bit_obs[i])

        # Aplly the rule if iterations are greater than 0
        if self.iterations > 0:
            for i in range(self.iterations):
                self.step()
        else:
            self.generation += 1
            self.rows.append(self.cells.copy())

            if (len(self.rows) > config.HEIGHT and len(self.rows) > self.num_rows_input):
                self.rows.pop(0)
            
            if self.render:
                self.visualizer.draw()



    def read(self):
        '''
        returns the current reservoir state
        and the last num_rows_input number of rows

        formatted: [[state-4], [state-3], [state-2], [state-1], [state]] as a 1D array

        if number of rows < num_rows_input, previous states will consist of only 0's
        '''

        state = self.rows[-self.num_rows_input:]

        if (len(state) < self.num_rows_input):
            for _ in range(self.num_rows_input - len(state)):
                state.insert(0, [0 for _ in range(self.width)])

        state = np.array(state, dtype='uint8').reshape(-1, self.num_rows_input*self.width)[0]
        return state

    def save_image(self):
        if self.render:
            self.visualizer.save_image()

    def create_obs_mappings(self, method):
        obs_mappings = []
        num = config.NUM

        if method == 'all_random':
            for i in range(num):
                start = i * int(self.width / num)
                end = (i + 1) * int(self.width / num)
                mapping = random.sample(range(start, end), self.acc_per_obs * 4)
                
                for m in mapping:
                    obs_mappings.append(m)

        elif method == 'local_random':
            obs_width = int(config.WIDTH / 4)
            for i in range(4):
                for _ in range(self.acc_per_obs):
                    index = random.randint(obs_width * i, obs_width * i + (obs_width - 1))
                    while index in obs_mappings:
                        index = random.randint(obs_width * i, obs_width * i + (obs_width - 1))
                    obs_mappings.append(index)

        elif method == 'obs_place':
            obs_width = int(config.WIDTH / 4)
            for i in range(4):
                obs_center = obs_width*i + obs_width / 2
                for index in range(int(obs_center - self.acc_per_obs / 2), int(obs_center + self.acc_per_obs / 2)):
                    obs_mappings.append(index)


        return obs_mappings

    def reset(self):
        self.generation = 0
        self.cells = [0 for _ in range(self.width)]
        self.rows = [] # for storing previous generations

        if self.render:
            self.visualizer.reset()

    def save(self, path):
        '''
        Saves the reservoir's paramaters to a file:
         - rule
         - width
         - num_rows_input
         - acc_per_obs
         - iterations
         - obs_mappings
        '''
        res_config = {
            'rule': int(self.rule),
            'width': self.width,
            'num_rows_input': self.num_rows_input,
            'acc_per_obs': self.acc_per_obs,
            'iterations': self.iterations,
            'obs_mappings': self.obs_mappings,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(res_config, f)

    def load(self, path):
        ''' Loads a reservoir from configuration file '''
        with open(path, 'rb') as f:
            res_config = pickle.load(f)

        self.rule = ECA(res_config['rule'])
        self.width = res_config['width']
        self.num_rows_input = res_config['num_rows_input']
        self.acc_per_obs = res_config['acc_per_obs']
        self.iterations = res_config['iterations']
        self.obs_mappings = res_config['obs_mappings']



if __name__ == '__main__':
    print('------ Testing reservoir ------')
    print('Save and Looad')

    reservoir = Reservoir(57)
    print(len(reservoir.obs_mappings), reservoir.obs_mappings)


    # print(f'Befor load: rule {int(reservoir.rule)}, mappings {reservoir.obs_mappings}')
    # reservoir.save('models/test')

    # reservoir = Reservoir(load_path='models/test')
    # print(f'After load: rule {int(reservoir.rule)}, mappings {reservoir.obs_mappings}')

    


