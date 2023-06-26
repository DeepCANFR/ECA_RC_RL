import config
from collections import deque
import torch
import copy
import random
from reservoir import Reservoir

class Agent():
    def __init__(self, reservoir: Reservoir, load_path = ''):

        self.create_model(reservoir)

        if load_path:
            self.load(load_path)

        self.update_target_count = 0
        self.replay_memory = deque(maxlen=config.REPLAY_MEMOEY_SIZE)

    def create_model(self, reservoir):
        self.main = torch.nn.Sequential(
            torch.nn.Linear(reservoir.width * reservoir.num_rows_input, config.OUTPUTS),
        ).double()

        self.target = copy.deepcopy(self.main)
        self.optimizer = torch.optim.Adam(self.main.parameters(), lr=config.LEARNING_RATE)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def sync(self):
        self.target.load_state_dict(self.main.state_dict())

    def get_actions(self, state):
        state = torch.as_tensor(state).double()
        with torch.no_grad():
            return self.main(state)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < config.MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, config.MINIBATCH_SIZE)

        for index, (state, action, reward, state_next, done) in enumerate(minibatch):
        
            state = torch.as_tensor(state).double()
            action = torch.as_tensor(action).long()
            reward = torch.as_tensor(reward).double()
            state_next = torch.as_tensor(state_next).double()
            done = torch.as_tensor(done).double()

            with torch.no_grad():
                Q_next = torch.max(self.target(state_next))
                target = reward + (1-done) * config.DISCOUT * Q_next

            Q_pred = self.main(state)[action]
            loss = torch.mean((target - Q_pred)**2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        if terminal_state:
            self.update_target_count += 1

        if self.update_target_count > config.UPDATE_TARGET_EVERY:
            self.sync()
            self.update_target_count = 0

    def save(self, path):
        ''' Saves the agent model '''

        torch.save({
            'model_state_dict': self.main.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ''' Loads the agent model '''

        checkpoint = torch.load(path)
        self.main.load_state_dict(checkpoint['model_state_dict'])
        self.target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.sync()
        



if __name__ == "__main__":
    print('Testing Agent class')

    # agent = Agent('test')
    # agent.model.summary()
    # agent.save('models/test_')

    # agent = Agent('test', 'models/test_.h5')
    # agent.model.summary()

    print(torch.cuda.get_device_name(0))
