RUN_NAME = 'run_1' # groups all models under a folder with this name

''' RESERVOIR ECA '''
WIDTH = 128 # number of ECA cells
ITERATIONS = 5 # how many times to apply the rule before updating
ACCURASY_PER_OBSERVATION = 16 # how many bit accuracy for each observation
NUM = 1 # subdevisions of reservoir with own mappings
HEIGHT = 100 # for rendering
CELLSIZE = 5 # for rendering


''' AGENT '''
NUM_ROWS_INPUT = 5
OUTPUTS = 2


''' TRAINING '''
EPISODES = 1000
LEARNING_RATE = 0.001
REPLAY_MEMOEY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUT = 0.95
UPDATE_TARGET_EVERY = 10



