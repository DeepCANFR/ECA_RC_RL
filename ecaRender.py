from reservoir import Reservoir
from ECAvisualizer import EcaVisualizer
import config
import random
from utils import unique_eca_rules
 

if __name__ == '__main__':


    cells = [round(random.random()) for _ in range(config.WIDTH)]
    for rule in unique_eca_rules():

        reservoir = Reservoir(rule, True)
        
        # reservoir.cells[int(len(reservoir.cells)/2)] = 1
        reservoir.cells = cells

        EcaVisualizer(reservoir)

        for i in range(config.HEIGHT):
            reservoir.step()
        reservoir.save_image()
