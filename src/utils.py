import numpy as np

class Fish:

    def __init__(self, width, height):

        spe = np.random.rand(2)*2 -1
        while np.linalg.norm(spe) == 0:
            spe = np.random.rand(2) *2 -1
        spe = spe/np.linalg.norm(spe)

        self.speed = spe  # The fish's speed
        self.position = np.random.rand(2) * [width, height]  # The fish's position
