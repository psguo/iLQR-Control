import numpy as np

class Control():
    def __init__(self):
        self.u = np.zeros((2,1))


    def get_distance(self, arm):
        pass

    def control(self):
        raise NotImplementedError