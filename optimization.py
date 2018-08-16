from util import mediums
from NiaPy.algorithms.basic import BatAlgorithm, FireflyAlgorithm

import matplotlib.pyplot as plt
import numpy as np
import random
import logging

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

env = mediums(500, 500, 15)

class Brachistochrone(object):
    def __init__(self):
        self.Lower = 0.0
        self.Upper = 1.0

    def function(self):
        def evaluate(D, sol):
            return env.calculateFitness(sol)
        return evaluate

# algo = BatAlgorithm(15, 100, 10000, 0.5, 0.5, 0.1, 0.9, Brachistochrone())
# algo.best
# algo = FireflyAlgorithm(15, 100, 10000, 0.5, 0.5, 0.5, Brachistochrone())
# algo.Fireflies[0]

algo.run()

env.drawSolution()
plt.show()
