import os, sys

sys.path.insert(0, os.path.abspath("../NiaPy"))

from util import mediums
from NiaPy.algorithms.basic import BatAlgorithm, DifferentialEvolutionAlgorithm
from NiaPy.algorithms.other import MultipleTrajectorySearch

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

# algo = BatAlgorithm(D=15, NP=100, nFES=10000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark=Brachistochrone())
# algo = DifferentialEvolutionAlgorithm(D=15, NP=100, nFES=10000, F=0.5, CR=0.9, benchmark=Brachistochrone())
# algo = MultipleTrajectorySearch(D=15, NP=100, nFES=10000, benchmark=Brachistochrone())

solution, score = algo.run()
env.drawSolution(solution)
plt.show()
