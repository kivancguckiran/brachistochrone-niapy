import os, sys

sys.path.insert(0, os.path.abspath("../NiaPy"))

from util import mediums
from NiaPy.algorithms.basic import BatAlgorithm, DifferentialEvolutionAlgorithm, ArtificialBeeColonyAlgorithm, GeneticAlgorithm, CamelAlgorithm

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
# algo = ArtificialBeeColonyAlgorithm(D=15, NP=100, nFES=10000, Limit=100, benchmark=Brachistochrone())
# algo = GeneticAlgorithm(D=15, NP=100, nFES=10000, Ts=4, Mr=0.05, CR=0.9, benchmark=Brachistochrone())
# algo = CamelAlgorithm(D=15, NP=100, nFES=10000, omega=0.25, mu=0.5, alpha=0.5, S_init=10, E_init=10, T_min=-1, T_max=10, benchmark=Brachistochrone())

solution, score = algo.run()
env.drawSolution(solution)
plt.show()