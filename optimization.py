import os, sys

niaPath = os.path.abspath("../NiaPy")
sys.path.insert(0, niaPath)

import NiaPy

from util import mediums

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

algorithms = ['DifferentialEvolutionAlgorithm']
benchmarks = [Brachistochrone()]

results = NiaPy.Runner(15, 100, 10000, 1, algorithms, benchmarks).run()

solution, time = results['DifferentialEvolutionAlgorithm']['Brachistochrone'][0]
error = np.sum(env.calculateError(solution))

print(error)

env.drawSolution(solution)
plt.show()
