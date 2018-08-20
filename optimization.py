import os, sys

sys.path.insert(0, os.path.abspath("../NiaPy"))

from util import mediums
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm, DifferentialEvolutionAlgorithm, EvolutionStrategy1p1, EvolutionStrategyMp1, BareBonesFireworksAlgorithm
from NiaPy.algorithms.modified import HybridBatAlgorithm, DynNPSelfAdaptiveDifferentialEvolutionAlgorithm, SelfAdaptiveDifferentialEvolutionAlgorithm
from NiaPy.algorithms.other MultipleTrajectorySearch

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

# algo = ArtificialBeeColonyAlgorithm(D=D, nGEN=nGEN, benchmark=Brachistochrone())
# algo = DifferentialEvolutionAlgorithm(D=D, nGEN=nGEN, F=0.5, CR=0.9, benchmark=Brachistochrone())
# algo = EvolutionStrategy1p1(D=D, nGEN=nGEN, benchmark=Brachistochrone())
# algo = EvolutionStrategyMp1(D=D, nGEN=nGEN, benchmark=Brachistochrone())
# algo = BareBonesFireworksAlgorithm(D=D, nGEN=nGEN, benchmark=Brachistochrone())
# algo = HybridBatAlgorithm(D=D, nGEN=nGEN, NP=40, A=0.9, r=0.1, F=0.001, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark=Brachistochrone())
# algo = DynNPSelfAdaptiveDifferentialEvolutionAlgorithm(NP=50, D=D, nGEN=nGEN, F=0.5, F_l=-1, F_u=2.0, Tao1=0.1, CR=0.45, Tao2=0.25, benchmark=Brachistochrone())
# algo = SelfAdaptiveDifferentialEvolutionAlgorithm(NP=50, D=D, nGEN=nGEN, F=0.5, F_l=-1, F_u=2.0, Tao1=0.1, CR=0.45, Tao2=0.25, benchmark=Brachistochrone())
# algo = MultipleTrajectorySearch(D=D, nGEN=nGEN, benchmark=Brachistochrone())

solution, score = algo.run()
env.drawSolution(solution)
plt.show()
