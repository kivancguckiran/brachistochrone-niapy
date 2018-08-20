import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("../NiaPy"))

class Brachistochrone(object):
    def __init__(self):
        self.Lower = 0.0
        self.Upper = 1.0

    def function(self):
        def evaluate(D, sol):
            return env.calculateFitness(sol)
        return evaluate


env = mediums(500, 500, 15)

D = 15
nGEN = 1000

epoch = 10

def optimize(optimizer):
  solutions = []
  times = []
  errors = []
  
  for i in range(epoch):
    startTime = time.time()
    solution, score = optimizer.run()
    endTime = time.time()

    solutions.append(solution)
    errors.append(env.calculateError(solution))
    times.append(endTime - startTime)

  bestSolutionIdx = np.argmin(errors)
  bestSolution = solutions[bestSolutionIdx]
    
  return bestSolution, np.average(errors), np.average(times)

abc_solution, abc_errors, abc_times = optimize(NiaPy.algorithms.basic.ArtificialBeeColonyAlgorithm(D=D, nGEN=nGEN, benchmark=Brachistochrone()))
de_solution, de_errors, de_times = optimize(NiaPy.algorithms.basic.DifferentialEvolutionAlgorithm(D=D, nGEN=nGEN, F=0.5, CR=0.9, benchmark=Brachistochrone()))
es1_solution, es1_errors, es1_times = optimize(NiaPy.algorithms.basic.EvolutionStrategy1p1(D=D, nGEN=nGEN, benchmark=Brachistochrone()))
esm_solution, esm_errors, esm_times = optimize(NiaPy.algorithms.basic.EvolutionStrategyMp1(D=D, nGEN=nGEN, benchmark=Brachistochrone()))
bbfa_solution, bbfa_errors, bbfa_times = optimize(NiaPy.algorithms.basic.BareBonesFireworksAlgorithm(D=D, nGEN=nGEN, benchmark=Brachistochrone()))
hba_solution, hba_errors, hba_times = optimize(NiaPy.algorithms.modified.HybridBatAlgorithm(D=D, nGEN=nGEN, NP=40, A=0.9, r=0.1, F=0.001, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark=Brachistochrone()))
dpsade_solution, dpsade_errors, dpsade_times = optimize(NiaPy.algorithms.modified.DynNPSelfAdaptiveDifferentialEvolutionAlgorithm(NP=50, D=D, nGEN=nGEN, F=0.5, F_l=-1, F_u=2.0, Tao1=0.1, CR=0.45, Tao2=0.25, benchmark=Brachistochrone()))
sade_solution, sade_errors, sade_times = optimize(NiaPy.algorithms.modified.SelfAdaptiveDifferentialEvolutionAlgorithm(NP=50, D=D, nGEN=nGEN, F=0.5, F_l=-1, F_u=2.0, Tao1=0.1, CR=0.45, Tao2=0.25, benchmark=Brachistochrone()))

print('\n')
print('Type\tError\tTime')
print('ABC\t', abc_errors, '\t', abc_times)
print('DE\t', de_errors, '\t', de_times)
print('ES1\t', es1_errors, '\t', es1_times)
print('ESM\t', esm_errors, '\t', esm_times)
print('BBFA\t', bbfa_errors, '\t', bbfa_times)
print('HBA\t', hba_errors, '\t', hba_times)
print('DPSADE\t', dpsade_errors, '\t', dpsade_times)
print('SADE\t', sade_errors, '\t', sade_times)

