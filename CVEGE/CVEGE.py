import os
from opfunu.cec_based import cec2020
import numpy as np
from scipy.stats import levy

PopSize = 10
Dim = 10
LB = -100
UB = 100
trialRuns = 30
maxFEs = Dim * 1000
GC = 6
GR = 1

Population = np.zeros((PopSize, Dim))
Population_fitness = np.zeros(PopSize)
current_lifespan = 0
curFEs = 0
Fun_num = 1
seedNum = 6
MS = 2


def CheckIndi(Indi):
    range_width = UB - LB
    for i in range(Dim):
        if Indi[i] > UB:
            n = int((Indi[i] - UB) / range_width)
            mirrorRange = (Indi[i] - UB) - (n * range_width)
            Indi[i] = UB - mirrorRange
        elif Indi[i] < LB:
            n = int((LB - Indi[i]) / range_width)
            mirrorRange = (LB - Indi[i]) - (n * range_width)
            Indi[i] = LB + mirrorRange
        else:
            pass


def Initialization(func):
    global Population, Population_fitness, current_lifespan, curFEs
    for i in range(PopSize):
        for j in range(Dim):
            Population[i][j] = np.random.uniform(LB, UB)
        Population_fitness[i] = func(Population[i])
        curFEs += 1
    current_lifespan = 1


def ChebyshevMap():
    global Dim, GR
    v = np.zeros(Dim)
    v[0] = np.random.rand()
    for i in range(1, Dim):
        v[i] = np.cos(i / np.cos(v[i - 1]))
    return v * GR


def CircleMap():
    global Dim, GR
    b = 0.2
    a = 0.5
    v = np.zeros(Dim)
    v[0] = np.random.rand()
    for i in range(1, Dim):
        v[i] = (v[i - 1] + b - (a / (2 * np.pi)) * np.sin(2 * np.pi * v[0])) % 1
    return GR * (2 * v - 1)


def GaussianMap():
    global Dim, GR
    v = np.zeros(Dim)
    v[0] = np.random.rand()
    for i in range(1, Dim):
        v[i] = 1 / v[i - 1] - int(1 / v[i - 1])
    return GR * (2 * v - 1)


def IterativeMap():
    global Dim, GR
    v = np.zeros(Dim)
    v[0] = np.random.rand()
    for i in range(1, Dim):
        v[i] = np.sin(np.random.rand() * np.pi / v[i - 1])
    return GR * v


def LogisticMap():
    global Dim, GR
    v = np.zeros(Dim)
    r = np.random.rand()
    mu = 4
    while r == 0.25 or r == 0.5 or r == 0.75:
        r = np.random.rand()
    v[0] = np.random.rand()
    for i in range(1, Dim):
        v[i] = mu * v[i - 1] * (1 - v[i - 1])
    return GR * (2 * v - 1)


def SawtoothMap():
    global Dim, GR
    v = np.zeros(Dim)
    v[0] = np.random.rand()
    for i in range(1, Dim):
        v[i] = 2 * v[i - 1] % 1
        if v[i] == 0:
            v[i] = np.random.rand()
    return GR * (2 * v - 1)


def SineMap():
    global Dim, GR
    v = np.zeros(Dim)
    v[0] = np.random.rand()
    for i in range(1, Dim):
        v[i] = np.sin(np.pi * v[i - 1])
    return GR * (2 * v - 1)


def TentMap():
    global Dim, GR
    v = np.zeros(Dim)
    v[0] = np.random.rand()
    for i in range(1, Dim):
        if v[i - 1] < 0.7:
            v[i] = v[i - 1] / 0.7
        else:
            v[i] = 10 / 3 * (1 - v[i - 1])
    return GR * (2 * v - 1)


def Growth(func):
    global Population, Population_fitness, curFEs
    offspring = np.zeros((PopSize, Dim))
    offspring_fitness = np.zeros(PopSize)
    Maps = [ChebyshevMap, CircleMap, GaussianMap, IterativeMap, LogisticMap, SawtoothMap, SineMap, TentMap]
    for i in range(PopSize):
        Chaos = np.random.choice(Maps)
        offspring[i] = Population[i] + Chaos()
        CheckIndi(offspring[i])
        offspring_fitness[i] = func(offspring[i])
        curFEs += 1
        if offspring_fitness[i] < Population_fitness[i]:
            Population_fitness[i] = offspring_fitness[i]
            Population[i] = offspring[i].copy()


def DynamicAllocation():
    global PopSize
    allocation = np.zeros(PopSize)
    for i in range(PopSize):
        allocation[i] = seedNum
    return allocation


def Maturity(func, factor):
    global Population, Population_fitness, Dim, curFEs
    seed = np.zeros((PopSize * seedNum, Dim))
    seed_fitness = np.zeros(PopSize * seedNum)
    X_best = Population[np.argmin(Population_fitness)]
    allocation = DynamicAllocation()
    idx = 0
    for i in range(PopSize):
        for j in range(int(allocation[i])):
            r1, r2 = np.random.choice(list(range(PopSize)), 2, replace=False)
            if np.random.rand() < 0.5:
                for k in range(Dim):
                    seed[idx][k] = np.random.normal(Population[i][k], factor * (abs(X_best[k] - Population[i][k]))) + MS * (
                            2 * np.random.rand() - 1) * (X_best[k] - Population[i][k])
            else:
                seed[idx] = Population[i] + MS * (2 * np.random.rand() - 1) * (Population[r2] - Population[r1])

            r = np.random.rand()
            if r < 1 / 3:
                for m in range(Dim):
                    if np.random.random() < 0.1:
                        seed[idx][m] += 0.05 * np.random.normal() * (UB - LB)
            elif 1 / 3 <= r < 2 / 3:
                for m in range(Dim):
                    if np.random.random() < 0.5:
                        seed[idx][m] = seed[idx][m]
                    else:
                        seed[idx][m] = Population[i][m]
            else:
                for m in range(Dim):
                    if np.random.random() < 0.01:
                        sym = [-1, 1]
                        seed[idx][m] = seed[idx][m] + np.random.choice(sym) * levy.rvs()
            CheckIndi(seed[idx])
            seed_fitness[idx] = func(seed[idx])
            curFEs += 1
            idx += 1
    temp_individual = np.vstack((Population, seed))
    temp_individual_fitness = np.hstack((Population_fitness, seed_fitness))

    tmp = list(map(list, zip(range(len(temp_individual_fitness)), temp_individual_fitness)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(PopSize):
        key, _ = small[i]
        Population_fitness[i] = temp_individual_fitness[key]
        Population[i] = temp_individual[key].copy()


# the implementation process of differential evolution
def VegetationEvolution(func, factor):
    global current_lifespan, GC
    if current_lifespan < GC:
        Growth(func)
        current_lifespan += 1
    elif current_lifespan == GC:
        Maturity(func, factor)
        current_lifespan = 0
    else:
        print("Error: Maximum generation period exceeded.")


def RunVEGE(func):
    global curFEs, Fun_num, Population_fitness, maxFEs
    All_best = []
    for i in range(trialRuns):
        Best_list = []
        curFEs = 0
        np.random.seed(2022 + 88 * i)
        Initialization(func)
        Best_list.append(min(Population_fitness))
        MaxIter = int(maxFEs / 120)
        while curFEs < maxFEs:
            curIter = int(curFEs / 120)
            VegetationEvolution(func, np.log(curIter + 1) / MaxIter)
            Best_list.append(min(Population_fitness))
        All_best.append(Best_list)
    np.savetxt('./iVEGE_Data/CEC2020/F{}_{}D.csv'.format(Fun_num, Dim), All_best, delimiter=",")


def main(dim):
    global Fun_num, Dim, Population, maxFEs
    Dim = dim
    Population = np.zeros((PopSize, Dim))
    maxFEs = Dim * 1000
    CEC2020Funcs = [cec2020.F12020(dim), cec2020.F22020(dim), cec2020.F32020(dim), cec2020.F42020(dim),
                    cec2020.F52020(dim), cec2020.F62020(dim), cec2020.F72020(dim), cec2020.F82020(dim),
                    cec2020.F92020(dim), cec2020.F102020(dim)]

    for i in range(9, len(CEC2020Funcs)):
        Fun_num = i + 1
        RunVEGE(CEC2020Funcs[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./iVEGE_Data/CEC2020') == False:
        os.makedirs('./iVEGE_Data/CEC2020')
    Dims = [100]
    for dim in Dims:
        main(dim)
