import os
import numpy as np
from os.path import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import math
from scipy.stats import levy
import warnings
warnings.filterwarnings("ignore")


File_name = 'cervical cancer risk.csv'
This_path = dirname(dirname(abspath(__file__))) + "\\Problems\\Datasets\\" + File_name
Dataset = None

PopSize = 10
Dim = 10
LB = -10
UB = 10
trialRuns = 30
maxFEs = 1000
GC = 6
GR = 1
MS = 2
seedNum = 6

Population = np.zeros((PopSize, Dim))
Population_fitness = np.zeros(PopSize)
current_lifespan = 0
curFEs = 0
CurIter = 0
Best_Acc = np.zeros(trialRuns)


def read_data():
    global This_path
    Data = pd.read_csv(This_path, sep=',', na_values=["?"])
    le = LabelEncoder()
    cols = [Data.columns[x] for x in range(len(Data.columns))]
    for i in range(len(Data.columns)):
        if Data[cols[i]].dtypes == 'object':
            temp = le.fit_transform(Data[cols[i]])
            m = pd.DataFrame(temp)
            Data[cols[i]] = m
    shape = Data.shape[0]
    trainData = Data.values
    return trainData


def read_data_label():
    global File_name, This_path, Dataset
    Data = Dataset
    dim = len(Data[0])
    if "mushroom" in File_name:
        label = Data[:, 0]
        data = Data[:, 1:dim]
    else:
        label = Data[:, dim-1]
        data = Data[:, 0:dim-1]
    for i in range(len(data)):
        for j in range(len(data[i])):
            if math.isnan(data[i][j]) is True:
                data[i][j] = not_nan_mean(data[:, j])
    return data, label


def not_nan_mean(vector):
    add = 0
    size = 0
    for e in vector:
        if math.isnan(e) is False:
            add += e
            size += 1
    return int(add / size)


def KNN_accuracy(train_X, train_y, test_X, test_y):
    knn = KNeighborsClassifier()
    knn.fit(train_X, train_y)
    pred_y = knn.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    return 1 - acc


def feature_select(indi, Data):
    ndim = 0
    size = len(Data)
    for i in indi:
        if i == 1:
            ndim += 1
    nData = np.zeros((size, ndim))
    flag = 0
    for i in range(len(indi)):
        if indi[i] == 1:
            nData[:, flag] = Data[:, i]
            flag += 1
    return nData


def k_fold_error(Data, Label):
    k = 5
    kfold = KFold(n_splits=k)
    Error = np.zeros(k)
    for fold_, (train_idx, test_idx) in enumerate(kfold.split(Data)):
        train_X, train_y, test_X, test_y = Data[train_idx], Label[train_idx], Data[test_idx], Label[test_idx]
        Error[fold_] = KNN_accuracy(train_X, train_y, test_X, test_y)
    return np.mean(Error)


def Binary(X, epsilon=0.5):
    dim = len(X)
    bX = np.zeros(dim)
    flag = 0
    for i in range(dim):
        if 1 / (1 + np.exp(-X[i])) > epsilon:
            bX[i] = 1
            flag += 1
    return bX, flag / dim


def Evaluation(indi, c1=0.99, c2=0.01):
    global CurIter, Best_Acc
    Data, Label = read_data_label()
    bIndi, rate = Binary(indi)
    if rate == 0:
        return 1
    else:
        nData = feature_select(bIndi, Data)
        error = k_fold_error(nData, Label)
        Best_Acc[CurIter] = max(1 - error, Best_Acc[CurIter])
        return error * c1 + rate * c2


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


def RunVEGE(prob, func):
    global curFEs, Population_fitness, CurIter, Best_Acc
    All_best = []
    Best_Acc = np.zeros(trialRuns)
    for i in range(trialRuns):
        Best_list = []
        CurIter = i
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
    np.savetxt('./iVEGE_Data/FS/{}.csv'.format(prob), All_best, delimiter=",")
    np.savetxt('./iVEGE_Data/FS/{}_ac.csv'.format(prob), [Best_Acc], delimiter=",")


def main():
    global Dim, Population, maxFEs, File_name, This_path, Dataset
    maxFEs = 1000
    Dims = [20, 21, 10, 14, 16, 27, 20]
    file_names = ['mobile price.csv', 'fetal health.csv', 'diabetes.csv', 'heart disease.csv', 'dry bean.csv',
                  'music genre.csv', 'water quality.csv']
    probs = ['MP', 'FH', 'diabetes', 'HD', 'DB', 'MG', 'WQ']

    for i in range(len(Dims)):
        File_name = file_names[i]
        This_path = dirname(dirname(abspath(__file__))) + "\\Problems\\Datasets\\" + file_names[i]
        Dataset = read_data()
        # print(Dataset.shape)
        Dim = Dims[i]
        Population = np.zeros((PopSize, Dim))
        RunVEGE(probs[i], Evaluation)


if __name__ == "__main__":
    if os.path.exists('./iVEGE_Data/FS') == False:
        os.makedirs('./iVEGE_Data/FS')
    main()
