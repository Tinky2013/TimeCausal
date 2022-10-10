import random
import pandas as pd
import numpy as np
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def transitionFunc(pattern):
    if pattern == 0:
        return np.random.choice(3, 1, p=[0.9, 0.05, 0.05])
    elif pattern == 1:
        return np.random.choice(3, 1, p=[0.05, 0.9, 0.05])
    else:
        return np.random.choice(3, 1, p=[0.05, 0.05, 0.9])

def genTrajectory(pattern, weekday, traits):
    # base rate: (pattern, channel)
    mu_weekday = [[0.3, 0.25, 0.05, 0.1],
               [0.1, 0.05, 0.25, 0.35],
               [0.05, 0.4, 0.05, 0.15]]
    mu_weekend = [[0.15, 0.1, 0.02, 0.05],
               [0.05, 0.02, 0.1, 0.2],
               [0.02, 0.15, 0.02, 0.05]]
    if weekday == True:
        mu = mu_weekday[pattern]
    else:
        mu = mu_weekend[pattern]

    Lambda = 1 * mu + 1 * traits
    return Lambda



def generateData():
    u = np.random.uniform(0, 1, size=(PARAM['N'],4))
    print(u.shape)
    print(np.mean(u))


def main():
    # channel
    c1, c2, c3, c4 = [], [], [], []
    generateData()


SEED = 100
set_seed(SEED)

PARAM = {
    'N': 100,
}

if __name__ == "__main__":
    main()