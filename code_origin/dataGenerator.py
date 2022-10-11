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

def genClickTrajectory(pattern, traits, weekday):
    # traits: (N, )
    # base rate: (pattern, channel)
    mu_weekday = [[0.3, 0.25, 0.05, 0.1],
               [0.1, 0.05, 0.25, 0.35],
               [0.05, 0.4, 0.05, 0.15]]
    mu_weekend = [[0.15, 0.1, 0.02, 0.05],
               [0.05, 0.02, 0.1, 0.2],
               [0.02, 0.15, 0.02, 0.05]]
    if weekday == True:
        mu = np.array(mu_weekday[pattern])    # array: (channel)
    else:
        mu = np.array(mu_weekend[pattern])    # array: (channel)

    Mu = np.tile(mu, (PARAM['N'],1))    # (N, channel)
    Lambda = 1 * Mu + 1 * traits[:, np.newaxis] # (N, channel)
    return Lambda

def genConverTrajectory(pattern, traits, click):
    # traits: (N, u_dim); click: (N, channel)
    # base rate: (pattern, channel)
    phi = [[0.4, 0.25, 0.2, 0.15],
            [0.15, 0.2, 0.3, 0.3],
            [0.2, 0.25, 0.15, 0.2]]
    # U->Y par: (pattern, u_dim)
    betaU = [[0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.1, 0.4],
            [0.3, 0.1, 0.4, 0.2]]

    Phi = np.tile(np.array(phi[pattern]), (PARAM['N'],1))   # array: copy(channel) -> (N, channel)
    BetaU = np.tile(np.array(betaU[pattern]), (PARAM['N'],1))   # array: copy(u_dim) -> (N, u_dim)
    # The vector with dim=1 is multiplied by the corresponding position
    Lambda1 = np.diag(np.dot(Phi, click.T)) + np.diag(np.dot(BetaU, traits.T))  # (N, )
    return Lambda1

def generateData():
    u = np.random.uniform(0, 1, size=(PARAM['N'],4))    # (N, u_dim)
    ave_u = np.average(u,axis=1)    # (N, )
    pattern = 1
    Lambda = genClickTrajectory(pattern, ave_u, weekday=True)
    click = np.random.poisson(Lambda)   # one day's click, (N, channel)
    Lambda1 = genConverTrajectory(pattern, u, click)
    conver = np.random.poisson(Lambda1)  # one day's conversion, (N, channel)


def main():
    generateData()


SEED = 100
set_seed(SEED)

PARAM = {
    'N': 100,
    'T': 120,
}

if __name__ == "__main__":
    main()