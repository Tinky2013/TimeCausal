import random
import pandas as pd
import numpy as np
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def transitionFunc(pattern):
    if pattern == 0:
        return np.random.choice([0,1,2], p=[0.9, 0.05, 0.05])
    elif pattern == 1:
        return np.random.choice([0,1,2], p=[0.05, 0.9, 0.05])
    else:
        return np.random.choice([0,1,2], p=[0.05, 0.05, 0.9])

def genClickTrajectory(pattern, traits, weekday):
    # traits: (1, )
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

    Mu = np.tile(mu, (1,1))    # (1, channel)
    Lambda = 1 * Mu + 1 * traits[:, np.newaxis] # (1, channel)
    return Lambda

def genConverTrajectory(pattern, traits, click):
    # traits: (1, u_dim); click: (1, channel)
    # base rate: (pattern, channel)
    phi = [[0.4, 0.25, 0.2, 0.15],
            [0.15, 0.2, 0.3, 0.3],
            [0.2, 0.25, 0.15, 0.2]]
    # U->Y par: (pattern, u_dim)
    betaU = [[0.02, 0.04, 0.06, 0.08],
            [0.04, 0.06, 0.02, 0.08],
            [0.06, 0.02, 0.08, 0.04]]

    Phi = np.tile(np.array(phi[pattern]), (1,1))   # array: copy(channel) -> (1, channel)
    BetaU = np.tile(np.array(betaU[pattern]), (1,1))   # array: copy(u_dim) -> (1, u_dim)
    # The vector with dim=1 is multiplied by the corresponding position
    Lambda1 = np.diag(np.dot(Phi, click.T)) + np.diag(np.dot(BetaU, traits.T))  # (1, )
    return Lambda1

def generateData():
    all_Click, all_Conver = [], []
    for i in range(PARAM['N']):
        Click, Conver = np.array([[0,0,0,0]]), np.array([0])
        pattern = np.random.choice([0, 1, 2], p=[0.35, 0.33, 0.32])
        u = np.random.uniform(0, 1, size=(1, 4))  # (1, u_dim)
        ave_u = np.average(u, axis=1)  # (1, )
        for t in range(PARAM['T']):
            Lambda = genClickTrajectory(pattern, ave_u, weekday=True)
            click = np.random.poisson(Lambda)   # one day's click, (1, channel)
            Lambda1 = genConverTrajectory(pattern, u, click)
            conver = np.random.poisson(Lambda1)  # one day's conversion, (1, channel)
            pattern = transitionFunc(pattern)

            Click = np.concatenate((Click, click), axis=0)
            Conver = np.concatenate((Conver, conver), axis=0)
        # for an individual, Click: (T+1, channel); Conver: (T+1,)
        all_Click.append(Click), all_Conver.append(Conver)

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