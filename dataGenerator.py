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
        mu = mu_weekday[pattern]    # array: (channel)
    else:
        mu = mu_weekend[pattern]    # array: (channel)

    Lambda = [i+j for i,j in zip(mu,[traits]*4)]
    return Lambda

def genConverTrajectory(pattern, traits, click):
    # traits: (u_dim); click: (channel)
    # base rate: (pattern, channel)
    phi = [[0.4, 0.25, 0.2, 0.15],
            [0.15, 0.2, 0.3, 0.3],
            [0.2, 0.25, 0.15, 0.2]]
    # U->Y par: (pattern, u_dim)
    betaU = [[0.02, 0.04, 0.06, 0.08],
            [0.04, 0.06, 0.02, 0.08],
            [0.06, 0.02, 0.08, 0.04]]

    Phi, BetaU = phi[pattern], betaU[pattern]
    path1 = [i*j for i,j in zip(Phi, click)]
    path2 = [i*j for i,j in zip(BetaU, traits)]
    Lambda1 = np.sum(path1)+np.sum(path2)
    return Lambda1

def generateData():
    all_Click, all_Conver, all_u, uid, all_pattern = [], [], [], [], []

    for i in range(PARAM['N']):
        #Click, Conver = np.array([[0,0,0,0]]), np.array([0])
        Click, Conver, Pattern = [], [], []
        pattern = np.random.choice([0, 1, 2], p=[0.35, 0.33, 0.32])
        u = np.random.uniform(0, 1, size=4)
        ave_u = np.average(u)

        for t in range(PARAM['T']):
            Lambda = genClickTrajectory(pattern, ave_u, weekday=True)
            click = np.random.poisson(Lambda)   # one day's click, (channel)
            Lambda1 = genConverTrajectory(pattern, u, click)
            conver = np.random.poisson(Lambda1)  # one day's conversion, (1)
            Pattern.append(pattern)
            # pattern shift
            pattern = transitionFunc(pattern)
            Click.append(click)
            Conver.append(conver)
        # Click: list(T, channel), Conver: list(T, 1)
        all_u.append(u)
        all_Click.append(Click)
        all_Conver.append(Conver)
        all_pattern.append(Pattern)
        uid.append([i]*PARAM['T'])

    all_Click = np.array(all_Click).reshape(-1, 4)   # (N, T, channel) -> (N*T, channel)
    all_Conver = np.array(all_Conver).reshape(-1, 1)    # (N, T, 1) -> (N*T, 1)
    all_pattern = np.array(all_pattern).reshape(-1, 1)
    all_u, all_uid = np.array(all_u), np.array([i for i in range(PARAM['N'])])[:,np.newaxis] # ndarray(N, u_dim), (N, 1)

    time = np.array([i for i in range(PARAM['T'])] * PARAM['N'])[:,np.newaxis]
    uid = np.array(uid).reshape(-1,1)
    temporal_dt = np.concatenate((uid, time, all_Click, all_Conver, all_pattern), axis=-1)  # (N*T, 1+1+channel+conver+1)
    temporal_dt = pd.DataFrame(temporal_dt, columns=['uid', 'time', 'chan0', 'chan1', 'chan2', 'chan3', 'conver', 'pattern'])
    ux_dt = np.concatenate((all_uid, all_u), axis=-1)   # (N, 1+u_dim)
    ux_dt = pd.DataFrame(ux_dt, columns=['uid', 'u0', 'u1', 'u2', 'u3'])
    temporal_dt.to_csv('trajectory.csv', index=False)
    ux_dt.to_csv('ufeature.csv', index=False)

def main():
    generateData()


SEED = 100
set_seed(SEED)

PARAM = {
    'N': 100,
    'T': 500,
}

if __name__ == "__main__":
    main()