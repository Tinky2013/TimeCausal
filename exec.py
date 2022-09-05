import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as DataLoader

from model import MyModel
from util.causalData import CausalData
from util.visual import AllPlot
from util.criteria import AllLoss
from modeling.net import lstmExtractor, nnExtractor, Predictor, PropensityNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = pd.read_csv('time_dt.csv')
    T = np.array(dt['T']).reshape(-1,1) # (N, 1)
    col_pre = ['pre_'+str(i) for i in range(1,121)]
    col_post = ['post_'+str(i) for i in range(1,21)]
    col_counterpost = ['counterpost_' + str(i) for i in range(1, 21)]
    preTime, postTime, counterpostTime = np.array(dt[col_pre]), np.array(dt[col_post]), np.array(dt[col_counterpost])   # (N, pretime_len), (N, posttime_len), (N, posttime_len)

    Tc = T.copy()
    Tc[np.where(T == 0)] = 1
    Tc[np.where(T == 1)] = 0

    dt_train = CausalData(preTime, postTime, T)
    dt_test = CausalData(preTime, counterpostTime, Tc)

    trainset = DataLoader.DataLoader(dt_train, batch_size=PARAM['batch_size'], shuffle=True)
    testset = DataLoader.DataLoader(dt_test, batch_size=len(Tc))

    myModel = MyModel(PARAM).to(device)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=0.01)

    criterion = AllLoss(PARAM)
    allplot = AllPlot(figsize=(8,6))

    # training
    def train():
        for epoch in range(40):
            myModel.train()
            total_loss = []
            for i, (pretime, posttime, treat) in enumerate(trainset):
                # (batch, pretime_len), (batch, posttime_len), (batch, 1)
                pretime, posttime, treat = pretime.to(device), posttime.to(device), treat.to(device)
                optimizer.zero_grad()

                t0_idx = np.array(np.argwhere(treat == 0)[0])
                t1_idx = np.array(np.argwhere(treat == 1)[0])

                posttime0 = posttime[t0_idx]  # (batch[t==1], posttime_len)
                posttime1 = posttime[t1_idx]  # (batch[t==0], posttime_len)

                pred0, pred1, pred_pro0, pred_pro1 = myModel(pretime, t0_idx, t1_idx)
                loss = criterion.propensity_loss(pred0, pred1, posttime0, posttime1, pred_pro0, pred_pro1)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.detach().numpy())

            if epoch % 2 == 0:
                print("epoch:", epoch, "loss:", np.mean(total_loss))

    # testing
    def test():
        print("testing the model")
        myModel.eval()
        with torch.no_grad():
            for pretime, counterposttime, treat in testset:
                pretime, counterposttime, treat = pretime.to(device), counterposttime.to(device), treat.to(device)

                t0_idx = np.array(np.argwhere(treat == 0)[0])
                t1_idx = np.array(np.argwhere(treat == 1)[0])

                counterposttime0 = counterposttime[t0_idx]  # (N[t==0], posttime_len)
                counterposttime1 = counterposttime[t1_idx]  # (N[t==1], posttime_len)

                # pred0: actually receive T=1, predict counter factual T=0
                pred0, pred1, _, _ = myModel(pretime, t0_idx, t1_idx) # pred: # (N[t==j], posttime_len)

                preds = torch.cat([pred0, pred1], dim=0)  # (batch, posttime_len)
                posttime = torch.cat([counterposttime0, counterposttime1], dim=0)  # (batch, posttime_len)
                metric = F.mse_loss(preds, posttime).detach().numpy()

                print("test mse:", metric)

            allplot.timeplot(pred0[0], counterposttime0[0])

    train()
    test()


PARAM = {
    'Hidden': 4,
    'batch_size': 64,
    'pro_reg': 0.05,
}

SEED = 100

if __name__ == "__main__":
    set_seed(SEED)
    main()