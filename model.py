import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CausalData(Dataset.Dataset):
    def __init__(self, preData, postData, Treat):
        self.preData = preData
        self.postData = postData
        self.Treat = Treat

    def __len__(self):
        return len(self.Treat)

    def __getitem__(self, index):
        predata = torch.Tensor(self.preData[index])
        postdata = torch.Tensor(self.postData[index])
        treat = torch.IntTensor(self.Treat[index])
        return predata, postdata, treat

class lstmExtractor(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=1):
        super(lstmExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim, # time series feature dim
            hidden_size = hidden_dim,   # latent dim
            num_layers = 1,     # lstm layers
            batch_first=True,   # input: (batch, seq_len, input_size), output: (batch, seq_len, hidden_size), (hn, cn)
            dropout = 0.2,
        )

    def forward(self, pretime):
        pretime = pretime.unsqueeze(-1) # (batch, pretime_len, 1)
        emb = self.lstm(pretime)[0]
        emb = emb.transpose(1,2)   # (batch, seq_len, hidden_size) -> (batch, hidden_size, seq_len)
        return emb[:,:,-1]  # the last timestep's emb: (batch, hidden_size)

class nnExtractor(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=120):
        super(nnExtractor, self).__init__()
        self.NN = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, pretime):
        emb = self.NN1(pretime)
        return emb


class Predictor(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim=20):
        super(Predictor, self).__init__()
        #self.NN = torch.nn.Linear(PARAM['Hidden'], output_dim)
        self.NN1 = torch.nn.Linear(PARAM['Hidden'], 8)
        self.NN2 = torch.nn.Linear(8, output_dim)

    def forward(self, emb):
        mid = self.NN1(emb)
        pred = self.NN2(mid)
        return pred

class PropensityNet(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(PropensityNet, self).__init__()
        self.NN = torch.nn.Linear(hidden_dim,1)

    def forward(self, emb):
        pred_pro = self.NN(emb)
        return pred_pro


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.extractor = nnExtractor(input_dim=120, hidden_dim=PARAM['Hidden'])
        self.extractor = lstmExtractor(hidden_dim=PARAM['Hidden'])
        self.predictor0 = Predictor(hidden_dim=PARAM['Hidden'])
        self.predictor1 = Predictor(hidden_dim=PARAM['Hidden'])
        self.propensitynet = PropensityNet(hidden_dim=PARAM['Hidden'])

    def forward(self, pretime, t0_idx, t1_idx):
        emb = self.extractor(pretime)   # (batch, hidden_dim)
        emb0 = emb[t0_idx]    # (batch[t==1], hidden_dim)
        emb1 = emb[t1_idx]    # (batch[t==0], hidden_dim)
        preds0, preds1 = self.predictor0(emb0), self.predictor1(emb1)   # preds: (batch[t==j], posttime_len)
        pred_pro0, pred_pro1 = self.propensitynet(emb0), self.propensitynet(emb1)
        return preds0, preds1, pred_pro0, pred_pro1

def criterion(pred0, pred1, posttime0, posttime1, pred_pro0, pred_pro1):
    assert len(pred0)==len(posttime0), "lenght not match!"
    assert len(pred1) == len(posttime1), "lenght not match!"
    preds = torch.cat([pred0, pred1],dim=0)  # (batch, posttime_len)
    posttime = torch.cat([posttime0, posttime1],dim=0)  # (batch, posttime_len)

    t0 = torch.zeros(pred_pro0.shape)
    t1 = torch.ones(pred_pro1.shape)

    return F.mse_loss(preds, posttime) + PARAM['pro_reg'] * (F.mse_loss(pred_pro0, t0)+F.mse_loss(pred_pro1, t1))

def timeplot(pred, fact):
    assert len(pred) == len(fact), "lenght not match!"
    F = len(pred)
    xticks = np.arange(F)
    plt.figure(figsize=(8, 6))
    plt.plot(xticks, pred, 'x', label='predict')
    plt.plot(xticks, fact, 's', label='observed')
    plt.legend()
    plt.title('Forecasts for counterfactual')
    plt.xlabel('Time step')
    plt.show()

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

    myModel = MyModel().to(device)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=0.01)

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
                loss = criterion(pred0, pred1, posttime0, posttime1, pred_pro0, pred_pro1)
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

            timeplot(pred0[0], counterposttime0[0])

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