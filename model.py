import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import torch
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


class Extractor(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=120):
        super(Extractor, self).__init__()
        self.NN = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, pretime):
        emb = self.NN(pretime)
        return emb


class Predictor(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim=20):
        super(Predictor, self).__init__()
        self.NN = torch.nn.Linear(PARAM['Hidden'], output_dim)

    def forward(self, emb):
        pred = self.NN(emb)
        return pred

    def loss(self, y1_pred, y0_pred, y1, y0):
        return F.mse_loss(y1_pred, y1) + F.mse_loss(y0_pred, y0)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = Extractor(input_dim=120, hidden_dim=PARAM['Hidden'])
        self.predictor0 = Predictor(hidden_dim=PARAM['Hidden'])
        self.predictor1 = Predictor(hidden_dim=PARAM['Hidden'])

    def forward(self, pretime, t0_idx, t1_idx):
        emb = self.extractor(pretime)   # (batch, hidden_dim)
        emb0 = emb[t0_idx]    # (batch[t==1], hidden_dim)
        emb1 = emb[t1_idx]    # (batch[t==0], hidden_dim)
        preds0, preds1 = self.predictor0(emb0), self.predictor1(emb1)   # preds: (batch[t==j], posttime_len)
        return preds0, preds1

def criterion(pred0, pred1, posttime0, posttime1):
    assert len(pred0)==len(posttime0), "lenght not match!"
    assert len(pred1) == len(posttime1), "lenght not match!"
    preds = torch.cat([pred0, pred1],dim=0)  # (batch, posttime_len)
    posttime = torch.cat([posttime0, posttime1],dim=0)  # (batch, posttime_len)
    return F.mse_loss(preds, posttime)

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
    print(len(Tc))

    trainset = DataLoader.DataLoader(dt_train, batch_size=PARAM['batch_size'], shuffle=True)
    testset = DataLoader.DataLoader(dt_test, batch_size=len(Tc))

    myModel = MyModel().to(device)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=0.01)

    # training
    def train():
        for epoch in range(200):
            myModel.train()
            total_loss = 0.0
            for i, (pretime, posttime, treat) in enumerate(trainset):
                # (batch, pretime_len), (batch, posttime_len), (batch, 1)
                pretime, posttime, treat = pretime.to(device), posttime.to(device), treat.to(device)
                optimizer.zero_grad()

                t0_idx = np.array(np.argwhere(treat == 0)[0])
                t1_idx = np.array(np.argwhere(treat == 1)[0])

                posttime0 = posttime[t0_idx]  # (batch[t==1], posttime_len)
                posttime1 = posttime[t1_idx]  # (batch[t==0], posttime_len)

                pred0, pred1 = myModel(pretime, t0_idx, t1_idx)
                loss = criterion(pred0, pred1, posttime0, posttime1)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 20 == 0:
                print("epoch:", epoch, "loss:", total_loss)

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
                pred0, pred1 = myModel(pretime, t0_idx, t1_idx) # pred: # (N[t==j], posttime_len)
                metric = criterion(pred0, pred1, counterposttime0, counterposttime1).detach().numpy()

                print("test mse:", metric)

            timeplot(pred0[0], counterposttime0[0])

    train()
    test()


PARAM = {
    'Hidden': 4,
    'batch_size': 16,
}

SEED = 100

if __name__ == "__main__":
    set_seed(SEED)
    main()