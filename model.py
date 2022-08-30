import random
import pandas as pd
import numpy as np
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
    def __init__(self, hidden_dim):
        super(Extractor, self).__init__()
        # self.NN = torch.nn.Sequential(
        #     torch.nn.Linear(3, hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, H_dim)
        # )
        self.NN = torch.nn.Linear(3, hidden_dim)

    def forward(self, f1, f0):
        emb1 = self.NN(f1)
        emb0 = self.NN(f0)
        return emb1, emb0


class Predictor(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Predictor, self).__init__()
        # self.NN0 = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, H_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H_dim, 3)
        # )
        # self.NN1 = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, H_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H_dim, 3)
        # )
        self.NN0 = torch.nn.Linear(H_dim, 3)
        self.NN1 = torch.nn.Linear(H_dim, 3)

    def forward(self, emb1, emb0):
        pred_y1 = self.NN1(emb1)
        pred_y0 = self.NN0(emb0)
        return pred_y1, pred_y0

    def loss(self, y1_pred, y0_pred, y1, y0):
        return F.mse_loss(y1_pred, y1) + F.mse_loss(y0_pred, y0)

class MyModel(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.num_state = 3
        self.extractor = Extractor(hidden_dim=H_dim)
        self.fc = torch.nn.Linear(in_dim, self.num_state)
        self.predictors = Predictor(hidden_dim=H_dim)

    def forward(self, f1, f0, Train=True):
        emb1, emb0 = self.extractor(f1, f0)
        preds1, preds0 = self.predictors(emb1, emb0)
        input = torch.cat([emb1, emb0], dim=0)
        out = self.fc(input)
        prob = F.gumbel_softmax(out, dim=-1, tau=1, hard=False)

        if Train:
            final_pred1 = (preds1 * prob[:len(emb1),:]).sum(dim=-1)
            final_pred0 = (preds0 * prob[len(emb1):, :]).sum(dim=-1)
        else:
            final_pred1 = preds1[range(len(preds1)), prob[:len(emb1),:].argmax(dim=-1)]
            final_pred0 = preds0[range(len(preds0)), prob[len(emb1):, :].argmax(dim=-1)]

        # final_pred1 = preds1[range(len(preds1)), prob[:len(emb1),:].argmax(dim=-1)]
        # final_pred0 = preds0[range(len(preds0)), prob[len(emb1):, :].argmax(dim=-1)]
        # if not Train:
        #     for i in range(len(f1)):
        #         print(prob[i], preds1[i], f1[i])
        #     for i in range(len(f1), len(f1)+len(f0)):
        #         print(prob[i], preds0[i-len(f1)], f0[i-len(f1)])
        return final_pred1, final_pred0, preds1, preds0, prob


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = pd.read_csv('time_dt.csv')
    T = np.array(dt['T'])
    col_pre = ['pre_'+str(i) for i in range(1,121)]
    col_post = ['post_'+str(i) for i in range(1,21)]
    preTime, postTime = np.array(dt[col_pre]), np.array(dt[col_post])

    # train-val split
    preTime_train, preTime_val = preTime[:850], preTime[850:]
    postTime_train, postTime_val = postTime[:850], postTime[850:]
    T_train, T_val = T[:850], T[850:]

    dt_train = CausalData(preTime_train, postTime_train, T_train)
    dt_val = CausalData(preTime_val, postTime_val, T_val)

    trainset = DataLoader.DataLoader(dt_train, batch_size=16, shuffle=True)
    valset = DataLoader.DataLoader(dt_val)

    myModel = MyModel(in_dim=H_dim).to(device)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=0.01)

    def train():
        myModel.train()
        total_loss = 0.0
        for i, (dt, labels) in enumerate(trainset):
            df, labels = dt.to(device), labels.to(device)
            optimizer.zero_grad()
            output = myModel(dt, labels)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    def test():
        print("testing the model")
        myModel.eval()
        with torch.no_grad():
            for dt, labels in valset:
                dt, labels = dt.to(device), labels.to(device)
                output = myModel(dt, labels)


    for epoch in range(1000):
        train()
        if epoch%100==0:
            test()

H_dim = 2
SEED = 100

if __name__ == "__main__":
    set_seed(SEED)
    main()