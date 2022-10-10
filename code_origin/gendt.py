import random
import pandas as pd
import numpy as np
import math

import os
import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

class Tra(torch.nn.Module):
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

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = pd.read_csv('data.csv')
    dt1 = dt[dt['t']==1]
    dt0 = dt[dt['t']==0]
    f1 = torch.tensor(np.array(dt1[['z1','z2','z3']]), dtype=torch.float).to(device)
    f0 = torch.tensor(np.array(dt0[['z1', 'z2', 'z3']]), dtype=torch.float).to(device)
    y1 = torch.tensor(np.array(dt1[['y']]), dtype=torch.float).to(device)
    y0 = torch.tensor(np.array(dt0[['y']]), dtype=torch.float).to(device)


    dtc1 = dt[dt['cont']==1]
    dtc0 = dt[dt['cont']==0]
    fc1 = torch.tensor(np.array(dtc1[['z1','z2','z3']]), dtype=torch.float).to(device)
    fc0 = torch.tensor(np.array(dtc0[['z1','z2','z3']]), dtype=torch.float).to(device)
    yc1 = torch.tensor(np.array(dtc1[['cony']]), dtype=torch.float).to(device)
    yc0 = torch.tensor(np.array(dtc0[['cony']]), dtype=torch.float).to(device)

    tra = Tra(in_dim=H_dim).to(device)
    optimizer = torch.optim.Adam(tra.parameters(), lr=0.01)


    def train():
        tra.train()
        optimizer.zero_grad()
        # y1_pred, y0_pred = model(f1, f0)

        pred1, pred0, all_preds1, all_preds0, prob = tra(f1, f0, Train=True)
        # size: (252), (248), (252,3), (248,3), (252+248,3)
        loss = F.mse_loss(pred1, y1.squeeze(-1)) + F.mse_loss(pred0, y0.squeeze(-1))
        y1_all, y0_all = y1.repeat(1,3), y0.repeat(1,3)

        L = torch.cat([torch.pow((all_preds1.detach() - y1_all), 2), torch.pow((all_preds0.detach() - y0_all), 2)], dim=0)
        L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

        if prob is not None:
            P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
            # print("loss:", L[0])
            # print("p:", P[0])
            reg = prob.log().mul(P).sum(dim=-1).mean()
            loss = loss - lamb * reg * (rho ** (epoch+1))


        loss.backward()
        optimizer.step()
        #print("train loss:", loss.item())

    def test():
        print("testing the model")
        tra.eval()
        with torch.no_grad():
            pred1, pred0, all_preds1, all_preds0, prob = tra(fc1, fc0, Train=False)

        yc1_all, yc0_all = yc1.repeat(1, 3), yc0.repeat(1, 3)
        L = torch.cat([torch.pow((all_preds1.detach() - yc1_all), 2), torch.pow((all_preds0.detach() - yc0_all), 2)],
                      dim=0)
        L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
        loss = F.mse_loss(pred1, yc1.squeeze(-1))+F.mse_loss(pred0, yc0.squeeze(-1))

        print("test loss:", loss.detach().numpy())

        yc1s = yc1.reshape(1,-1)[0]
        yc0s = yc0.reshape(1, -1)[0]
        true = torch.cat([yc1s,yc0s],dim=0)

        pred = torch.cat([pred1,pred0],dim=0)
        result = pd.DataFrame({
            'true': np.array(true),
            'pred': np.array(pred),
        })
        result.to_csv(('result.csv'),index=False)

    lamb = 0.5
    rho = 0.99
    for epoch in range(1000):
        train()
        if epoch%100==0:
            test()

H_dim = 2
SEED = 100
set_seed(SEED)

if __name__ == "__main__":
    main()