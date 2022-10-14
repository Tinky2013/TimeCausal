import random
import pandas as pd
import numpy as np
import math

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(11, 20, 10)
        self.y = torch.linspace(1, 10, 10)
        self.len = len(self.x)

        self._feature = None
        self._label = None
        self._index = None

        self.seq_len = 48
        self.horizon = 0
        self.num_states = 3
        self.batch_size = 20
        self.shuffle = True

        # add memory to feature -> self._data: (N, channel + num_pattern)
        self._data = np.c_[self.feature, np.zeros((len(self._feature), self.num_states), dtype=np.float32)]
        # padding tensor -> self.zeros: (seq_len, channel + num_pattern)
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)
        # create batch slices
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)

    def __getitem__(self, index):
        return self._feature[index], self._label[index]

    def __len__(self):
        return len(self._feature)

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
    def __init__(self, input_size, num_states=3, hidden_size=8,):
        super().__init__()
        self.num_state = num_states

        self.router = torch.nn.LSTM(
            input_size = self.num_state,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True,
        )
        self.fc = torch.nn.Linear(hidden_size + input_size, num_states)
        self.predictors = torch.nn.Linear(input_size, num_states)

        # self.extractor = Extractor(hidden_dim=H_dim)
        # self.fc = torch.nn.Linear(in_dim, self.num_state)
        # self.predictors = Predictor(hidden_dim=H_dim)

    def forward(self, f1, f0, Train=True):
        pass
        # emb1, emb0 = self.extractor(f1, f0)
        # preds1, preds0 = self.predictors(emb1, emb0)
        # input = torch.cat([emb1, emb0], dim=0)
        # out = self.fc(input)
        # prob = F.gumbel_softmax(out, dim=-1, tau=1, hard=False)
        #
        # if Train:
        #     final_pred1 = (preds1 * prob[:len(emb1),:]).sum(dim=-1)
        #     final_pred0 = (preds0 * prob[len(emb1):, :]).sum(dim=-1)
        # else:
        #     final_pred1 = preds1[range(len(preds1)), prob[:len(emb1),:].argmax(dim=-1)]
        #     final_pred0 = preds0[range(len(preds0)), prob[len(emb1):, :].argmax(dim=-1)]
        #
        # return final_pred1, final_pred0, preds1, preds0, prob

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
    static = pd.read_csv('ufeature.csv')
    dynamic = pd.read_csv('trajectory.csv')

    tra = Tra().to(device)
    optimizer = torch.optim.Adam(tra.parameters(), lr=0.01)

    def train_epoch(train_data):
        count = 0
        total_loss = 0
        total_count = 0

        tra.train()
        for batch in tqdm(train_data, total=PARAM['max_step_per_epoch']):
            # data: (batch, N, channel + num_pattern)
            data, label, index = None, None, None    # click, conver, <uid, time>
            feature = data[:, :, : -3]  # (batch, N, channel)
            hist_loss = data[:, : -horizon, -3:]    # (batch, N-horizon, channel)

            hidden = model(feature)
            pred, all_preds, prob = tra(hidden, hist_loss)

            loss = (pred - label).pow(2).mean()

            L = (all_preds.detach() - label[:, None]).pow(2)
            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

            assign_data(index, L)

            if prob is not None:
                P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
                lamb = lamb * (rho ** global_step)
                reg = prob.log().mul(P).sum(dim=-1).mean()
                loss = loss - lamb * reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_count += len(pred)

        total_loss /= total_count

        return total_loss

        pass

    def test_epoch(data):
        pass

    # prepare the data
    train_set, valid_set, test_set = None, None, None
    # initialize the memory
    test_epoch(train_set)

    for epoch in range(PARAM['n_epoch']):
        train_epoch(train_set)
        # during evaluating, the whole memory will be refreshed
        train_set.clear_memory()
        train_metrics = test_epoch(train_set)
        valid_metrics = test_epoch(valid_set)

    metrics, preds = test_epoch(test_set)

SEED = 100
set_seed(SEED)

PARAM = {
    'n_epoch': 100,
    'max_step_per_epoch': 20,
}

if __name__ == "__main__":
    main()