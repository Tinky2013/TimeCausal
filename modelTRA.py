import random
import pandas as pd
import numpy as np
import math

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _create_ts_slices(index, seq_len):
    """
    create time series slices from pandas index
    Args:
        index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
        seq_len (int): sequence length
    """
    assert index.is_lexsorted(), "index should be sorted"

    # number of dates for each code
    sample_count_by_codes = pd.Series(0, index=index).groupby(level=0).size().values

    # start_index for each code
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    start_index_of_codes[0] = 0

    # all the [start, stop) indices of features
    # features btw [start, stop) are used to predict the `stop - 1` label
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices)
    return slices

class MyDataset(Dataset):
    def __init__(self, feature, label, index):
        self._feature = feature
        self._label = label
        self._index = index

        self.seq_len = 48
        self.horizon = 0
        self.num_states = 3
        self.batch_size = 20
        self.shuffle = True

        # add memory to feature -> self._data: (N, channel + num_pattern)
        self._data = np.c_[self._feature, np.zeros((len(self._feature), self.num_states), dtype=np.float32)]
        # padding tensor -> self.zeros: (seq_len, channel + num_pattern)
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)
        # create batch slices
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)

    def __getitem__(self, id):
        return self._feature[id], self._label[id], self._index[id]

    def __len__(self):
        return len(self._feature)

    def assign_data(self, index, vals):
        vals = vals.detach().cpu().numpy()
        index = index.detach().cpu().numpy()
        self._data[index, -self.num_states :] = vals

    def clear_memory(self):
        self._data[:, -self.num_states :] = 0



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

class TRAModel(object):
    def __init__(self):
        self.tra = Tra().to(device)
        self.optimizer = torch.optim.Adam(self.tra.parameters(), lr=0.01)
        self.global_step = -1

def train_epoch(train_data):
    traModel = TRAModel()
    traModel.tra.train()
    count = 0
    total_loss = 0
    total_count = 0
    max_step = PARAM['max_step_per_epoch']

    for batch in tqdm(train_data, total=max_step):
        count += 1
        if count > max_step:
            break
        traModel.global_step += 1

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


def test_epoch(data):
    pass

def main():

    static = pd.read_csv('ufeature.csv').set_index('uid')
    dynamic = pd.read_csv('trajectory.csv').set_index('uid')
    dynamic = dynamic.set_index('time', append=True)

    index = dynamic.index
    feature = dynamic[['chan0','chan1','chan2','chan3']].values.astype("float32")
    label = dynamic[['conver']].values.astype("float32")

    HData = MyDataset(feature=feature, label=label, index=index)

    # prepare the data
    train_set, valid_set, test_set = None, None, None
    # train
    global_step = -1
    if PARAM['pattern']>1:
        test_epoch(train_set)   # initialize the memory

    for epoch in range(PARAM['n_epoch']):
        print("Training Epoch: ", epoch)
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
    'pattern': 3,
}

if __name__ == "__main__":
    main()