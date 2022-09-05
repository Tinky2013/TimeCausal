import random
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F


class AllLoss():
    def __init__(self, PARAM):
        self.pro_reg = PARAM['pro_reg']

    def propensity_loss(self, pred0, pred1, posttime0, posttime1, pred_pro0, pred_pro1):
        assert len(pred0)==len(posttime0), "lenght not match!"
        assert len(pred1) == len(posttime1), "lenght not match!"
        preds = torch.cat([pred0, pred1],dim=0)  # (batch, posttime_len)
        posttime = torch.cat([posttime0, posttime1],dim=0)  # (batch, posttime_len)

        t0 = torch.zeros(pred_pro0.shape)
        t1 = torch.ones(pred_pro1.shape)

        return F.mse_loss(preds, posttime) + self.pro_reg * (F.mse_loss(pred_pro0, t0)+F.mse_loss(pred_pro1, t1))

