import numpy as np
import torch
from torch.nn import BCELoss
from utils import bivariate_gausssain

def MDN_Loss(x,y, mux, muy, varx, vary, corr, w, e_pred, e_target):
    """params: 
        x: (B * L * 1)
        y: (B * L * 1)
        mux: (B * L * M)
        muy: (B * L * M)
        varx:(B * L * M)
        vary:(B * L * M)
        corr:(B * L * M)
        w :  (B * L * M)
        eos: (B * L * 1)
    output:
        loss : (B * 1 * 1)
    """
    prob = bivariate_gausssain(x,y,mux,muy,varx,vary,corr)      # (B * L * M)
    weighted_probs = w * prob                                   # (B * L * M)
    log_loss_t = torch.log(torch.sum(weighted_probs, axis = -1))# (B * L)
    log_loss_b = torch.sum(log_loss_t, axis = -1)               # (B)
    log_loss_prob = torch.mean(log_loss_b)                      # (scalar)
    log_loss_e = BCELoss(e_pred, e_target)                      # (scalar)
    nll = -log_loss_prob - log_loss_e

    return nll




