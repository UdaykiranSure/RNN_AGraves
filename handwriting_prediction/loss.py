import numpy as np
import math
import torch
from torch.nn import BCELoss
# from utils import bivariate_gausssainp
from handwriting_prediction.utils import bivariate_gaussian


def MDN_Loss(x,y,e_target,e_pred,w, mux, muy, stdx, stdy, corr):
    """params: 
        x: (B * L * 1)
        y: (B * L * 1)
        e_target: (B * L * 1)
        e_pred: (B * L * 1)
        w :  (B * L * M)
        mux: (B * L * M)
        muy: (B * L * M)
        stdx:(B * L * M)
        stdy:(B * L * M)
        corr:(B * L * M)
    output:
        loss : (B * 1 * 1)
    """
    prob = bivariate_gaussian(x,y,mux,muy,stdx,stdy,corr)      # (B * L * M)
    weighted_probs = w * prob                                   # (B * L * M)
    log_loss_t = torch.log(torch.sum(weighted_probs, dim = -1) + 1e-8) # (B * L)
    log_loss_b = torch.sum(log_loss_t, dim = -1)                # (B)
    log_loss_prob = torch.mean(log_loss_b)                      # (scalar)
    bce = BCELoss()
    bceloss = bce(e_pred, e_target)                      # (scalar)
    nll = -log_loss_prob - bceloss

    return nll


# torch.set_printoptions(precision=6)

# # Case A: random sanity check (shapes)
# B, L, M = 2, 5, 3
# x = torch.randn(B, L, 1)
# y = torch.randn(B, L, 1)
# # mixture params shapes (B, L, M)
# mux  = torch.randn(B, L, M)
# muy  = torch.randn(B, L, M)
# stdx = torch.abs(torch.randn(B, L, M)) + 1e-3
# stdy = torch.abs(torch.randn(B, L, M)) + 1e-3
# corr = torch.tanh(torch.randn(B, L, M) * 0.5)
# # mixture weights must be non-negative and sum to 1 over M
# w = torch.softmax(torch.randn(B, L, M), dim=-1)
# # end-of-stroke prediction shapes (B, L, 1)
# e_target = torch.randint(0, 2, (B, L, 1)).float()
# e_pred   = torch.sigmoid(torch.randn(B, L, 1))

# loss_val = MDN_Loss(x, y, e_target, e_pred, w, mux, muy, stdx, stdy, corr)
# print("Case A: random shapes -> loss:", loss_val.item())

# # Case B: deterministic single-mixture check (compute expected value)
# # B=1, L=1, M=1, x == mux, y == muy, stdx=stdy=1, corr=0
# B, L, M = 1, 1, 1
# x = torch.zeros(B, L, 1)
# y = torch.zeros(B, L, 1)
# mux  = torch.zeros(B, L, M)
# muy  = torch.zeros(B, L, M)
# stdx = torch.ones(B, L, M)
# stdy = torch.ones(B, L, M)
# corr = torch.zeros(B, L, M)
# w    = torch.ones(B, L, M)                 # single mixture -> weight 1
# e_target = torch.zeros(B, L, 1)            # target eos = 0
# e_pred   = torch.tensor([[[0.5]]])         # predicted eos = 0.5

# loss_val = MDN_Loss(x, y, e_target, e_pred, w, mux, muy, stdx, stdy, corr)
# print("Case B: deterministic -> loss:", loss_val.item())

# # compute expected analytic for Case B:
# # bivariate normal at zero with std=1 and corr=0 -> 1 / (2*pi)
# prob0 = 1.0 / (2.0 * math.pi)
# log_prob = math.log(prob0)   # ~ -1.837877
# bce = - ( (1-0.0) * math.log(1-0.5) )  # BCE for target=0, pred=0.5 -> -log(0.5)
# expected_nll = -log_prob - bce
# print("Case B: expected approx ->", expected_nll)
