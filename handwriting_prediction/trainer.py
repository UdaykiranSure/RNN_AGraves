import torch.nn as nn
import numpy as np
import torch
from dataloader import get_dataloader

def train(model, tr_dataloader,lr, epochs, criterion,M = 20,optim = 'sgd', device = 'cpu'):
    model.to(device=device)
    if optim  == 'rms':
        optimiser = torch.optim.RMSprop(model.parameters(),lr,alpha=0.95, eps =0.0001,centered=True, momentum=0.9)
    else:
        optimiser = torch.optim.SGD(model.parameters(), lr)

    losses = []
    model.train()
    for epoch in range(1, epochs+1):
        running_loss = []
        for batch_data,lengths in tr_dataloader:
                batch_data.to(device)
                optimiser.zero_grad()
                outputs,hidden  = model.forward(batch_data,lengths)  #(B * L * 6M+1)
                if not torch.isfinite(outputs).all():
                     print('nan from model in trainer')
                e_pred = outputs[:,:,:1]
                w = outputs[:,:,1:M+1]
                mux = outputs[:, :, M+1:2*M+1]
                muy = outputs[:, :, 2*M+1:3*M+1]
                stdx = outputs[:, :, 3*M+1: 4*M+1]
                stdy = outputs[:, :, 4*M+1: 5*M+1]
                corr = outputs[:, :, 5*M+1: 6*M + 1]
                # debug checks (print only on problem)
                # if not torch.isfinite(e_pred).all() or (e_pred.min() < 0) or (e_pred.max() > 1):
                #     print(f"[DEBUG] batch {batch_idx} epoch {epoch}: e_pred min={float(e_pred.min())} max={float(e_pred.max())} finite={torch.isfinite(e_pred).all().item()}")
                #     # optionally inspect a few values:
                #     print(e_pred.flatten()[:10])
                loss = criterion(batch_data[:,:,:1],batch_data[:,:,1:2],batch_data[:,:,2:],e_pred,w, mux, muy, stdx, stdy, corr)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                running_loss.append(loss.item())
                optimiser.step()
        model.eval()
        running_val_loss = []

        with torch.no_grad():
            for batch_data, lengths in val_dataloader:
                batch_data = batch_data.to(device)

                outputs, hidden = model(batch_data, lengths)

                e_pred = outputs[:, :, :1]
                w = outputs[:, :, 1 : M + 1]
                mux = outputs[:, :, M + 1 : 2 * M + 1]
                muy = outputs[:, :, 2 * M + 1 : 3 * M + 1]
                stdx = outputs[:, :, 3 * M + 1 : 4 * M + 1]
                stdy = outputs[:, :, 4 * M + 1 : 5 * M + 1]
                corr = outputs[:, :, 5 * M + 1 : 6 * M + 1]

                val_loss = criterion(
                    batch_data[:, :, :1],
                    batch_data[:, :, 1:2],
                    batch_data[:, :, 2:],
                    e_pred,
                    w,
                    mux,
                    muy,
                    stdx,
                    stdy,
                    corr,
                )

                running_val_loss.append(val_loss.item())

        val_loss = np.mean(running_val_loss)
        val_losses.append(val_loss)

        print(
            f"epoch: {epoch}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )
        losses.append(running_loss)
    return losses


