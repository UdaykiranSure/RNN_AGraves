from dataloader import get_dataloader
from handwriting_prediction.model import model
from handwriting_prediction.loss import MDN_Loss
from handwriting_prediction.trainer import train
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM
import numpy as np
import torch
import math
data_dir = '/Users/udaykiran/ML/ResearchPapers/rnn_graves/data/lineStrokes'
batch_size = 32
tr_dataloader = get_dataloader(data_dir,batch_size)


# M = 20
# in_size = 3
# h_size = 400
# out_size = 6*M + 1
# num_layers = 3
# rnn = LSTM(in_size,h_size,num_layers,batch_first=True)
# batch,lens  = next(iter(tr_dataloader))
# packed = pack_padded_sequence(batch,lens,batch_first=True,enforce_sorted=False)
# rnn_out, hidden = rnn(packed)
# unpacked,p_lens = pad_packed_sequence(rnn_out)  
# if (torch.tensor(lens) != p_lens).all():
#     print('lens not match')
# torch.isfinite(unpacked).all()

M = 20


in_size = 3
h_size = 400
out_size = 6*M + 1
num_layers = 3
hp_model = model(in_size, h_size,out_size, num_layers)

criterion = MDN_Loss

lr = 0.01
epochs = 3

losses = train(hp_model,tr_dataloader,lr,epochs= epochs, criterion= criterion)
losses.shape

losses[:,:,:1]
for batch in losses:
    for seq in batch:
        if torch.isnan(seq[0]):
            print('not a number', seq)
        elif not torch.isfinite(seq[0]):
            print(seq)