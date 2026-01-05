import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class model(nn.Module):
    def __init__(self, in_size, h_size, out_size, num_layers):
        """
        params:
            in_size: input size       
            h_size: hidden state size 
            out_size: size of output, should be (6*M+1), M = no of mixtures
            num_layers: number of hidden layers in RNN( here LSTM)

        output:
            self.out: mixture model parameter (B * L * out_size)
        """
        super(model,self).__init__()
        self.rnn_layer = nn.LSTM(in_size, h_size,num_layers,batch_first=True) 

        for name,param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            if 'bias' in name:
                nn.init.zeros_(param)


        self.output_layer = nn.Linear(h_size, out_size)
        # nn.init.xavier_uniform_(self.output_layer.weight)
        # if self.output_layer.bias is not None:
        #     nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs,lengths):
        packed = pack_padded_sequence(inputs,lengths, batch_first=True,enforce_sorted=False)
        rnn_out, hidden = self.rnn_layer(packed) # (B * L * H), (B * H)
        seq_unpacked, lens_unpacked = pad_packed_sequence(rnn_out,batch_first=True)
        if (lens_unpacked != torch.tensor(lengths)).all():
            print('lengths not matched')
        if not torch.isfinite(seq_unpacked).all():
            print('nan in unpacked')
        x = self.output_layer(seq_unpacked) #output params order: (e, 1M(W) 2M(mu), 2M(std), 1M(corr)) => (1 + 6M) ;(B * L * (1+6M))
        if not torch.isfinite(x).all():
            print('nan in model output layer') 
        M = (x.shape[-1] - 1) // 6
        e =  torch.sigmoid(x[:,:,0:1]) # e = 1/1 + exp(e_hat) => (B * L* 1)
        W =  F.softmax(x[:,:,1:1+M], -1)          
        mux = x[:, :, M+1:2*M+1]
        muy = x[:, :, 2*M+1:3*M+1]
        stdx = torch.exp(torch.clamp(x[:, :, 3*M+1: 4*M+1], min= -10.0, max = 10.0)) 
        stdy = torch.exp(torch.clamp(x[:, :, 4*M+1: 5*M+1],  min= -10.0, max = 10.0))
        corr = torch.tanh(x[:, :, 5*M+1: 6*M + 1])
        self.out = torch.cat((e,W,mux,muy,stdx,stdy,corr),dim=-1)
        return self.out 

    
