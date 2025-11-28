import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, in_size, h_size, out_size, num_layers):
        super(model,self).__init__()
        # self.in_size = in_size
        # self.h_size = h_size
        # self.out_size = out_size
        # self.num_layers = num_layers
        self.rnn_layer = nn.LSTM(in_size, h_size,num_layers)    
        self.output_layer = nn.Linear(h_size, out_size)

    def forward(self, inputs):
        rnn_out, hidden = self.rnn_layer(inputs)
        x = self.output_layer(rnn_out)
        self.out = x
        return self.out
    
