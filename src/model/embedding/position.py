import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim).float()
        pe.requires_grad = False  

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 批次维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



if __name__ == '__main__':

    position_embedding = PositionalEmbedding(d_model=64, max_len=256)
    position_embedding_ = position_embedding(x = torch.rand(size=(1, 256, 64)))


    print(position_embedding_.shape)