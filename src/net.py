import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from src.decoder import Decoder
from src.model.layer.decoder_layer import DecoderLayer
from src.model.embedding.token_embedding import Embedding
from src.model.embedding.position import PositionalEmbedding
from src.model.attention.multi_head_attention import FullAttention, MultiAttention_Layer



class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiAttention_Layer(
                        FullAttention(mask_flag=True,          # wherether Mask is not 
                                      factor=64,               # the length of feature 
                                      attention_dropout=0.1,   
                                      output_attention=False    # whether output the attention matrix
                                      ),                       
                        d_model=64,
                        n_heads=4       
                    ),
                    d_model=64,           # the length is equal of the length of feature, the input 
                    d_ff=128,             # the dimession of linear network
                    dropout=0.1,
                    activation='relu'
                ) for layer in range(4)   # the number of layers
            ],
            norm_layer=torch.nn.LayerNorm(64)
        ) 

        self.linear = nn.Linear(3 * 64, 128000)


        self.to(device='cuda:0' if torch.cuda.is_available() else 'cpu')

    
    def forward(self, x):
        v = self.decoder(x)
        v = v[0].reshape(-1, 3  * 64)
        return self.linear(v)
    

if __name__ == '__main__':
    net = Net()
    x = torch.rand(size=(4,3,64)).cuda()
    result = net(x)
    print(result.shape)