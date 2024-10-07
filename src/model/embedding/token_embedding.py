import sys
sys.path.append('./')
import torch
import torch.nn as nn
from tokenization.tokenizer import tokenizer


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.embedding(x)
    


if __name__ == '__main__':
    # 实例化词表编码器和token器
    tokenizer = tokenizer()
    embedd = Embedding(vocab_size=128000, dim=64)
    token_idx = tokenizer.encode('今年是新中国成立75周年。')

    token_idx = torch.tensor(token_idx)
    embedding_matrix = embedd(token_idx).clone().detach().unsqueeze(0)
    print('token id:{}, length:{}, embedding matrix:{}'.format(token_idx, len(token_idx), embedding_matrix.shape))

    from src.model.embedding.position import PositionalEmbedding
    p_emb = PositionalEmbedding(max_len=8, dim=64)
    p_ = p_emb(embedding_matrix)
    print('位置编码', p_.shape)



