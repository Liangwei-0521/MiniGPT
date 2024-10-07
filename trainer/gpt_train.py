import sys
sys.path.append('./')
import json 
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.net import Net
from src.model.embedding.token_embedding import Embedding
from src.model.embedding.position import PositionalEmbedding
from data import TextDataset, create_dataloader


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 加载训练集和测试集
train_data = load_data('./src/dataset/train.json')["train"]
embedd = Embedding(vocab_size=128000, dim=64)
position_emb = PositionalEmbedding(max_len=3, dim=64)


def GPT_Train():
    
    # 初始化模型
    gpt = Net()
    # 优化器
    optim = torch.optim.Adam(lr=1e-5, params=gpt.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # 导入语料
    train_dataloader = create_dataloader(train_data=train_data)
    n_epoches = 1

    while n_epoches < 100 :
        for index, context in enumerate(train_dataloader):
            # 词向量编码
            word_embedding = embedd(context[0])
            # 位置编码
            position_embedding = position_emb(embedd(context[0]))

            all_embedding = word_embedding + position_embedding

            next_word = gpt(all_embedding.to(device = 'cuda:0' if torch.cuda.is_available() else 'cpu'))
            # context[1]为标签，即目标词
            target = context[1].to(device = 'cuda:0' if torch.cuda.is_available() else 'cpu')
            # 计算损失
            loss = criterion(next_word, target.squeeze(dim=-1))
            # 清零梯度
            optim.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optim.step()
            # 每隔一定次数打印损失值
            print(f'Epoch [{n_epoches}], Step [{index}], Loss: {loss.item():.4f}')
        n_epoches = n_epoches + 1

    torch.save(gpt.state_dict(), './trainer/gpt_'+str(n_epoches)+'_.pth', _use_new_zipfile_serialization=False)



if __name__ == "__main__":

    GPT_Train()
