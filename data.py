import json
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenization.tokenizer import tokenizer
from torch.utils.data import Dataset, DataLoader
from src.model.embedding.token_embedding import Embedding
from src.model.embedding.position import PositionalEmbedding


def collate_fn(batch):
    inputs, labels = zip(*batch)
    # 将输入和标签动态填充到批次中的最大长度
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=128001)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=128001)
    return inputs_padded, labels_padded


def pad_sequence(sentence, max_length, padding_value):
    if len(sentence) <= max_length:
        tmp = max_length - len(sentence)
        sentence.extend(tmp * [padding_value])  # 直接扩展列表
        return sentence  # 返回修改后的列表
    else:
        return sentence[:max_length]  # 截断列表
    
        

# 自定义Dataset类
class TextDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list): 输入数据，包含 'input' 和 'label' 的列表。
        """
        self.data = data
        self.tokenizer = tokenizer()

    def __len__(self):
        """返回数据集的总长度"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取数据样本。
        Args:
            idx (int): 索引
        Returns:
            dict: 包含输入和标签的字典
        """
        sample = self.data[idx]
       
        input_text = torch.tensor(pad_sequence(sentence=self.tokenizer.encode(sample['input']), 
                                               max_length=3, padding_value=128001))
        label_text = torch.tensor(pad_sequence(self.tokenizer.encode(sample['label']), 
                                               max_length=1, padding_value=128001))
        return input_text, label_text
    

def create_dataloader(train_data):
    train_dataset = TextDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return train_loader



# # 加载JSON文件
# def load_data(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return data

# # 加载训练集和测试集
# train_data = load_data('./src/dataset/train.json')["train"]
# test_data = load_data('./src/dataset/test.json')["test"]

# # 创建Dataset对象
# train_dataset = TextDataset(train_data)
# test_dataset = TextDataset(test_data)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# embedd = Embedding(vocab_size=128000, dim=64)
# p_emb = PositionalEmbedding(max_len=8, dim=64)

# for index, context in enumerate(train_loader):

#     print('index:', index, context, len(context[0]), context[0].shape, len(context[1]))
#     print(embedd(context[0]).shape)
#     p_ = p_emb(embedd(context[0]))
#     print(p_.shape,)
#     context_emb = p_emb(embedd(context[0])) + embedd(context[0])
#     print(context_emb.shape)

