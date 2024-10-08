# MiniGPT: 手把手教你实现

### Token
* tokenization : tokenizer.model

### **Embedding**
* word embedding : torch.nn.Embedding(vocab_size, features_dim)
* position embedding : Absolute position encoding

### **Decoder**
```python

Decoder(
    [
        DecoderLayer(
            MultiAttention_Layer(
                FullAttention(mask_flag=True,        # whether Mask is not applied
                               factor=12,               # the length of feature
                               attention_dropout=0.1,
                               output_attention=True    # whether to output the attention matrix
                               ),
                d_model=12,
                n_heads=4
            ),
            d_model=12,           # the length is equal to the length of feature, which is the input
            d_ff=128,             # the dimension of the feed-forward network
            dropout=0.1,
            activation='relu'
        ) for layer in range(4)   # the number of layers
    ],
    norm_layer=LayerNorm(12)
)

```

### **Data**
path: [text](src/dataset) <br>
example:
```python
{
    "test": [
      {"input": "They are discussing a", "label": "project"},
      {"input": "She is preparing for the", "label": "interview"},
      {"input": "He is waiting for the", "label": "bus"},
      ……
    ]
  }
  
```

### **Generate**


![image](https://github.com/user-attachments/assets/dc88c66a-368b-49cf-8020-7a7816683c49)

