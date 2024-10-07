# MiniGPT: 手把手教你实现

Introduction

![1725980610601](image/README/GPT.png)

### Token

* tokenization : tokenizer.model

### **Embedding**
* word embedding : torch.nn.Embedding(vocab_size, features_dim)
* position embedding : Absolute position encoding

### **Decoder**
'''
    Decoder(
                [
                    DecoderLayer(
                        MultiAttention_Layer(
                            FullAttention(mask_flag=True,        # wherether Mask is not 
                                        factor=12,               # the length of feature 
                                        attention_dropout=0.1,   
                                        output_attention=True    # whether output the attention matrix
                                        ),                       
                            d_model=12,
                            n_heads=4       
                        ),
                        d_model=12,           # the length is equal of the length of feature, the input 
                        d_ff=128,             # the dimession of linear network
                        dropout=0.1,
                        activation='relu'
                    ) for layer in range(4)   # the number of layers
                ],
                norm_layer=torch.nn.LayerNorm(12)
            )
'''

### **Data**
[text](src/dataset)

### **Generate**

<video controls src="23d6012fb9a44ce2366ecc6a084b90ba(1).mp4" title="Title"></video>