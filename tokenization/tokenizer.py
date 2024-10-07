import tiktoken
from pathlib import Path
from tiktoken.load import load_tiktoken_bpe


def gpt_tokenizer(tokenizer_path, special_tokens):
    
    # load the BPE model 
    token_dict = load_tiktoken_bpe(tokenizer_path)

    return tiktoken.Encoding(
        name='tokenizer.model', 
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        special_tokens={token: len(token_dict) + i for i, token in enumerate(special_tokens)},
        mergeable_ranks= token_dict, 
    )


def tokenizer():
    tokenizer = gpt_tokenizer(tokenizer_path='tokenization\\tokenizer.model', special_tokens=[
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)])

    return tokenizer


if __name__ == '__main__':

    tokenizer = tokenizer()
    print(tokenizer.encode('hello, AI'))
    print(tokenizer.decode([15339, 11, 15592]))