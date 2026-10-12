
model_config = {
    "num_layers": 2,
    "num_heads": 4,
    "hidden_dim": 256,
    "context_len": 128,
    "dropout": 0.1,
    "ff_dim": 1024,
    "batch_size": 64,
    "epochs": 20,
    "lr": 0.0003,
    "clip_norm": 2.4
}

corpus = {
    "train" : {
        "en": "/srv/data/lt2326-h26/a1/train/en.txt", 
        "tr": "/srv/data/lt2326-h26/a1/train/tr.txt",
        "zh": "/srv/data/lt2326-h26/a1/train/zh.txt"
    },
    "valid" : {
        "en": "/srv/data/lt2326-h26/a1/valid/en.txt", 
        "tr": "/srv/data/lt2326-h26/a1/valid/tr.txt",
        "zh": "/srv/data/lt2326-h26/a1/valid/zh.txt"
    },
    "test" : {
        "en": "/srv/data/lt2326-h26/a1/test/en.txt", 
        "tr": "/srv/data/lt2326-h26/a1/test/tr.txt",
        "zh": "/srv/data/lt2326-h26/a1/test/zh.txt"
    } 
}

tok_prefix = {
    "char_level": "tokenizers/char_level_tok",
    "bpe_small": "tokenizers/bpe_small_tok",
    "bpe_large": "tokenizers/bpe_large_tok"
}
