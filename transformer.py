import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor


def create_causal_mask(seq_len: int, device):
    """
    Mask the upper right-hand triangle in the matrix (the future tokens)
    -inf = masked
    0 = unmasked

    :seq_len: len of context window
    return: masked tensor
    """
    # initialize matrix with all masked values
    start_matrix = torch.full((seq_len, seq_len), float("-inf"), device=device)
    # change triangle to 0's
    mask = torch.triu(start_matrix, diagonal=1)
    return mask
    
class MultiLangDataset(torch.utils.data.Dataset):
    """
    This class handles turning text from the corpus into PyTorch tensors and slices that the 
    batches can handle.
    """
    def __init__(self, sents, tokenizer, seq_len=128):
        self.seq_len = seq_len
        # turn the corpus text into one continuous list of tokenized ids
        # this will later be sliced into context windows of size seq_len
        all_ids = []
        for sent in tokenizer.encode(sents):
            # add bos and eos to each sentence encoding
            new_sent = [tokenizer.bos_id()] + sent + [tokenizer.eos_id()]
            all_ids.extend(new_sent)
        # turn id list into tensor
        self.encoded = torch.tensor(all_ids)

    def __len__(self):
        """
        return: number of context windows in the dataset
        """
        return (len(self.encoded) -1) // self.seq_len

    def __getitem__(self, idx):
        """
        :idx: index of interest
        return: feature and label vector at the given index
        """
        # multiply by context_window size to get to the right start idx in the "complete tensor"
        start_idx = idx * self.seq_len 
        # stop idx is 1 space after the end of context window 
        stop_idx = start_idx + self.seq_len + 1 
        window = self.encoded[start_idx:stop_idx]

        X = window[:-1]
        y = window[1:]
        return X,y

class DecoderLayer(nn.Module):
    """
    This is the class managing the decoder layer of the Transformer.
    It uses :
    - Multi-head attention for the masked self-attention mechanism
    - SwiGLU activation function for feed-forward network
    - RMSNorm for normalization at the input of attention and feed-forward network 
    """
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        """
        :hidden_dim: dimension of hidden vector
        :num_heads: number of attention heads
        :ff_dim: feed-forward dimension
        :dropout: dropout probability
        """
        super().__init__()
        self.self_attn = MultiHeadAttn(hidden_dim, num_heads, dropout)
        self.ffn = SwiGLU(hidden_dim, ff_dim)
        # set up two normalizations (before attention block and before feed-forward)
        self.norm1 = nn.RMSNorm(hidden_dim)  # RMSNorm as used in Jurafsky
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, mask=None):
        # self attention
        out = self.norm1(X)
        out = self.self_attn(out, out, out, mask)
        X = X + self.dropout(out)

        # feed-forward
        out = self.norm2(X)
        out = self.ffn(out)

        return X + self.dropout(out)

class TransformerModel(nn.Module):
    """
    Decoder-only Transformer language model.
    
    """
    def __init__(self, num_layers, num_heads, hidden_dim, seq_len, vocab_size, ff_dim, device, dropout=0.1):
        """
        :num_layers: number of decoder layers
        :num_heads: number of attention heads
        :hidden_dim: dimension of hidden vector
        :seq_len: context window length
        :vocab_size: number of tokens in vocabulary
        :ff_dim: feed-forward network dimension
        :device: device that model runs on (gpu)
        :dropout: dropout probability (consistent throughout model)
        """
        super().__init__()
        # lookup table for token vector in vocab
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # lookup table for vector per position in context window
        self.position_embedding = nn.Embedding(seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoders = nn.ModuleList()
        self.seq_len = seq_len
        
        # loop to create the number of decoder layers in config
        for _ in range(num_layers):
            self.decoders.append(DecoderLayer(hidden_dim, num_heads, ff_dim, dropout))
            
        self.norm = nn.RMSNorm(hidden_dim)
        # output logits (score) for each token in the vocab
        self.out = nn.Linear(hidden_dim, vocab_size)

        self.device = device

    def forward(self, ids, mask=None):
        """
        :ids: tensor of token ids (shape: batch_size, len)
        :mask: causal mask 

        return: score for every possible next token (shape: batch_size, seq_len, vocab_size)
        """
        # calculate position embedding
        # (use the actual ids tensor len as stop value instead of seq_len
        # because last window might be shorter than seq_len) 
        positions = torch.arange(ids.shape[1], device=self.device)

        # each token's vector is the token embedding + position embedding
        X = self.embedding(ids) + self.position_embedding(positions)
        X = self.dropout(X)
        
        for decoder in self.decoders:
            X = decoder(X, mask)
            
        X = self.norm(X)
        
        return self.out(X)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function for feed-forward network.
    
    """
    def __init__(self, hidden_dim, interm_dim):
        """
        :hidden_dim: size of hidden vector
        :interm_dim: size of bigger intermediate vectors
        """
        super().__init__()
        self.gate = nn.Linear(hidden_dim, interm_dim)
        self.up = nn.Linear(hidden_dim, interm_dim)
        self.down = nn.Linear(interm_dim, hidden_dim)
        self.activation = nn.SiLU()
    
    def forward(self, X):
        # pass gate vector through activation function
        new_gate_vec = self.activation(self.gate(X))
        X =  new_gate_vec * self.up(X)
        # shrink back
        X = self.down(X)
        
        return X


class MultiHeadAttn(nn.Module):
    """
    Masked multi-head self-attention.
    """
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = floor(hidden_dim/num_heads)
        self.dropout = dropout
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, hidden_dim = q.shape

        #project
        queries = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        keys = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        values = self.v_proj(v).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # use dropout in dot product only if training
        if self.training:
            dropout_prob = self.dropout
        else:
            dropout_prob = 0.0

        out = F.scaled_dot_product_attention(queries,
                                               keys,
                                               values,
                                               attn_mask=mask,
                                               dropout_p=dropout_prob
                                               )
        # merge info from the heads
        out = out.transpose(1,2).reshape(batch_size, seq_len, hidden_dim)
        out = self.out_proj(out)
        
        return out