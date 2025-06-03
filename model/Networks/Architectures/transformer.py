from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed

import torch
from torch import nn


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
    ):

        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        output = src
        for layer in self.layers:
            output = layer(output, pos_emb, src_mask, src_key_padding_mask)
        return self.norm(output)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout
        )
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        if self.norm_first:
            src_norm = self.norm1(src)
            q = k = src_norm + pos_emb
            src = src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0])

            src_norm = self.norm2(src)
            src = src + self.dropout2(self.mlp(src_norm))
        else:
            q = k = src + pos_emb
            src = self.norm1(src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0]))

            src = self.norm2(src + self.dropout2(self.mlp(src)))

        return src

class TransformerDecoder(nn.Module):

    def __init__(
        self,
        #transformer arguments
        num_layers: int,
        norm: bool,
        #layer arguments
        emb_dim: int, # dmodel
        num_heads: int, # nhead
        dropout: float, # dropout
        layer_norm_eps: float,
        mlp_factor: int, # equivalent to dim_feed_forward => factor * emb_dim 
        norm_first: bool,
        activation: nn.Module, # activation
        deformable=False
        # two arguments less than deformableDecoder : num_feature_levels, dec_npoints => propre à deformable
        # num queries : number of k we want to keep, and thus the number of queries of the deformable attention
    ):

        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()
    
    def forward(
        self, tgt, memory, pos_emb, memory_pos_emb,memory_mask=None,
        memory_key_padding_mask=None
    ):

        output = tgt.flatten(2).permute(2, 0, 1) # 2304xbsx256
        memory = memory.permute(1,2,3,0,4).flatten(0,2) # bsx3x3x3x256 to bsx27x256 to 27xbsx256
        # memory is what we have in k and v
        outputs = list()
        for i, layer in enumerate(self.layers):
            output = layer(
                output, memory, pos_emb, memory_pos_emb, memory_mask, 
                memory_key_padding_mask
            )
            outputs.append(self.norm(output))

        return torch.stack(outputs)
    

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
    ):
        super(TransformerDecoderLayer, self).__init__()

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout
        )
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, memory, pos_emb, memory_pos_emb, memory_mask, memory_key_padding_mask
    ):
        if self.norm_first:
            tgt = self.norm1(tgt)
            tgt = tgt + self.dropout1(self.self_attn(
                query=self.with_emb(tgt,pos_emb),
                key=self.with_emb(memory,memory_pos_emb),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])

            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.mlp(tgt_norm))
        else:
            tgt = tgt + self.dropout1(self.self_attn(
                query=self.with_emb(tgt,pos_emb),
                key=self.with_emb(memory,memory_pos_emb),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])

            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.mlp(tgt_norm))

        return tgt
