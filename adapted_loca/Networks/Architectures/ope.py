from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed

import torch
from torch import nn

from torchvision.ops import roi_align


class OPEModule(nn.Module):

    def __init__(
        self,
        num_iterative_steps: int,
        emb_dim: int,
        kernel_dim: int,
        num_heads: int,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        scale_only:bool,
        scale_as_key:bool
    ):

        super(OPEModule, self).__init__()

        self.num_iterative_steps = num_iterative_steps
        self.kernel_dim = kernel_dim
        #self.num_objects = num_objects
        self.emb_dim = emb_dim
        #self.reduction = reduction

        if num_iterative_steps > 0:
            self.iterative_adaptation = IterativeAdaptationModule(
                num_layers=num_iterative_steps, emb_dim=emb_dim, num_heads=num_heads,
                dropout=0, layer_norm_eps=layer_norm_eps,
                mlp_factor=mlp_factor, norm_first=norm_first,
                activation=activation, norm=norm, scale_only=scale_only,
                scale_as_key=scale_as_key
            )
        
        self.shape_or_objectness = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, self.kernel_dim**2 * emb_dim)
        )
        
    def forward(self, f_e, references, pos_emb, query_pos_emb, bboxes):
        bs, _, _, _ = f_e.size()
        # extract the shape features or objectness
        memory = f_e.flatten(2).permute(2, 0, 1) # 2304xbsx256
        query = references.permute(1,2,3,0,4).flatten(0,2) # bsx3x3x3x256 to bsx27x256 to 27xbsx256
        
        box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
        box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0] #width
        box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1] #height

        shape_or_objectness = self.shape_or_objectness(box_hw).reshape(
            bs, -1, self.kernel_dim ** 2, self.emb_dim
        ).flatten(1, 2).transpose(0, 1)
    
        memory = f_e.flatten(2).permute(2, 0, 1) # 2304xbsx256
        all_prototypes = self.iterative_adaptation(
            shape_or_objectness, query, memory, pos_emb, query_pos_emb
        )

        return all_prototypes


class IterativeAdaptationModule(nn.Module):

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
        scale_only:bool,
        scale_as_key:bool
    ):

        super(IterativeAdaptationModule, self).__init__()
        self.scale_only = scale_only

        if scale_only:
            self.layers = nn.ModuleList([
                IterativeAdaptationLayerScaleOnly(
                    emb_dim, num_heads, dropout, layer_norm_eps,
                    mlp_factor, norm_first, activation, scale_as_key
                ) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                IterativeAdaptationLayer(
                    emb_dim, num_heads, dropout, layer_norm_eps,
                    mlp_factor, norm_first, activation, scale_as_key
                ) for _ in range(num_layers)
            ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):

        output = tgt
        outputs = list()
        for i, layer in enumerate(self.layers):
            if self.scale_only:
                output = layer(
                    output, appearance, query_pos_emb, tgt_mask,
                    tgt_key_padding_mask
                )
            else:
                output = layer(
                    output, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask
                )
            outputs.append(self.norm(output))

        return torch.stack(outputs)


class IterativeAdaptationLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        scale_as_key: bool
    ):
        super(IterativeAdaptationLayer, self).__init__()

        self.norm_first = norm_first
        self.scale_as_key = scale_as_key

        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask
    ):
        if self.norm_first:
            tgt_norm = self.norm1(tgt)
            if self.scale_as_key:
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(appearance, query_pos_emb),
                    key=self.with_emb(tgt_norm, query_pos_emb),
                    value=tgt_norm,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])
            else:
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]) 

            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt_norm, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])
            tgt_norm = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.mlp(tgt_norm))

        else:
            
            if self.scale_as_key:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(appearance, query_pos_emb),
                    key=self.with_emb(tgt, query_pos_emb),
                    value=tgt,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))
            else:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]))

            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        return tgt
    
class IterativeAdaptationLayerScaleOnly(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        scale_as_key: bool
    ):
        super(IterativeAdaptationLayerScaleOnly, self).__init__()

        self.norm_first = norm_first
        self.scale_as_key = scale_as_key
        
        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance,query_pos_emb, tgt_mask,
        tgt_key_padding_mask
    ):
        if self.norm_first:

            if self.scale_as_key:
                # juste exchange query with key and values
                tgt_norm = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(appearance, query_pos_emb),
                    key=self.with_emb(tgt_norm, query_pos_emb),
                    value=tgt_norm,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])

                tgt = tgt + self.dropout2(self.mlp(self.norm2(tgt)))

            else:
                
                tgt_norm = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])

                tgt = tgt + self.dropout2(self.mlp(self.norm2(tgt)))

        else:

            if self.scale_as_key:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(appearance, query_pos_emb),
                    key=self.with_emb(tgt, query_pos_emb),
                    value=tgt,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

                tgt = self.norm2(tgt + self.dropout2(self.mlp(tgt)))

            else:
            
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

                tgt = self.norm2(tgt + self.dropout2(self.mlp(tgt)))

        return tgt
    
class IterativeAdaptationLayerFeatureEnhancementOnly(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        scale_as_key: bool
    ):
        super(IterativeAdaptationLayerFeatureEnhancementOnly, self).__init__()

        self.norm_first = norm_first
        self.scale_as_key = scale_as_key
        
        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance,query_pos_emb, tgt_mask,
        tgt_key_padding_mask
    ):
        if self.norm_first:

            if self.scale_as_key:
                # juste exchange query with key and values
                tgt_norm = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(appearance, query_pos_emb),
                    key=self.with_emb(tgt_norm, query_pos_emb),
                    value=tgt_norm,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])

                tgt = tgt + self.dropout2(self.mlp(self.norm2(tgt)))

            else:
                
                tgt_norm = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])

                tgt = tgt + self.dropout2(self.mlp(self.norm2(tgt)))

        else:

            if self.scale_as_key:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(appearance, query_pos_emb),
                    key=self.with_emb(tgt, query_pos_emb),
                    value=tgt,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

                tgt = self.norm2(tgt + self.dropout2(self.mlp(tgt)))

            else:
            
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

                tgt = self.norm2(tgt + self.dropout2(self.mlp(tgt)))

        return tgt
    