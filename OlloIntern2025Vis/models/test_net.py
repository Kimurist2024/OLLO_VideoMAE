import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class EfficientAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.dropout_p = attn_drop
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv

        hidden_states = F.scaled_dot_product_attention(
            query * self.scale,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout_p,
            is_causal=False,
        )

        x = hidden_states.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 400,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        k=128,
        num_heads=8,
        one_kv_head=False,
        share_kv=False,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        qkv_bias=None,
        qk_scale=None,
    ):
        super().__init__()
        assert (
            dim % num_heads
        ) == 0, "dimension must be divisible by the number of heads"

        self.seq_len = seq_len
        self.k = k

        self.num_heads = num_heads

        # dim_head = default(dim_head, dim // num_heads)
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim

        self.scale = qk_scale or head_dim**-0.5

        self.head_dim = head_dim
        all_head_dim = head_dim * self.num_heads

        # self.to_q = nn.Linear(dim, head_dim * num_heads, bias=False)

        # self.to_k = nn.Linear(dim, kv_dim, bias=False)

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            # self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout_p = attn_drop
        self.attn_dropout = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def normal_forward(self, x, context=None, **kwargs):
        b, n, _, d_h, h, k = *x.shape, self.head_dim, self.num_heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert (
            kv_len == self.seq_len
        ), f"the sequence length of the key / values must be {self.seq_len} - {kv_len} given"

        queries = self.to_q(x)
        # print("query shape=", queries.shape)

        # proj_seq_len = lambda args: torch.einsum("bnd,nk->bkd", *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        # print("kv_input:", kv_input.shape)
        # print("keys, values:", keys.shape, values.shape)

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(
            lambda args: torch.einsum("bnd,nk->bkd", *args),
            zip((keys, values), kv_projs),
        )
        # print("projected keys, values:", keys.shape, values.shape)

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)
        # print("query before attn shape=", queries.shape)

        merge_key_values = (
            lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        )
        keys, values = map(merge_key_values, (keys, values))

        # attention
        # print("projected before attn keys, values:", keys.shape, values.shape)

        dots = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * self.scale
        # print("dots shape=", dots.shape)
        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)

        # print("out shape=", out.shape)
        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        # print("out transpose shape=", out.shape)
        out = self.to_out(out)
        # print("final out shape=", out.shape)

        out = self.proj_drop(out)

        return out

    def efficient_forward(self, x, context=None, **kwargs):
        b, n, _, _, _, _ = *x.shape, self.head_dim, self.num_heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert (
            kv_len == self.seq_len
        ), f"the sequence length of the key / values must be {self.seq_len} - {kv_len} given"

        # queries = self.to_q(x)
        # print("query shape=", queries.shape)

        # keys = self.to_k(x)
        # values = self.to_v(x) if not self.share_kv else keys

        qkv = self.qkv(x)
        # print("x shape after qkv =", qkv.shape, "num heads=", self.num_heads)
        qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        queries, keys, values = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # print("kv_input:", kv_input.shape)
        # queries = self.to_q(x)
        # print("query shape=", queries.shape)

        # proj_seq_len = lambda args: torch.einsum("bnd,nk->bkd", *args)

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
        # print("before matmul:", keys.shape)
        # keys = keys.transpose(2, 3) @ kv_projs[0]
        # print("after matmul:", keys.shape)
        # keys = keys.transpose(2, 3)
        # values = values.transpose(2, 3) @ kv_projs[0]
        # values = values.transpose(2, 3)

        # project keys and values along the sequence length dimension to k

        keys, values = map(
            lambda args: torch.einsum("bhnd,nk->bhkd", *args),
            zip((keys, values), kv_projs),
        )
        # print("projected keys, values:", keys.shape, values.shape)

        # merge head into batch for queries and key / values

        # queries = queries.reshape(b, n, h, -1).transpose(1, 2)
        # print("query before attn shape=", queries.shape)

        # merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        # keys, values = map(merge_key_values, (keys, values))

        keys = keys.contiguous()
        values = values.contiguous()

        # attention
        # print("projected before attn keys, values:", keys.shape, values.shape)

        # dots = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * self.scale
        # # print("dots shape=", dots.shape)
        # attn = dots.softmax(dim=-1)
        # attn = self.attn_dropout(attn)
        # out = torch.einsum("bhnk,bhkd->bhnd", attn, values)
        out = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=self.dropout_p,
            is_causal=False,
        )

        # print("out shape=", out.shape)
        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        # print("out transpose shape=", out.shape)
        out = self.to_out(out)
        # print("final out shape=", out.shape)

        out = self.proj_drop(out)

        return out

    def forward(self, x, context=None, **kwargs):
        # out = self.normal_forward(x, context, **kwargs)
        out = self.efficient_forward(x, context, **kwargs)
        # print()

        # diff = torch.abs(out1 - out2)
        # print("diff:", diff.mean(), diff.max(), diff.min())

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dropout_p = attn_drop
        # print("dropout_p:", self.dropout_p, "qk_scale:", qk_scale)

    def normal_forward(self, x):
        B, N, C = x.shape
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat(
        #         (
        #             self.q_bias,
        #             torch.zeros_like(self.v_bias, requires_grad=False),
        #             self.v_bias,
        #         )
        #     )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # print("x shape befoer qkv =", x.shape)
        qkv = self.qkv(x)
        # print("x shape after qkv =", qkv.shape, "num heads=", self.num_heads)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).contiguous()
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        # print(qkv.shape)
        q = qkv[(0,)]
        k = qkv[(1,)]
        v = qkv[(2,)]

        # q, k, v = (
        #     qkv[0],
        #     qkv[1],
        #     qkv[2],
        # )  # make torchscript happy (cannot use tensor as tuple)
        # query, key, value = torch.split(qkv, 1)
        # print(query.shape, key.shape, value.shape)
        # q = query.squeeze(0)
        # k = key.squeeze(0)
        # v = value.squeeze(0)
        # print("q,k,v shape=", q.shape, k.shape, v.shape)

        # v_test = v.transpose(1, 2).reshape(B, N, -1) + orig_x

        # return v_test

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()
        # print("attn shape=", attn.shape)

        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # print("attn:", attn.shape, "v shape=", v.shape)

        x = (attn @ v).transpose(1, 2).contiguous()
        x = x.reshape(B, N, -1).contiguous()
        # print(x.shape, orig_x.shape)
        # x = x + orig_x

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def efficient_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = torch.split(qkv, 1)
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)
        # query, key, value = (
        #     qkv[0],
        #     qkv[1],
        #     qkv[2],
        # )  # make torchscript happy (cannot use tensor as tuple)

        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0

        # print("query shape:", query.shape)
        # print("key shape:", key.shape)
        # print("value shape:", value.shape)

        # if not self.training:
        #     query = query.half()
        #     key = key.half()
        #     value = value.half()

        hidden_states = F.scaled_dot_product_attention(
            query,
            # query,
            key,
            # key,
            value,
            # value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )

        x = hidden_states.transpose(1, 2).reshape(B, N, -1)

        # if not self.training:
        # x = x.float()
        # x = hidden_states.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward(self, x):
        # out = self.normal_forward(x)
        out = self.efficient_forward(x)

        # diff = torch.abs(out1 - out2)
        # print("diff:", diff.mean(), diff.max(), diff.min())
        # dsfa

        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        attn_module=Attention,
        seq_len=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # print("attn_module:", attn_module, attn_module.__class__.__name__)
        # dsfa
        # if isinstance(attn_module, LinformerSelfAttention):
        if attn_module is LinformerSelfAttention:
            self.attn = attn_module(
                dim,
                seq_len=seq_len,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )

        else:
            self.attn = attn_module(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            h = self.norm1(x)
            h = self.attn(h)
            h = self.drop_path(h)

            x = x + h
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        # self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        # print("before proj:", x.shape)
        x = self.proj(x)
        B, C, T, H, W = x.shape
        # print("after proj:", x.shape)
        x = x.flatten(2)
        # print("after flatten:", x.shape)
        x = x.transpose(1, 2)
        return x, (B, C, T, H, W)


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class Head(nn.Module):
    def __init__(self, input_dim=768, output_dim=256, hidden_dims=[]):
        super().__init__()
        if len(hidden_dims):
            prev_dim = input_dim
            layers = []
            for hidden_dim in hidden_dims[:-1]:
                layer = nn.Linear(prev_dim, hidden_dim)
                layers.append(layer)
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim

            hidden_dim = hidden_dims[-1]
            layer = nn.Linear(prev_dim, output_dim)
            layers.append(layer)

            layers.append(nn.BatchNorm1d(output_dim))

        else:
            layers = [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim)]
            # layers.append(nn.ReLU())

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.head(x)
        return logits


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        head_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        init_scale=0.0,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        with_cp=False,
        num_segment=1,
        drop_head=False,
        add_projection=False,
        seq_len=None,
        attn_module=Attention,
        is_output_upscaling=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=self.tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp
        self.num_segment = num_segment

        self.is_output_upscaling = is_output_upscaling

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            # if img_size != 224 or all_frames != 16:
            #     org_img_size = (224, 224)
            #     org_num_frames = 16
            #     # org_img_size = (img_size, img_size)
            #     # org_num_frames = all_frames
            #     num_patches = (
            #         (org_img_size[1] // patch_size)
            #         * (org_img_size[0] // patch_size)
            #         * (org_num_frames // self.tubelet_size)
            #     )
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    attn_module=attn_module,
                    seq_len=seq_len,
                )
                for i in range(depth)
            ]
        )

        # self.attn_blocks = [block for block in self.blocks]

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        self.drop_head = drop_head
        if not drop_head:
            self.head_dropout = nn.Dropout(head_drop_rate)
            self.head = (
                nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        if not drop_head and num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        if self.is_output_upscaling and tubelet_size > 1:
            self.upscaling = nn.Upsample(scale_factor=tubelet_size, mode="linear")

        else:
            self.upscaling = None

        if add_projection:
            self.projection = Head(input_dim=embed_dim, output_dim=256, hidden_dims=[])

        else:
            self.projection = nn.Identity()

        self.add_projection = add_projection

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_with_mask(self, x, mask):
        _, _, T, _, _ = x.shape
        x, size = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks:
            if self.with_cp:
                x_vis = cp.checkpoint(blk, x_vis)
            else:
                x_vis = blk(x_vis)

        x_vis = self.fc_norm(x_vis)
        return x_vis

    def forward_features(self, x):
        # print("before  patch embed:", x.shape)
        # if self.training:
        # x.requires_grad_(True)
        x, (B, C, T, H, W) = self.patch_embed(x)
        # print("after patch embed:", x.shape)
        B, _, _ = x.size()

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            if self.training:
                x = (
                    x
                    + self.pos_embed.expand(B, -1, -1)
                    .type_as(x)
                    .to(x.device)
                    .clone()
                    .detach()
                )
            else:
                if self.pos_embed.device != x.device:
                    self.pos_embed = self.pos_embed.to(x.device)
                x = x + self.pos_embed.expand(B, -1, -1)

        # else:
        # x = x + self.pos_embed.expand(B, -1, -1)
        x = self.pos_drop(x)

        num_layer = 0

        # if self.with_cp:
        # x = cp.checkpoint_sequential(functions=self.blocks, segments=3, input=x)

        # else:

        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

            num_layer += 1
            # print("after layer {}, {}".format(num_layer, x.shape))

        x = self.norm(x)

        # print("after final norm shape = {}".format(x.shape))
        if self.fc_norm is not None:
            # return self.fc_norm(x[:, 1:].mean(1))
            out = self.fc_norm(x.mean(1))
            # print("out shape = {}".format(out.shape))
            return out
        else:
            x = x.mean(1)
            return x

    def forward_check_variance(self, x):
        x = self.patch_embed(x)

        B, _, _ = x.size()

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = (
                x
                + self.pos_embed.expand(B, -1, -1)
                .type_as(x)
                .to(x.device)
                .clone()
                .detach()
            )
        x = self.pos_drop(x)

        # x [B, N, C]
        avg_var_list = []
        for blk in self.blocks:
            x = blk(x)
            avg_var = torch.mean(torch.var(x, dim=-1))
            avg_var_list.append(avg_var)

        for i, avg_var in enumerate(avg_var_list):
            print(f"avg variance of block {i}: {avg_var}", flush=True)

        x = self.norm(x)
        if self.fc_norm is not None:
            # return self.fc_norm(x[:, 1:].mean(1))
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(
        self,
        x,
    ):
        # x = self.forward_check_variance(x)
        # if not self.training:
        # h = self.forward_features(x, is_mean=True)
        # return h

        # print(x.shape)
        x = self.forward_features(x)

        if self.drop_head or self.head is None:
            return x

        feats = x
        x = self.head_dropout(feats)
        x = self.head(x)
        if self.num_segment > 1 and self.training:
            x = x.view((-1, self.num_segment) + x.size()[1:])
            x = x.mean(dim=1)

        return x


class VisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2,
        with_cp=False,
        with_fp16=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size**2
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_size = patch_size
        self.with_cp = with_cp
        self.with_fp16 = with_fp16

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, return_token_num):
        # print("decoder input x shape =", x.shape)
        with torch.cuda.amp.autocast_mode.autocast(enabled=self.with_fp16):
            for depth, blk in enumerate(self.blocks):
                if self.with_cp:
                    x = cp.checkpoint(blk, x)
                else:
                    x = blk(x)
                    # print("x shape after {} th decoder attn:".format(depth), x.shape)

            if return_token_num > 0:
                # only return the mask tokens predict pixels
                x = self.head(self.norm(x[:, -return_token_num:]))
            else:
                # [B, N, 3*16^2]
                x = self.head(self.norm(x))
        return x


class VisionTransformerMAE(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        tubelet_size=2,
        num_classes=0,
        in_chans=0,
        with_cp=False,
        embed_dim=768,
        depth=12,
        num_heads=12,
        head_drop_rate=0.0,
        init_scale=0.0,
        all_frames=16,
        use_mean_pooling=True,
        num_segment=1,
        drop_head=False,
    ):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            with_cp=with_cp,
            # use_mean_pooling=False,
        )

        self.decoder = VisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            with_cp=with_cp,
            with_fp16=True,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.drop_head = drop_head

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        trunc_normal_(self.mask_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(
        self,
        x,
        mask,
    ):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder.forward_with_mask(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N_vis, C = x_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1
        )  # [B, N, C_d]
        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        return x


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    if "pretrained_cfg" in kwargs:
        kwargs.pop("pretrained_cfg")
        kwargs.pop("pretrained_cfg_overlay")
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_384(pretrained=False, **kwargs):
    if "img_size" in kwargs:
        kwargs["img_size"] = 384
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_giant_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1664,
        depth=48,
        num_heads=16,
        mlp_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def mae_vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformerMAE(
        # img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # decoder_num_classes=768,
        decoder_embed_dim=384,  # decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
