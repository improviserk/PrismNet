import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Dict, Tuple
import warnings
from xml.etree.ElementInclude import include

import torch
import torch.nn as nn
# from torchvision.ops.misc import Conv2dNormActivation
import torchvision
from torchvision.models.vision_transformer import Encoder
from .PrismNet import Conv2d

# input N C W H : N C 101 5
# output N 1

class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: Tuple[int],
        patch_size: Tuple[int],
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_config: int =  0 
    ):
        super().__init__()
        # _log_api_usage_once(self)
        # torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # if conv_stem_configs is not None:
        #     # As per https://arxiv.org/abs/2106.14881
        #     seq_proj = nn.Sequential()
        #     prev_channels = 1
        #     for i, conv_stem_layer_config in enumerate(conv_stem_configs):
        #         seq_proj.add_module(
        #             f"conv_bn_relu_{i}",
        #             conv_stem_layer_config,
        #         )
        #         prev_channels = 8
        #     seq_proj.add_module(
        #         "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        #     )
        #     self.conv_proj: nn.Module = seq_proj
        if conv_config:
            self.conv_proj = nn.Sequential()
            self.conv_proj.add_module(
                f'conv2d_1', nn.Conv2d(in_channels=1, out_channels=12,
                kernel_size=(3, 1), stride=(2, 1))
            )
            self.conv_proj.add_module('bn1', nn.BatchNorm2d(12))
            self.conv_proj.add_module('re1', nn.ReLU(inplace=True))

            self.conv_proj.add_module(
                f'conv2d_2', nn.Conv2d(in_channels=12, out_channels=24,
                kernel_size=(3, 1), stride=(2, 1))
            )
            self.conv_proj.add_module('bn2', nn.BatchNorm2d(24))
            self.conv_proj.add_module('re2', nn.ReLU(inplace=True))

            self.conv_proj.add_module(
                f'conv2d_3', nn.Conv2d(in_channels=24, out_channels=48,
                kernel_size=(3, 1), stride=1)
            )
            self.conv_proj.add_module('bn3', nn.BatchNorm2d(48))
            self.conv_proj.add_module('re3', nn.ReLU(inplace=True))

            self.conv_proj.add_module(
                f'conv2d_4', nn.Conv2d(in_channels=48, out_channels=96,
                kernel_size=(3, 1), stride=1)
            )
            self.conv_proj.add_module('bn4', nn.BatchNorm2d(96))
            self.conv_proj.add_module('re4', nn.ReLU(inplace=True))
            self.conv_proj.add_module(f'conv_last', nn.Conv2d(in_channels=96, out_channels=hidden_dim, kernel_size=1, stride=1))

        else:
            self.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:,:,:100,:]
        n, c, h, w = x.shape
        p = self.patch_size
        # torch._assert(h == self.image_size, "Wrong image height!")
        # torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p[0]
        n_w = w // p[1]

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # print(x.shape)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

def ViT_large()->VisionTransformer:
    model = VisionTransformer(
        image_size=(101, 5),
        patch_size=(5, 1),
        num_layers=12,
        num_heads=12,
        hidden_dim=768 ,
        mlp_dim=3072
    )
    return model

def ViT_medium()->VisionTransformer:
    model = VisionTransformer(
        image_size=(101, 5),
        patch_size=(5, 1),
        num_layers=8,
        num_heads=6,
        hidden_dim= 384,
        mlp_dim = 1536
    )
    return model

def ViT_small() -> VisionTransformer:
    model = VisionTransformer(
        image_size=(101, 5),
        patch_size=(5, 1),
        num_layers=5,
        num_heads=4,
        hidden_dim= 192,
        mlp_dim = 768
    )
    return model

def ViT_RBP_hybrid()->VisionTransformer:
    model = VisionTransformer(
        image_size=(101, 5),
        patch_size=(5, 1),
        num_layers=4,
        num_heads=4,
        hidden_dim=192 ,
        mlp_dim=768,
        conv_config = 1
    )
    return model