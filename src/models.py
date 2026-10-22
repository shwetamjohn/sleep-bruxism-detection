"""
Model Architecture Module for Sleep Bruxism Detection
Provides Large Selective Kernel Network (LSKNet) and ResNet baselines in pure PyTorch.
Official reference: 'LSKNet: Large Selective Kernel Network for Remote Sensing' (ICCV 2023)
"""

import math
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

# ==============================================================================
# Pure PyTorch LSKNet Architecture Implementation (No external mmcv needed)
# ==============================================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    """
    Large Selective Kernel Block:
    Combines multi-scale depthwise convolutions (5x5 and dilated 7x7)
    with spatial selection gating.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]
        attn = self.conv(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones((dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LSKNet(nn.Module):
    """
    LSKNet Backbone and Classification Network for CWT Scalograms.
    """
    def __init__(
        self,
        in_chans=3,
        num_classes=2,
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dims = embed_dims

        dpr = [float(v) for v in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i]
            )
            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j]
                )
                for j in range(depths[i])
            ])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dims[-1], num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x)
        return self.head(x)


def lsknet_tiny(num_classes=2, in_chans=3):
    """LSKNet-Tiny variant (fast, lightweight, ideal for biosignal scalograms)."""
    return LSKNet(
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dims=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.1,
        drop_path_rate=0.1
    )


def lsknet_small(num_classes=2, in_chans=3):
    """LSKNet-Small variant (higher capacity)."""
    return LSKNet(
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 4, 2],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.1,
        drop_path_rate=0.1
    )


def build_model(model_name="lsknet_tiny", num_classes=2, pretrained=True):
    """
    Builds the vision classification model.
    Supported models:
      - 'lsknet_tiny' or 'lsknet_t' (Native PyTorch LSKNet)
      - 'lsknet_small' or 'lsknet_s' (Native PyTorch LSKNet)
      - 'resnet18' (timm / torchvision ResNet-18)
    """
    clean_name = model_name.lower().replace("-", "_")

    if "lsk" in clean_name:
        if "s" in clean_name and "small" in clean_name:
            print(f"Instantiating native PyTorch LSKNet-Small (num_classes={num_classes})...")
            return lsknet_small(num_classes=num_classes)
        else:
            print(f"Instantiating native PyTorch LSKNet-Tiny (num_classes={num_classes})...")
            return lsknet_tiny(num_classes=num_classes)

    # ResNet or other standard architectures via timm
    if timm is not None:
        try:
            print(f"Instantiating {model_name} via timm (pretrained={pretrained}, num_classes={num_classes})...")
            return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        except Exception as e:
            print(f"Warning: timm could not load {model_name} ({e}). Falling back to resnet18.")
            return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)

    # Fallback to torchvision
    from torchvision import models
    print(f"Instantiating torchvision resnet18 (num_classes={num_classes})...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
