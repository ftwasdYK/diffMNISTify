import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


class BaseNN(torch.nn.Module):
    """
    Base class for all neural network models.
    """
    def __init__(self):
        super(BaseNN, self).__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
    def get_transformation(self):
        """Returns the transformation weights if available."""
        return self.weights.transforms()
    
    def count_total_parameters(self) -> int:
        """Counts the total number of trainable parameters in the model."""
        return sum([i.numel() for i in self.model.parameters() if i.requires_grad])

    def load_weights(self, weights_path:str) -> None:
        """
        Load model weights from a specified path.
        
        Args:
            weights_path (str): Path to the model weights file.
        """
        self.model.load_state_dict(torch.load(weights_path)['model_state_dict'])
        print(f"Checkpoint loaded from {weights_path}")

class CustomResNet50(BaseNN):
    def __init__(self, num_classes:int=10, pretrained:bool=True):
        super(CustomResNet50, self).__init__()
        self.num_classes = num_classes
        self.weights = ResNet50_Weights.DEFAULT 
        self.model = resnet50(weights=None)
        if pretrained:
            self.model = resnet50(weights=self.weights)
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

class CustomViT(BaseNN):
    def __init__(self, num_classes:int=10, pretrained:bool=True):
        super(CustomViT, self).__init__()
        self.num_classes = num_classes
        self.weights = ViT_B_16_Weights.DEFAULT 
        self.model = vit_b_16(weights=None)
        if pretrained:
            self.model = vit_b_16(weights=self.weights)
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.heads = nn.Sequential(torch.nn.Linear(768, num_classes))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)


def calculate_avg_weights(weights) -> None:
    """
    Calculate the average weights of a conv layer.
    
    Args:
        weights (dict): Dictionary containing conv weights.
    
    Returns:
        dict: Dictionary with average weights.
    """
    avg_weights = {}
    for key, value in weights.items():
        if isinstance(value, torch.Tensor):
            avg_weights[key] = value.mean(dim=1, keepdim=True)  # Average across the channel dimension


class FeatureExtractor(BaseNN):
    """
    Feature extractor class that uses a pretrained model to extract features.
    """
    def __init__(self, model:nn.Module, num_layers:int=6):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.model.eval()
        self.layers = nn.Sequential(*nn.ModuleList(model.children())[:num_layers]) 
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.layers(x).flatten()
            


################# Diffusion models #################


from training.utils import get_data

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, n_feat_init=64, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, n_feat_init)
        self.down1 = Down(n_feat_init, n_feat_init*2) 
        self.sa1 = SelfAttention(n_feat_init*2)
        self.down2 = Down(n_feat_init*2, n_feat_init*4) 
        self.sa2 = SelfAttention(n_feat_init*4)
        self.down3 = Down(n_feat_init*4, n_feat_init*4)
        self.sa3 = SelfAttention(n_feat_init*4)


        if remove_deep_conv:
            self.bot1 = DoubleConv(n_feat_init*4, n_feat_init*4) # n_feat*4
            self.bot3 = DoubleConv(n_feat_init*4, n_feat_init*4)
        else:
            self.bot1 = DoubleConv(n_feat_init*4, n_feat_init*8) #  n_feat*4, n_feat*8
            self.bot2 = DoubleConv(n_feat_init*8, n_feat_init*8)
            self.bot3 = DoubleConv(n_feat_init*8, n_feat_init*4)

        self.up1 = Up(n_feat_init*8, n_feat_init*2)
        self.sa4 = SelfAttention(n_feat_init*2)
        self.up2 = Up(n_feat_init*4, n_feat_init)
        self.sa5 = SelfAttention(n_feat_init)
        self.up3 = Up(n_feat_init*2, n_feat_init)
        self.sa6 = SelfAttention(n_feat_init)
        self.outc = nn.Conv2d(n_feat_init, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    def __init__(self, c_in=1, c_out=1, time_dim=256, n_feat_init=64, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, n_feat_init, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)
    
__all__ = [CustomResNet50, CustomViT, FeatureExtractor, UNet_conditional]