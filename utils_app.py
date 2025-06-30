import torch
import torchvision.transforms as transforms
import numpy as np
from training.utils import ChannelRepeat
from typing import Union

from training.models import UNet_conditional


class pipeline_prediction:
    def __init__(self, model):
        self.model = model

    def count_nan_values_percentage(self, img_raw: np.array):
        total_elements = img_raw.size
        nan_count = np.isnan(img_raw).sum()
        percentage = (nan_count / total_elements)
        return percentage

    def handle_missing_val(self, img_raw: np.array, mode:str = 'zero') -> Union[np.array, int]:
        if self.count_nan_values_percentage(img_raw) > 0.3:
                return -1
        if mode == 'zero':
            img_raw[np.isnan(img_raw)] = 0
        else:
            # Use OpenCV average pooling to fill NaNs
            mask = np.isnan(img_raw)
            img_filled = img_raw.copy()
            # Replace NaNs with zero temporarily for pooling
            img_filled[mask] = 0
            # Create a kernel for average pooling (e.g., 3x3)
            kernel_size = 3
            kernel = (kernel_size, kernel_size)
            # Count valid (non-NaN) neighbors
            valid_mask = (~mask).astype(np.float32)
            sum_pool = cv2.blur(img_filled, kernel)
            count_pool = cv2.blur(valid_mask, kernel)
            # Avoid division by zero
            avg_pool = np.divide(sum_pool, count_pool, out=np.zeros_like(sum_pool), where=count_pool!=0)
            # Fill NaNs with pooled average
            img_raw[mask] = avg_pool[mask]

        return img_raw 

    def preprocess_image(self, img_raw: np.array) -> torch.Tensor:
        self.model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            ChannelRepeat(), 
            self.model.get_transformation(),
        ])
        proc_img = transform(img_raw)
        return proc_img

    def make_predictions(self, proc_img: torch.Tensor) -> int:
        self.model.eval()
        with torch.no_grad():
            out = self.model(proc_img.unsqueeze(0))  # Add batch dimension
            return torch.argmax(out, dim=1).item()

def calculate_params(beta_start=1e-4, beta_end=0.02, noise_steps=400):
    beta = torch.linspace(beta_start, beta_end, noise_steps)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha, alpha_hat, beta

@torch.inference_mode()
def sample_ddpm(labels:int, n:int=1, cfg_scale:float=3, noise_steps:int=400, beta_start=1e-4, beta_end=0.02):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = torch.tensor(labels).long().to(device)
    checkpoint = torch.load('./checkpoints/DDPM_conditional/ddpm.pth', map_location=torch.device(device))
    model = UNet_conditional(num_classes=10).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    Alpha, Alpha_hat, Beta = calculate_params(beta_start, beta_end, noise_steps)
    with torch.inference_mode():
        x = torch.randn((n, 1, 32, 32)).to(device)
        for i in reversed(range(1, noise_steps)):
            t = (torch.ones(n) * i).long().to(device)
            predicted_noise = model(x, t, labels)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, t, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            
            alpha = Alpha[t][:, None, None, None]
            alpha_hat = Alpha_hat[t][:, None, None, None]
            beta = Beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8).squeeze()
    return x