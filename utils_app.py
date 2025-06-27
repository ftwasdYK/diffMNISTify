import torch
import torchvision.transforms as transforms
import numpy as np
from training.utils import ChannelRepeat
from typing import Union

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