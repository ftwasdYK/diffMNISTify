import torch
from torch import nn
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
        return self.weights.transforms() if self.weights else None
    
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
        self.model.heads = torch.nn.Linear(self.model.heads.in_features, num_classes)
    
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
                return self.layers(x).flatten().numpy()
        
__all__ = [CustomResNet50, CustomViT, FeatureExtractor]