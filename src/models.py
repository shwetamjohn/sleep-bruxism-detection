"""
Model Architecture Module for Sleep Bruxism Detection
Provides Large Selective Kernel Network (LSKNet) and ResNet baselines using timm.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import timm
except ImportError:
    timm = None

def build_model(model_name="lsknet_t", num_classes=2, pretrained=True):
    """
    Builds the vision classification model.
    
    Supported models:
      - 'lsknet_t' (LSKNet-Tiny): Lightweight, highly selective kernel mechanism.
      - 'lsknet_s' (LSKNet-Small): Standard LSKNet.
      - 'resnet18' : Standard baseline (as used in Syed Anna's reference work).
      
    Parameters:
        model_name (str): Architecture identifier.
        num_classes (int): 2 for binary classification (Bruxism vs Normal).
        pretrained (bool): Whether to initialize with ImageNet weights.
    """
    if timm is not None:
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            return model
        except Exception as e:
            print(f"Warning: Could not instantiate {model_name} from timm ({e}). Falling back to ResNet-18 or simple CNN.")
            if model_name != "resnet18":
                return build_model(model_name="resnet18", num_classes=num_classes, pretrained=pretrained)
                
    # Fallback Lightweight CNN (for systems without timm)
    if nn is not None:
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(128, num_classes)
                
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)
                
        return SimpleCNN(num_classes=num_classes)
        
    raise RuntimeError("PyTorch is required to build neural network models.")
