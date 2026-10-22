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

def build_model(model_name="lsknet_tiny", num_classes=2, pretrained=True):
    """
    Builds the vision classification model.
    Supported models:
      - 'lsknet_tiny' or 'lsknet_t'
      - 'lsknet_small' or 'lsknet_s'
      - 'resnet18'
    """
    # Map common aliases to official timm model names
    alias_map = {
        "lsknet_t": "lsknet_tiny",
        "lsknet_s": "lsknet_small",
        "lsk_t": "lsknet_tiny",
        "lsk_s": "lsknet_small",
    }
    target_name = alias_map.get(model_name.lower(), model_name)
    
    if timm is not None:
        # Check if model exists in timm
        available = timm.list_models(f"*{target_name}*")
        if not available and "lsk" in target_name:
            # Check for any LSK variants
            lsk_models = timm.list_models("*lsk*")
            if lsk_models:
                target_name = lsk_models[0]
                print(f"Using available LSK model in timm: {target_name}")
            else:
                target_name = "resnet18"
                print(f"Warning: LSKNet not found in this timm version. Using ResNet-18.")
        
        try:
            print(f"Instantiating {target_name} (pretrained={pretrained}, num_classes={num_classes})...")
            model = timm.create_model(
                target_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            return model
        except Exception as e:
            print(f"Warning: Could not create {target_name} ({e}). Falling back to resnet18.")
            return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
            
    raise RuntimeError("PyTorch and timm are required to build models.")
