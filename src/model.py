import torch
import segmentation_models_pytorch as smp

def get_model(encoder_name='efficientnet-b6', classes=12, weights='imagenet'):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=3,
        classes=classes
    )
    return model

def load_trained_model(path_model, device, encoder_name='efficientnet-b6', classes=12):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None, # Not needed since we're loading our own
        in_channels=3,
        classes=classes
    )
    state_dict = torch.load(path_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
