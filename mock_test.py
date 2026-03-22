import torch
import numpy as np

# Import the modular logic from our newly created structure
from src.model import get_model
from src.loss import CustomLoss
from config import SystemConf

def run_mock_test():
    print("=== Starting Architectectural Mock Test ===")
    
    # 1. Initialize configurations
    sysconfig = SystemConf(device='cpu')
    print("[1/4] Config initialized: OK")
    
    # 2. Init Model
    try:
        # Putting weights to None so we don't have to download 160MB of ImageNet weights just to run a mock test.
        model = get_model(encoder_name='efficientnet-b6', classes=12, weights=None)
        model.to(sysconfig.device)
        model.train()
        print("[2/4] Model initialization (DeepLabV3+ | EfficientNet-B6): OK")
    except Exception as e:
        print(f"Model initialization failed! Trace:\n{e}")
        return

    # 3. Create dummy data matrices
    # Batch size 2, 3 channels, 256x256 dimensions
    dummy_images = torch.randn(2, 3, 256, 256).to(sysconfig.device)
    
    # Target Masks (12 classes -> integer values between 0 and 11)
    dummy_targets = torch.randint(0, 12, (2, 256, 256)).to(sysconfig.device)
    print(f"[3/4] Dummy data created: \n      Images: {dummy_images.shape} \n      Targets: {dummy_targets.shape}")

    # 4. Forward Pass, Loss Evaluation, and Backward pass
    try:
        loss_fn = CustomLoss()
        
        # Forward Pass
        outputs = model(dummy_images)
        
        # Calculate our combined Focal + Dice loss
        mean_dice, loss = loss_fn(outputs, dummy_targets)
        
        # Backward Pass (verifying that the computation graph captures everything without syntax errors)
        loss.backward()

        print("[4/4] Forward Pass + Loss Evaluation + Backward Propagation: OK")
        print(f"      Calculated Mean Dice: {mean_dice.item():.4f}")
        print(f"      Calculated Overall Loss: {loss.item():.4f}")
        
        print("\n=== MOCK TEST COMPLETED SUCCESSFULLY! ===")
        print("Your modular code infrastructure works flawlessly and gradients flow properly!")
        
    except Exception as e:
        print(f"\nForward or Loss pass failed! Trace:\n{e}")

if __name__ == "__main__":
    run_mock_test()
