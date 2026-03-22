import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Import configurations
import config
from config import SystemConf

# Import modularized components
from src.model import get_model, load_trained_model
from src.engine import get_optimizer_scheduler, run_training_loop
from src.visualization import plot_learning_curves, plot_class_iou_evolution
from src.inference import create_submission_file

def main():
    print("--- Semantic Segmentation Pipeline ---")
    
    # 1. Load DataFrames
    print("Loading data splits...")
    full_train_df = pd.read_csv(config.TRAIN_CSV)
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)
    test_df = pd.read_csv(config.TEST_CSV)
    
    # 2. Configure System Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sysconfig = SystemConf(device=device)
    
    if device.type == 'cpu':
        sysconfig.batch_size = 1
        sysconfig.num_workers = 2
        
    print(f"Using device: {sysconfig.device}")
    
    # === TRAINING PIPELINE ===
    # Uncomment below to Train
    '''
    print("Initializing Model...")
    model = get_model(encoder_name='efficientnet-b6', classes=12)
    optimizer, scheduler = get_optimizer_scheduler(model, sysconfig)
    
    print("Starting Training...")
    model, history = run_training_loop(
        model=model,
        train_df=train_df,
        val_df=val_df,
        image_path=config.IMAGE_DIR,
        mask_path=config.MASK_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
        sysconfig=sysconfig
    )
    
    print("Plotting Results...")
    plot_learning_curves(history)
    plot_class_iou_evolution(history, config.CLASS_NAMES)
    '''
    
    # === INFERENCE PIPELINE ===
    print("Generating Submission CSV...")
    path_model = 'best_model.pth'
    if os.path.exists(path_model):
        trained_model = load_trained_model(path_model, device=sysconfig.device, encoder_name='efficientnet-b6', classes=12)
        create_submission_file(
            trained_model, 
            test_df, 
            config.IMAGE_DIR, 
            config.MASK_DIR, 
            batch_size=1, 
            num_workers=sysconfig.num_workers, 
            device=sysconfig.device,
            out_file='submission.csv'
        )
    else:
        print(f"Model weights not found at {path_model}. Please train the model first by uncommenting the training logic.")

if __name__ == '__main__':
    main()
