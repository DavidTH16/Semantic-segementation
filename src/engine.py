import torch
import torch.optim as optim
import numpy as np
from torch.amp import autocast, GradScaler
import sys
import os

# Import modules from src
from .loss import CustomLoss
from .transforms import get_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemConf

def get_optimizer_scheduler(model, sysconfig):
    init_lr = sysconfig.learning_rate
    optimizer = optim.AdamW(   
        model.parameters(),
        lr=init_lr,
        weight_decay=0.00001
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=11,
        gamma=0.1
    )
    return optimizer, scheduler

def train_one_epoch(model, train_loader, optimizer, sysconfig, scaler, loss_fn):
    model.train()
    batch_loss = 0.0
    batch_mean_dice = 0.0
    
    device_type = 'cuda' if 'cuda' in str(sysconfig.device) else 'cpu'

    for images, targets in train_loader:
        images = images.to(sysconfig.device)
        targets = targets.to(sysconfig.device)
        optimizer.zero_grad()

        with autocast(device_type):
            outputs = model(images)
            mean_dice, loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss += loss.item()
        batch_mean_dice += mean_dice.item() 
    
    epoch_loss = batch_loss / len(train_loader)
    epoch_mean_dice = batch_mean_dice / len(train_loader)
    return epoch_loss, epoch_mean_dice

def val_one_epoch(model, val_loader, sysconfig, loss_fn):
    model.eval()
    val_loss = 0.0
    val_mean_dice = 0.0
    total_intersection = np.zeros(sysconfig.num_clases)
    total_union = np.zeros(sysconfig.num_clases)
    
    device_type = 'cuda' if 'cuda' in str(sysconfig.device) else 'cpu'
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(sysconfig.device)
            targets = targets.to(sysconfig.device)

            with autocast(device_type):
                outputs = model(images)
                mean_dice, loss = loss_fn(outputs, targets)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets_np = targets.cpu().numpy()

                for cl in range(sysconfig.num_clases):
                    pred_cl = preds == cl
                    targets_cl = targets_np == cl
                    intersection = np.logical_and(pred_cl, targets_cl).sum()
                    union = np.logical_or(pred_cl, targets_cl).sum()
                    total_intersection[cl] += intersection
                    total_union[cl] += union

            val_loss += loss.item()
            val_mean_dice += mean_dice.item()
        
        epoc_val_loss = val_loss / len(val_loader)
        epoc_val_mean_dice = val_mean_dice / len(val_loader)

    return epoc_val_loss, epoc_val_mean_dice, total_intersection, total_union

def run_training_loop(model, train_df, val_df, image_path, mask_path, optimizer, scheduler=None, sysconfig=SystemConf()):
    batch_size = sysconfig.batch_size
    num_workers = sysconfig.num_workers
   
    train_loader = get_data(image_path, mask_path, train_df, batch_size, num_workers, data_split='train')
    val_loader = get_data(image_path, mask_path, val_df, batch_size, num_workers, data_split='val')

    model.to(sysconfig.device)
    loss_fn = CustomLoss()
    scaler = GradScaler()
    best_mean_dice = 0.0

    history = {
        'train_loss': [],
        'train_mean_dice': [],
        'val_loss': [],
        'val_mean_dice': [],
        'final_union': [],
        'final_intersections': []
    }

    for epoch in range(sysconfig.epochs):
        train_loss, mean_dice = train_one_epoch(model, train_loader, optimizer, sysconfig, scaler, loss_fn)
        history['train_loss'].append(train_loss)
        history['train_mean_dice'].append(mean_dice)
        print(f'\nEpoch: {epoch+1}/{sysconfig.epochs}')
        print(f'Training -->   Train loss: {train_loss:.4f} | Mean dice: {mean_dice:.4f}')

        if epoch % sysconfig.test_interval == 0:
            val_loss, val_mean_dice, intersections, unions = val_one_epoch(model, val_loader, sysconfig, loss_fn)
            history['val_loss'].append(val_loss)
            history['val_mean_dice'].append(val_mean_dice)
            history['final_union'].append(unions)
            history['final_intersections'].append(intersections)

            print(f'Validation -->   Val loss: {val_loss:.4f} | Val Mean dice: {val_mean_dice:.4f}')
            
            if val_mean_dice > best_mean_dice:
                best_mean_dice = val_mean_dice
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'New best mean dice score saved! Model saved...')
            
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current learning rate: {current_lr:.6f}')
            
    print("-" * 60)
    return model, {k: np.array(v) for k, v in history.items()}
