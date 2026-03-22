import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemConf

def plot_learning_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    test_epochs = np.linspace(1, len(epochs), len(history['val_loss']))
    plt.plot(test_epochs, history['val_loss'], 'r-s', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_mean_dice'], 'b-o', label='Train mean Dice')
    plt.plot(test_epochs, history['val_mean_dice'], 'r-s', label='Validation mean Dice')
    
    plt.title('Training and Validation mean Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Dice Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_class_iou_evolution(history, class_names):
    intersections = history['final_intersections']
    unions = history['final_union']
    
    class_ious = intersections / (unions + 1e-7) 
    mean_iou = class_ious.mean(axis=1)

    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    for i in range(12):
        plt.plot(class_ious[:, i], label=class_names[i], color=colors[i], alpha=0.7, linewidth=1.5)

    plt.plot(mean_iou, label='MEAN IoU', color='black', linestyle='--', linewidth=3)

    plt.title("Evolution of IoU per Class over Epochs", fontsize=16)
    plt.xlabel("Validation Epoch", fontsize=12)
    plt.ylabel("IoU Score", fontsize=12)
    plt.ylim(0, 1.05)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def visualize_pred(model, val_loader, random_index, sysconfig):
    model.eval()
    for i in random_index:
        images, targets = val_loader.dataset[i]
        input_tensor = images.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor.to(sysconfig.device))
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1); plt.imshow(images.permute(1,2,0).cpu()); plt.title(f"Image: {i}")
        plt.subplot(1, 3, 2); plt.imshow(targets.cpu()); plt.title(f"Ground Truth: {i}")
        plt.subplot(1, 3, 3); plt.imshow(preds); plt.title(f"Prediction: {i}")
        plt.show()

def visualize_pred_test(model, test_loader, random_index, sysconfig):
    model.eval()
    for i in random_index:
        images, targets = test_loader.dataset[i]
        input_tensor = images.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor.to(sysconfig.device))
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        plt.figure(figsize=(8, 2))
        plt.subplot(1, 2, 1); plt.imshow(images.permute(1,2,0).cpu()); plt.title(f"Image: {i}")
        plt.subplot(1, 2, 2); plt.imshow(preds); plt.title(f"Prediction: {i}")
        plt.show()
