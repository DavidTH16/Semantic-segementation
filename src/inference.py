import pandas as pd
import numpy as np
import cv2
import torch
from .transforms import get_data

def rle_encode(mask):
    """
    run-length encoding to go through each class and count the number of pixels
    if you need encode, check the requirements of your taks. 
    Key point: if your image is 256*256 = 65536, your encode method must produce 65536 pixels in total
    """

    pixels = mask.flatten() 
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(map(str, runs)).replace('\n', '').replace('\r', '').strip()

def generate_submission(model, test_loader, device):
    """
    befor the enconde, resize your image to its original size since we need to encode all the
    pixels in the image, you can use  cv2 to resize
    """

    
    model.eval()
    submission_list = []
    
    TARGET_W, TARGET_H = 1280, 720 

    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for i in range(len(img_ids)):
                img_id = img_ids[i]
                
                for class_idx in range(12):
                    binary_mask = (preds[i] == class_idx).astype(np.uint16)
                    resized_mask = cv2.resize(binary_mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
                    rle_str = rle_encode(resized_mask)
                    
                    submission_list.append({
                        "ImageID": f"{img_id}_{class_idx}",
                        "EncodedPixels": rle_str
                    })
                    
    df_sub = pd.DataFrame(submission_list)
    return df_sub
    
def create_submission_file(model, test_df, image_path, mask_path, batch_size, num_workers, device, out_file='submission.csv'):
    """
    create df according submission ---> ImageID and EncodedPixels columns
    """
    
    test_loader = get_data(image_path, mask_path, test_df, batch_size, num_workers, data_split='test')
    df_submission = generate_submission(model, test_loader, device)
    df_submission.to_csv(out_file, index=False)
    print(f"Submission saved to {out_file}")
