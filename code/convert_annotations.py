import json
import os
import cv2
import numpy as np

def convert_to_paddle_format(json_path, output_path, path_prefix):
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as out_f:
        for entry in data:
            image_path = entry['image']
            # Adjust path based on local structure
            # JSON: rumsey/train/X -> Disk: train/rumsey/train/X
            # JSON: rumsey/val/X -> Disk: val/rumsey/val/X
            
            if 'train' in image_path:
                final_path = os.path.join('train', image_path)
            elif 'val' in image_path:
                final_path = os.path.join('val', image_path)
            else:
                final_path = image_path # Fallback

            # Normalize separators
            final_path = final_path.replace('\\', '/')

            paddle_labels = []
            for group in entry.get('groups', []):
                for item in group:
                    if item.get('illegible', False):
                        continue # Skip illegible text for training if desired, or keep it. Usually skip or mark as ###
                    
                    text = item['text']
                    vertices = item['vertices']
                    
                    points = np.array(vertices, dtype=np.float32)
                    
                    # Convert to 4 points (rotated rectangle)
                    rect = cv2.minAreaRect(points)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    # Sort points to be consistent (top-left, top-right, bottom-right, bottom-left) - PaddleOCR expects this order roughly
                    # Actually PaddleOCR just needs 4 points in clockwise order. boxPoints returns them in clockwise order usually.
                    
                    paddle_labels.append({
                        "transcription": text,
                        "points": box.tolist()
                    })
            
            if paddle_labels:
                json_str = json.dumps(paddle_labels)
                out_f.write(f"{final_path}\t{json_str}\n")

if __name__ == "__main__":
    base_dir = "dataset"
    convert_to_paddle_format(
        os.path.join(base_dir, "rumsey_train.json"),
        os.path.join(base_dir, "train_label.txt"),
        "train"
    )
    convert_to_paddle_format(
        os.path.join(base_dir, "rumsey_val.json"),
        os.path.join(base_dir, "val_label.txt"),
        "val"
    )
    print("Conversion complete.")
