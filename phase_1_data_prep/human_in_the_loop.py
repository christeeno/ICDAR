import os
import json
import cv2
import numpy as np

def process_labels(label_file, dataset_root):
    if not os.path.exists(label_file):
        print(f"File not found: {label_file}")
        return

    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated = False
    for idx, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
            
        rel_image_path = parts[0]
        gt_json = json.loads(parts[1])
        
        image_path = os.path.join(dataset_root, rel_image_path)
        img = None
        
        needs_update = False
        for item in gt_json:
            if item.get('transcription') == '###_REVIEW_NEEDED_###':
                if img is None:
                    img = cv2.imread(image_path)
                    if img is None:
                        # Map images might not be present locally if data is incomplete
                        # Fallback or skip
                        print(f"Warning: Image file not found for {image_path}. Skipping.")
                        continue
                
                points = np.array(item['points'], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)
                # Add small margin for readability
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)
                
                crop = img[y1:y2, x1:x2]
                
                if crop.size > 0:
                    cv2.imshow("Review Needed - Translate this text", crop)
                    cv2.waitKey(1)  # Refresh UI
                    
                    print(f"\n[{label_file}] Image: {rel_image_path}")
                    text = input("Enter transcription (or press Enter to skip and keep illegible): ").strip()
                    
                    if text:
                        item['transcription'] = text
                        needs_update = True
                        print(f"Updated transcript to: '{text}'")
                    else:
                        print("Skipped.")
                        
        if needs_update:
            lines[idx] = f"{rel_image_path}\t{json.dumps(gt_json)}\n"
            updated = True
            
    cv2.destroyAllWindows()
    
    if updated:
        with open(label_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Successfully updated labels in {label_file}")
    else:
        print(f"No changes made to {label_file}")

if __name__ == "__main__":
    base_dir = "dataset"
    print("--- Historical Map OCR: Human In The Loop Validation ---")
    process_labels(os.path.join(base_dir, "train_label.txt"), base_dir)
    process_labels(os.path.join(base_dir, "val_label.txt"), base_dir)
    print("Review session complete.")
