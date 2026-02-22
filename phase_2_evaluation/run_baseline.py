import os
import json
import cv2
import numpy as np
from paddleocr import PaddleOCR
from shapely.geometry import Polygon

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two polygons.
    box1, box2: list of [x, y] points
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0.0
        return inter / union
    except:
        return 0.0

def evaluate_baseline(label_file, dataset_root, num_samples=50):
    # Initialize PaddleOCR
    # attempting to use latest English model
    ocr = PaddleOCR(use_angle_cls=True, lang='en') 

    with open(label_file, 'r') as f:
        lines = f.readlines()

    # Use a subset for quick baseline
    subset_lines = lines[:num_samples]
    
    total_gt_boxes = 0
    total_pred_boxes = 0
    true_positives = 0
    
    iou_threshold = 0.5

    print(f"Running baseline on {len(subset_lines)} images...")

    for line in subset_lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
            
        rel_image_path = parts[0]
        gt_json = json.loads(parts[1])
        
        image_path = os.path.join(dataset_root, rel_image_path)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Run inference
        result = ocr.ocr(image_path)
        
        # result is a list of [line_res] where line_res is [box, (text, score)]
        # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        if subset_lines.index(line) == 0:
             print(f"DEBUG: result type: {type(result)}")
             print(f"DEBUG: result len: {len(result) if result else 'None'}")
             if result:
                 print(f"DEBUG: result[0] type: {type(result[0])}")
                 print(f"DEBUG: result[0]: {result[0]}")

        pred_boxes = []
        if result and result[0]:
            for line_res in result[0]:
                if isinstance(line_res, list) and len(line_res) >= 1:
                    box = line_res[0]
                    pred_boxes.append(box)
                else:
                    print(f"Unexpected line_res format: {line_res}")
        
        gt_boxes = [item['points'] for item in gt_json]
        
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)
        
        # Match predictions to GT
        # Simple greedy matching
        matched_gt = set()
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, g_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                # print(f"p_box: {p_box}, g_box: {g_box}")
                try:
                    iou = calculate_iou(p_box, g_box)
                except Exception as e:
                    print(f"Error calculating IoU: {e}")
                    print(f"p_box: {p_box}")
                    print(f"g_box: {g_box}")
                    iou = 0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)

    precision = true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
    h_mean = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("-" * 30)
    print(f"Baseline Results (Subset of {num_samples} images)")
    print(f"Total GT Boxes: {total_gt_boxes}")
    print(f"Total Pred Boxes: {total_pred_boxes}")
    print(f"True Positives (IoU >= {iou_threshold}): {true_positives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"H-mean: {h_mean:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_baseline(
        label_file="dataset/val_label.txt",
        dataset_root="dataset",
        num_samples=20 # Small subset for speed
    )
