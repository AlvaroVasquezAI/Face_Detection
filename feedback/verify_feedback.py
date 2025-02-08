# verify_feedback.py
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def verify_feedback_data(json_path):
    with open(json_path, 'r') as f:
        feedback_data = json.load(f)
    
    print(f"Total feedback samples: {len(feedback_data)}")
    
    for idx, sample in enumerate(feedback_data):
        img_path = sample['image_path']
        if not Path(img_path).exists():
            print(f"Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = sample['image_size']
        
        plt.figure(figsize=(12, 8))
        
        plt.imshow(img)
        
        if sample['model_prediction']['has_face']:
            bbox = sample['model_prediction']['bbox']
            x1, y1, x2, y2 = bbox
 
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)

            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color='red', linewidth=2,
                label=f'Model Prediction (conf: {sample["model_prediction"]["confidence"]:.2f})'
            ))
        
        bbox = sample['human_correction']
        x1, y1, x2, y2 = bbox

        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, color='green', linewidth=2,
            label='Human Correction'
        ))
        
        plt.title(f"Feedback Sample {idx+1}\n"
                 f"Rating: {sample['rating']}/5.0\n"
                 f"Image size: {w}x{h}")
        plt.legend()
        
        print(f"\nSample {idx+1}:")
        print(f"Image: {Path(img_path).name}")
        print(f"Model bbox (normalized): {sample['model_prediction']['bbox']}")
        print(f"Human bbox (normalized): {sample['human_correction']}")
        print(f"Model bbox (pixels): ({int(sample['model_prediction']['bbox'][0]*w)}, "
              f"{int(sample['model_prediction']['bbox'][1]*h)}, "
              f"{int(sample['model_prediction']['bbox'][2]*w)}, "
              f"{int(sample['model_prediction']['bbox'][3]*h)})")
        print(f"Human bbox (pixels): ({x1}, {y1}, {x2}, {y2})")
        
        plt.axis('on')
        plt.grid(True, alpha=0.3) 
        plt.show()
        
        if idx < len(feedback_data) - 1:
            response = input("Press Enter to continue, 'q' to quit: ")
            if response.lower() == 'q':
                break

if __name__ == "__main__":
    json_path = "feedback/feedback_data.json" 
    verify_feedback_data(json_path)