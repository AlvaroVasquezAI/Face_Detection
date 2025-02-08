# src/gui/Face_Detection_Tracker.py
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.face_detection import FaceDetection

class FaceDetector:
    def __init__(self, model_path, img_size=224):
        self.IMG_SIZE = img_size
        self.setup_gpu()
        self.model = self.load_model(model_path)
    
    def setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU configured successfully")
                return True
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
                return False
        print("No GPU available")
        return False
        
    def load_model(self, model_path):
        try:
            model_dir = os.path.dirname(model_path)

            params_path = os.path.join(model_dir, 'parameters.json')
            if not os.path.exists(params_path):
                raise FileNotFoundError(f"parameters.json not found in {model_dir}")
                
            with open(params_path, 'r') as f:
                params = json.load(f)

            face_detection = FaceDetection(
                class_weight=params['class_weight'],
                reg_weight=params['reg_weight'],
                learning_rate=params['learning_rate'],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                early_stopping_patience=params['early_stopping_patience'],
                reduce_lr_patience=params['reduce_lr_patience'],
                lr_decay_rate=params['lr_decay_rate']
            )

            face_detection.build_model(
                input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                use_batch_norm=True,
                dropout_rate=params['dropout_rate']
            )

            face_detection.compile()

            dummy_input = tf.zeros((1, self.IMG_SIZE, self.IMG_SIZE, 3))
            _ = face_detection(dummy_input)

            try:
                face_detection.load_weights(model_path)
                print("Model weights loaded successfully")
            except Exception as e:
                print(f"Error loading weights: {e}")
                raise

            print("\nModel Parameters:")
            print("="*50)
            for key, value in params.items():
                if key != 'model_architecture':
                    print(f"{key}: {value}")

            return face_detection
                
        except Exception as e:
            print(f"Error in load_model: {e}")
            print(f"Model path: {model_path}")
            print(f"File exists: {os.path.exists(model_path)}")
            raise

    def preprocess_image(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            
        original_size = img.shape[:2]
        return img, original_size

    def detect_face(self, image_path, threshold=0.5):
        original_image, original_size = self.preprocess_image(image_path)
        
        result = self.model.predict(image_path, threshold)
        
        result['original_image'] = original_image
        result['original_size'] = original_size
        
        return result

    def draw_detection(self, image, result):
        img_draw = image.copy()
        
        if result['has_face']:
            h, w = result['original_size']
            bbox = result['bbox']
            
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"Face: {result['confidence']:.2f}"
            cv2.putText(img_draw, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return img_draw

    def visualize_detection(self, image_path, threshold=0.5, save_path=None):
        result = self.detect_face(image_path, threshold)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(result['original_image'])
        
        if result['has_face']:
            h, w = result['original_size']
            bbox = result['bbox']
            
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color='r', linewidth=2,
                               label=f'Face: {result["confidence"]:.2f}')
            plt.gca().add_patch(rect)
            plt.legend()
            
        plt.title(f'Face Detected (Confidence: {result["confidence"]:.2f})'
                 if result['has_face'] else 'No Face Detected')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        
        plt.show()
        plt.close()
        
        return result

    def process_image_for_display(self, image_path, threshold=0.5):
        result = self.detect_face(image_path, threshold)
        processed_image = self.draw_detection(result['original_image'], result)
        return cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR), result
