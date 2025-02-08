# rlhf/model_improver.py
import tensorflow as tf
import numpy as np
import json
import os
from src.model.face_detection import FaceDetection
from rlhf.utils import calculate_iou
import datetime
from rlhf.dataset_creator import FeedbackDatasetCreator
from rlhf.validation import RLHFValidator

class ModelImprover:
    def __init__(self, feedback_path, original_model_path):
        self.feedback_data = self.load_feedback(feedback_path)
        self.original_model_path = original_model_path
        self.model_dir = os.path.dirname(original_model_path)
        self.final_params = None
        
        # Initialize validator
        self.validator = RLHFValidator(self.feedback_data)
        # Get split feedback data
        self.train_feedback = self.validator.get_training_feedback()
        self.val_feedback = self.validator.get_validation_feedback()
        
        print(f"Training feedback samples: {len(self.train_feedback)}")
        print(f"Validation feedback samples: {len(self.val_feedback)}")

    def load_feedback(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def analyze_feedback_metrics(self):
        """Analyze feedback to suggest parameter adjustments"""
        metrics = {
            'average_iou': np.mean([
                calculate_iou(
                    f['model_prediction']['bbox'],
                    f['human_correction']
                ) for f in self.train_feedback if f['model_prediction']['has_face']
            ]),
            'average_confidence': np.mean([
                f['model_prediction']['confidence']
                for f in self.train_feedback
            ]),
            'average_rating': np.mean([
                f['rating'] for f in self.train_feedback
            ])
        }
        
        suggestions = self.suggest_parameter_adjustments(metrics)
        return metrics, suggestions

    def suggest_parameter_adjustments(self, metrics):
        """Suggest parameter adjustments based on metrics"""
        suggestions = {}
        
        # More granular adjustments based on metrics
        if metrics['average_iou'] < 0.3:
            suggestions['reg_weight'] = 2.5
            suggestions['learning_rate'] = 1e-5
            suggestions['dropout_rate'] = 0.7
        elif metrics['average_iou'] < 0.5:
            suggestions['reg_weight'] = 2.0
            suggestions['learning_rate'] = 2e-5
            suggestions['dropout_rate'] = 0.6
        
        if metrics['average_confidence'] > 0.85:
            suggestions['class_weight'] = 0.2
        elif metrics['average_confidence'] < 0.6:
            suggestions['class_weight'] = 0.8
        
        if metrics['average_rating'] < 2.5:
            suggestions['epochs'] = 30
            suggestions['early_stopping_patience'] = 10
        elif metrics['average_rating'] < 3.5:
            suggestions['epochs'] = 20
            suggestions['early_stopping_patience'] = 7
        
        return suggestions

    def create_priority_datasets(self, final_params):
        """Create datasets for two-phase training"""
        try:
            tf.keras.backend.clear_session()
            
            with tf.device('/CPU:0'):
                # Calculate priorities based on feedback
                priorities = self.calculate_sample_priorities()
                
                # Split feedback into high and normal priority
                priority_threshold = np.median(priorities)
                high_priority_feedback = [
                    f for f, p in zip(self.train_feedback, priorities)
                    if p > priority_threshold
                ]
                
                # Create high-priority dataset
                priority_creator = FeedbackDatasetCreator(
                    high_priority_feedback,
                    augmentations_per_sample=7  # More augmentations for priority samples
                )
                priority_dataset, priority_info = priority_creator.create_dataset(
                    batch_size=final_params['batch_size']
                )
                
                # Create full dataset
                full_creator = FeedbackDatasetCreator(
                    self.train_feedback,
                    augmentations_per_sample=8
                )
                full_dataset, full_info = full_creator.create_dataset(
                    batch_size=final_params['batch_size']
                )
                
                # Create validation dataset
                val_creator = FeedbackDatasetCreator(
                    self.val_feedback,
                    augmentations_per_sample=0  # No augmentation for validation
                )
                val_dataset, _ = val_creator.create_dataset(
                    batch_size=final_params['batch_size']
                )
                
                return {
                    'priority_dataset': priority_dataset,
                    'full_dataset': full_dataset,
                    'val_dataset': val_dataset,
                    'priority_info': priority_info,
                    'full_info': full_info
                }
                
        except Exception as e:
            print(f"Error in create_priority_datasets: {e}")
            tf.keras.backend.clear_session()
            raise e

    def calculate_sample_priorities(self):
        """Calculate priority scores for each sample"""
        priorities = []
        for feedback in self.train_feedback:
            # Rating component (lower ratings get higher priority)
            rating_priority = 1.0 - (feedback['rating'] / 5.0)
            
            # IoU component
            if feedback['model_prediction']['has_face']:
                iou = calculate_iou(
                    feedback['model_prediction']['bbox'],
                    feedback['human_correction']
                )
                iou_priority = 1.0 - iou
            else:
                iou_priority = 1.0
            
            # Confidence error component
            conf_error = abs(0.5 - feedback['model_prediction']['confidence'])
            conf_priority = conf_error * 2
            
            # Combine priorities
            final_priority = (rating_priority + iou_priority + conf_priority) / 3
            priorities.append(final_priority)
            
        return np.array(priorities)

    def improve_model(self, custom_params=None):
        """Improve model using feedback and suggested parameters"""
        # Analyze feedback and get parameter suggestions
        metrics, suggested_params = self.analyze_feedback_metrics()
        
        # Load original parameters
        with open(os.path.join(self.model_dir, 'parameters.json'), 'r') as f:
            original_params = json.load(f)
        
        # Combine parameters (priority: custom > suggested > original)
        final_params = original_params.copy()
        
        # Print parameters and changes
        print("\nOriginal Parameters:")
        for key, value in original_params.items():
            print(f"- {key}: {value}")
        
        print("\nSuggested Changes:")
        for key, value in suggested_params.items():
            if key in original_params and original_params[key] != value:
                print(f"- {key}: {original_params[key]} -> {value}")
                final_params[key] = value
        
        if custom_params:
            print("\nCustom Parameter Overrides:")
            for key, value in custom_params.items():
                print(f"- {key}: {final_params[key]} -> {value}")
                final_params[key] = value
        
        print("\nFinal Parameters:")
        print(json.dumps(final_params, indent=2))
        
        self.final_params = final_params
        
        # Initialize and build improved model
        improved_model = FaceDetection(
            class_weight=final_params['class_weight'],
            reg_weight=final_params['reg_weight'],
            learning_rate=final_params['learning_rate'],
            epochs=final_params['epochs'],
            batch_size=final_params['batch_size'],
            early_stopping_patience=final_params['early_stopping_patience'],
            reduce_lr_patience=final_params['reduce_lr_patience'],
            lr_decay_rate=final_params['lr_decay_rate']
        )
        
        improved_model.build_model(
            input_shape=(224, 224, 3),
            use_batch_norm=True,
            dropout_rate=final_params['dropout_rate']
        )
        improved_model.compile()
        
        # Initialize variables
        dummy_input = tf.zeros((1, 224, 224, 3))
        _ = improved_model(dummy_input)
        
        # Load weights
        improved_model.load_weights(self.original_model_path)
        
        return improved_model, final_params

    def load_original_model(self):
        """Load the original model for comparison"""
        with open(os.path.join(self.model_dir, 'parameters.json'), 'r') as f:
            params = json.load(f)
            
        model = FaceDetection(**params)
        model.build_model(
            input_shape=(224, 224, 3),
            use_batch_norm=True,
            dropout_rate=params['dropout_rate']
        )
        model.compile()
        
        dummy_input = tf.zeros((1, 224, 224, 3))
        _ = model(dummy_input)
        
        model.load_weights(self.original_model_path)
        return model

def improve_model_with_feedback(
    feedback_path,
    model_path,
    custom_params=None,
    save_dir=None
):
    """Main function to improve model using feedback"""
    
    def convert_to_serializable(obj):
        """Convert a value to JSON serializable format"""
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif tf.is_tensor(obj):
            return float(obj.numpy())
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'numpy'):
            return float(obj.numpy())
        return obj

    def convert_history(history):
        """Convert history object to serializable format"""
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        return convert_to_serializable(history_dict)
    
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    try:
        # Initialize improver
        with tf.device('/CPU:0'):
            improver = ModelImprover(feedback_path, model_path)
        
        # Create two sets of parameters for each phase
        if custom_params is None:
            custom_params = {}
            
        # Phase 1 parameters
        phase1_params = custom_params.copy()
        phase1_params['epochs'] = 25
        
        # Phase 2 parameters
        phase2_params = custom_params.copy()
        phase2_params['epochs'] = 50
        
        # Get improved model for phase 1
        improved_model, final_params = improver.improve_model(phase1_params)
        
        # Create priority datasets
        datasets = improver.create_priority_datasets(final_params)
            
        # Print dataset details
        print("\nFeedback Dataset Details:")
        print("="*50)
        print(f"Training samples: {datasets['full_info']['original_samples']}")
        print(f"High-priority samples: {datasets['priority_info']['original_samples']}")
        print(f"Validation samples: {len(improver.val_feedback)}")
        print(f"Augmented training samples: {datasets['full_info']['augmented_samples']}")
        print(f"Augmented priority samples: {datasets['priority_info']['augmented_samples']}")
        
        # Set up save directory
        if save_dir is None:
            save_dir = f"models/face_detection_improved_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Clear memory before training
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Train model
        print("\nStarting training...")
        if gpus:
            print_gpu_memory_usage()
        
        # Phase 1: train on high-priority samples
        print("\nPhase 1: Training on high-priority samples...")
        history1, _ = improved_model.train(
            datasets['priority_dataset'],
            datasets['val_dataset'],
            model_dir=os.path.join(save_dir, 'phase1')
        )
        
        # Create new model for phase 2
        improved_model, _ = improver.improve_model(phase2_params)
        
        # Phase 2: train on all samples
        print("\nPhase 2: Fine-tuning on all samples...")
        history2, model_dir = improved_model.train(
            datasets['full_dataset'],
            datasets['val_dataset'],
            model_dir=os.path.join(save_dir, 'phase2')
        )
        
        # Convert and combine histories
        combined_history = {
            'phase1': convert_history(history1) if history1 else {},
            'phase2': convert_history(history2) if history2 else {}
        }
        
        # Save combined history
        try:
            with open(os.path.join(save_dir, 'combined_history.json'), 'w') as f:
                json.dump(combined_history, f, indent=4)
        except Exception as e:
            print(f"Error saving combined history: {e}")
        
        # Evaluate improvement
        print("\nEvaluating improvement...")
        evaluation_results = improved_model.evaluateModel(datasets['val_dataset'])
        
        # Convert and save evaluation results
        try:
            serializable_results = convert_to_serializable(evaluation_results)
            with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(serializable_results, f, indent=4)
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
        
        return improved_model, combined_history, model_dir
        
    except Exception as e:
        print(f"Error in improve_model_with_feedback: {e}")
        tf.keras.backend.clear_session()
        gc.collect()
        raise e

def visualize_feedback_samples(feedback_data, n_samples=4):
    """Visualize random samples from feedback data"""
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    
    # Randomly select samples
    selected_samples = np.random.choice(
        feedback_data, 
        size=min(n_samples, len(feedback_data)), 
        replace=False
    )
    
    # Create subplot
    fig, axes = plt.subplots(1, n_samples, figsize=(20, 5))
    if n_samples == 1:
        axes = [axes]
    
    for idx, sample in enumerate(selected_samples):
        # Load image
        img = cv2.imread(sample['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Draw model prediction (red)
        if sample['model_prediction']['has_face']:
            bbox = sample['model_prediction']['bbox']
            x1, y1, x2, y2 = [int(coord * w) if i % 2 == 0 else int(coord * h) 
                             for i, coord in enumerate(bbox)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw human correction (green)
        bbox = sample['human_correction']
        x1, y1, x2, y2 = [int(coord * w) if i % 2 == 0 else int(coord * h) 
                         for i, coord in enumerate(bbox)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"Rating: {sample['rating']}/5\n"
                          f"Confidence: {sample['model_prediction']['confidence']:.2f}")
    
    plt.suptitle("Random Feedback Samples\nRed: Model Prediction, Green: Human Correction", 
                 y=1.05)
    plt.tight_layout()
    plt.show()

def print_gpu_memory_usage():
    """Helper function to monitor GPU memory usage"""
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"GPU Memory Usage:")
        print(f"- Total: {info.total / 1024**2:.2f} MB")
        print(f"- Used: {info.used / 1024**2:.2f} MB")
        print(f"- Free: {info.free / 1024**2:.2f} MB")
        print(f"- Utilization: {(info.used / info.total) * 100:.2f}%")
        
        nvidia_smi.nvmlShutdown()
    except:
        print("Could not get GPU memory information. Installing nvidia-smi might help.")