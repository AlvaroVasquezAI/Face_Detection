# rlhf/validation.py
import numpy as np
import matplotlib.pyplot as plt
from rlhf.utils import calculate_iou

class RLHFValidator:
    def __init__(self, feedback_data, validation_split=0.2, seed=42):
        """
        Initialize RLHF validator
        
        Args:
            feedback_data: List of feedback samples
            validation_split: Proportion of feedback to use for validation
            seed: Random seed for reproducibility
        """
        self.all_feedback = feedback_data
        np.random.seed(seed)
        
        # Split feedback data
        self.train_feedback, self.val_feedback = self.split_feedback(
            feedback_data, 
            validation_split
        )
        
        # Initialize metrics tracking
        self.validation_history = []
        
    def split_feedback(self, feedback_data, split_ratio):
        """
        Split feedback data into training and validation sets
        Ensures balanced split based on ratings
        """
        # Group feedback by rating
        rating_groups = {}
        for feedback in feedback_data:
            rating = feedback['rating']
            if rating not in rating_groups:
                rating_groups[rating] = []
            rating_groups[rating].append(feedback)
        
        train_feedback = []
        val_feedback = []
        
        # Split each rating group
        for rating, samples in rating_groups.items():
            n_val = int(len(samples) * split_ratio)
            indices = np.random.permutation(len(samples))
            
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            val_feedback.extend([samples[i] for i in val_indices])
            train_feedback.extend([samples[i] for i in train_indices])
        
        return train_feedback, val_feedback
    
    def evaluate_improvement(self, original_model, improved_model):
        """
        Evaluate improvement using validation feedback
        """
        metrics = {
            'original': self.evaluate_model(original_model),
            'improved': self.evaluate_model(improved_model)
        }
        
        self.validation_history.append(metrics)
        return self.calculate_improvement(metrics)
    
    def evaluate_model(self, model):
        """
        Evaluate model on validation feedback
        """
        metrics = {
            'average_iou': [],
            'confidence_error': [],
            'rating_correlation': []
        }
        
        for feedback in self.val_feedback:
            # Get model prediction
            prediction = model.predict(feedback['image_path'])
            
            # Calculate IoU
            if prediction['has_face'] and feedback['model_prediction']['has_face']:
                iou = calculate_iou(
                    prediction['bbox'],
                    feedback['human_correction']
                )
                metrics['average_iou'].append(iou)
            
            # Calculate confidence error
            conf_error = abs(prediction['confidence'] - 
                           feedback['model_prediction']['confidence'])
            metrics['confidence_error'].append(conf_error)
            
            # Store for rating correlation
            metrics['rating_correlation'].append({
                'rating': feedback['rating'],
                'confidence': prediction['confidence']
            })
        
        return {
            'average_iou': np.mean(metrics['average_iou']),
            'confidence_error': np.mean(metrics['confidence_error']),
            'rating_correlation': self.calculate_rating_correlation(
                metrics['rating_correlation']
            )
        }
    
    def calculate_rating_correlation(self, data):
        """
        Calculate correlation between ratings and model confidence
        """
        ratings = [d['rating'] for d in data]
        confidences = [d['confidence'] for d in data]
        return np.corrcoef(ratings, confidences)[0, 1]
    
    def calculate_improvement(self, metrics):
        """
        Calculate improvement metrics
        """
        return {
            'iou_improvement': (
                metrics['improved']['average_iou'] - 
                metrics['original']['average_iou']
            ),
            'confidence_improvement': (
                metrics['original']['confidence_error'] - 
                metrics['improved']['confidence_error']
            ),
            'correlation_improvement': (
                metrics['improved']['rating_correlation'] - 
                metrics['original']['rating_correlation']
            )
        }
    
    def get_validation_feedback(self):
        """
        Get validation feedback data
        """
        return self.val_feedback
    
    def get_training_feedback(self):
        """
        Get training feedback data
        """
        return self.train_feedback
    
    def plot_improvement_history(self):
        """
        Plot improvement metrics over time
        """
        if not self.validation_history:
            print("No validation history available")
            return
            
        metrics = ['average_iou', 'confidence_error', 'rating_correlation']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            original_values = [h['original'][metric] 
                             for h in self.validation_history]
            improved_values = [h['improved'][metric] 
                             for h in self.validation_history]
            
            axes[idx].plot(original_values, label='Original')
            axes[idx].plot(improved_values, label='Improved')
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].legend()
            
        plt.tight_layout()
        plt.show()