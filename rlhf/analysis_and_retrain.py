# analysis_and_retrain.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from rlhf.utils import calculate_iou
from rlhf.model_improver import improve_model_with_feedback

def analyze_feedback_patterns(feedback_path):
    """Detailed analysis of feedback patterns"""
    with open(feedback_path, 'r') as f:
        feedback_data = json.load(f)
    
    analysis = {
        'total_samples': len(feedback_data),
        'rating_distribution': defaultdict(int),
        'confidence_stats': {
            'mean': 0,
            'std': 0,
            'distribution': defaultdict(int)
        },
        'iou_stats': {
            'mean': 0,
            'std': 0,
            'distribution': defaultdict(int)
        },
        'failure_patterns': {
            'low_confidence': 0,
            'high_confidence_wrong': 0,
            'poor_bbox': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    }
    
    # Collect metrics
    confidences = []
    ious = []
    ratings = []
    
    for feedback in feedback_data:
        # Rating distribution
        rating = feedback['rating']
        analysis['rating_distribution'][rating] += 1
        ratings.append(rating)
        
        # Confidence analysis
        conf = feedback['model_prediction']['confidence']
        confidences.append(conf)
        analysis['confidence_stats']['distribution'][round(conf, 1)] += 1
        
        # IoU analysis if face detected
        if feedback['model_prediction']['has_face']:
            iou = calculate_iou(
                feedback['model_prediction']['bbox'],
                feedback['human_correction']
            )
            ious.append(iou)
            analysis['iou_stats']['distribution'][round(iou, 1)] += 1
        
        # Analyze failure patterns
        if rating <= 2.0:  # Poor performance cases
            if conf < 0.5:
                analysis['failure_patterns']['low_confidence'] += 1
            elif conf > 0.8:
                analysis['failure_patterns']['high_confidence_wrong'] += 1
            
            if feedback['model_prediction']['has_face']:
                iou = calculate_iou(
                    feedback['model_prediction']['bbox'],
                    feedback['human_correction']
                )
                if iou < 0.5:
                    analysis['failure_patterns']['poor_bbox'] += 1
                    
            # False positives/negatives
            if feedback['model_prediction']['has_face'] and rating < 2.0:
                analysis['failure_patterns']['false_positives'] += 1
            elif not feedback['model_prediction']['has_face'] and rating < 2.0:
                analysis['failure_patterns']['false_negatives'] += 1
    
    # Calculate statistics
    analysis['confidence_stats']['mean'] = np.mean(confidences)
    analysis['confidence_stats']['std'] = np.std(confidences)
    
    if ious:
        analysis['iou_stats']['mean'] = np.mean(ious)
        analysis['iou_stats']['std'] = np.std(ious)
    
    return analysis

def visualize_analysis(analysis):
    """Visualize the analysis results"""
    plt.figure(figsize=(15, 10))
    
    # Rating distribution
    plt.subplot(2, 2, 1)
    ratings = sorted(analysis['rating_distribution'].items())
    plt.bar([r[0] for r in ratings], [r[1] for r in ratings])
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Confidence distribution
    plt.subplot(2, 2, 2)
    confs = sorted(analysis['confidence_stats']['distribution'].items())
    plt.bar([c[0] for c in confs], [c[1] for c in confs])
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    # IoU distribution
    plt.subplot(2, 2, 3)
    ious = sorted(analysis['iou_stats']['distribution'].items())
    plt.bar([i[0] for i in ious], [i[1] for i in ious])
    plt.title('IoU Distribution')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    
    # Failure patterns
    plt.subplot(2, 2, 4)
    patterns = analysis['failure_patterns']
    plt.bar(patterns.keys(), patterns.values())
    plt.title('Failure Patterns')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def determine_training_strategy(analysis):
    """Determine the best training strategy based on analysis"""
    failure_patterns = analysis['failure_patterns']
    
    # Base parameters
    # Recommended parameters based on analysis
    strategy = {
        'epochs': 40,
        'batch_size': 48,
        'early_stopping_patience': 12,
        'reduce_lr_patience': 5,
        'lr_decay_rate': 0.98,  
        'learning_rate': 1e-4, 
        'class_weight': 0.7,    
        'reg_weight': 2.2,      
        'dropout_rate': 0.5   
    }
    
    # Adjust weights based on dominant failure pattern
    total_failures = sum(failure_patterns.values())
    
    if failure_patterns['poor_bbox'] / total_failures > 0.4:
        print("Strategy: Focus on bounding box regression")
        strategy.update({
            'class_weight': 0.3,
            'reg_weight': 2.5,
            'learning_rate': 3e-4,
            'dropout_rate': 0.6
        })
    elif failure_patterns['low_confidence'] / total_failures > 0.4:
        print("Strategy: Focus on confidence improvement")
        strategy.update({
            'class_weight': 0.8,
            'reg_weight': 1.5,
            'learning_rate': 9e-4,
            'dropout_rate': 0.5
        })
    elif failure_patterns['high_confidence_wrong'] / total_failures > 0.4:
        print("Strategy: Focus on reducing false positives")
        strategy.update({
            'class_weight': 0.6,
            'reg_weight': 2.0,
            'learning_rate': 5e-5,
            'dropout_rate': 0.7
        })
    else:
        print("Strategy: Balanced approach")
        strategy.update({
            'class_weight': 0.5,
            'reg_weight': 2.0,
            'learning_rate': 1e-4,
            'dropout_rate': 0.6
        })
    
    return strategy

if __name__ == "__main__":
    feedback_path = "feedback/feedback_data.json"
    model_path = "models/face_detection_20250130_121042/best_weights.weights.h5"
    
    # Analyze feedback
    print("Analyzing feedback patterns...")
    analysis = analyze_feedback_patterns(feedback_path)

    # Save analysis results
    analysis_save_path = os.path.join(os.path.dirname(feedback_path), 'analysis_results.json')
    with open(analysis_save_path, 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        serializable_analysis = {
            k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
            for k, v in analysis.items()
        }
        json.dump(serializable_analysis, f, indent=4)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("="*50)
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Average confidence: {analysis['confidence_stats']['mean']:.3f} ± {analysis['confidence_stats']['std']:.3f}")
    if analysis['iou_stats']['mean']:
        print(f"Average IoU: {analysis['iou_stats']['mean']:.3f} ± {analysis['iou_stats']['std']:.3f}")
    
    print("\nFailure Patterns:")
    for pattern, count in analysis['failure_patterns'].items():
        print(f"- {pattern}: {count}")
    
    # Visualize analysis
    visualize_analysis(analysis)
    
    # Determine training strategy
    strategy = determine_training_strategy(analysis)
    print("\nProposed Training Strategy:")
    for param, value in strategy.items():
        print(f"- {param}: {value}")
    
    # Ask user if they want to proceed with training
    response = input("\nDo you want to proceed with training using this strategy? (y/n): ")
    
    if response.lower() == 'y':
        print("\nStarting training with determined strategy...")
        improved_model, history, model_dir = improve_model_with_feedback(
            feedback_path,
            model_path,
            custom_params=strategy
        )
        
        print(f"\nTraining completed!")
        print(f"Model saved in: {model_dir}")
    else:
        print("\nTraining cancelled.")