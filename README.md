# Face Detection with RLHF (Reinforcement Learning from Human Feedback)

A comprehensive **machine learning project** that implements an advanced face detection system combining transfer learning with human feedback for continuous improvement. The project leverages MobileNetV2's architecture as its backbone and implements a sophisticated RLHF (Reinforcement Learning from Human Feedback) pipeline, creating an adaptive system that learns from user interactions.

The model performs dual tasks: face detection with confidence scoring and precise bounding box prediction. Through a custom-built GUI application, users can not only detect faces in real-time but also provide feedback on the model's performance. This feedback is systematically collected and analyzed through a two-phase training approach that prioritizes challenging cases, ensuring the model continuously improves its accuracy and generalization capabilities.

What sets this project apart is its end-to-end implementation of the RLHF concept in computer vision. While traditional face detection models remain static after training, this system creates a continuous improvement loop where human feedback directly influences model behavior. The implementation includes comprehensive metrics tracking, automated parameter adjustment based on feedback patterns, and a structured approach to model enhancement.

<div align="center">
  <img src="results/best_model_improved_results/rlhf_dataset/3.png" width="400" alt="Face Detection">
</div>

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training with Grid Search](#training-with-grid-search)
- [RLHF Implementation](#rlhf-implementation)
- [Results](#results)
- [GUI Application](#gui-application)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This project implements a sophisticated face detection system that combines transfer learning with human feedback for continuous improvement. Built on MobileNetV2's architecture, the system performs dual tasks: face detection with confidence scoring and precise bounding box prediction, achieving robust performance through a carefully designed training pipeline.

The implementation features three key components:
1. **Transfer Learning Model**: Leverages MobileNetV2's pre-trained weights, adapted for face detection through a custom dual-head architecture for classification and bounding box regression.
2. **RLHF Pipeline**: Implements a systematic approach to collect and utilize human feedback, enabling continuous model improvement through a two-phase training strategy.
3. **Interactive GUI**: Provides a user-friendly interface for real-time face detection and feedback collection, creating a seamless loop between model predictions and user interactions.

The project is trained on a balanced dataset of 11,985 images, with a comprehensive evaluation system that tracks both traditional metrics and user feedback. Through the RLHF implementation, the model adapts to challenging cases and improves its performance based on real-world usage.

### Some results

<div align="center">
  <h4>Dataset Results</h4>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
    <img src="results/best_model_improved_results/original_dataset/1.png" width="200" alt="Dataset Result 1">
    <img src="results/best_model_improved_results/original_dataset/2.png" width="200" alt="Dataset Result 2">
    <img src="results/best_model_improved_results/original_dataset/3.png" width="200" alt="Dataset Result 3">
    <img src="results/best_model_improved_results/original_dataset/4.png" width="200" alt="Dataset Result 4">
  </div>
  
  <h4>RLHF Results</h4>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
    <img src="results/best_model_improved_results/rlhf_dataset/1.png" width="200" alt="RLHF Result 1">
    <img src="results/best_model_improved_results/rlhf_dataset/2.png" width="200" alt="RLHF Result 2">
    <img src="results/best_model_improved_results/rlhf_dataset/3.png" width="200" alt="RLHF Result 3">
    <img src="results/best_model_improved_results/rlhf_dataset/4.png" width="200" alt="RLHF Result 4">
  </div>
  
  <h4>Real World Results</h4>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
    <img src="results/best_model_improved_results/real_world_dataset/1.png" width="200" alt="Real World Result 1">
    <img src="results/best_model_improved_results/real_world_dataset/2.png" width="200" alt="Real World Result 2">
    <img src="results/best_model_improved_results/real_world_dataset/3.png" width="200" alt="Real World Result 3">
    <img src="results/best_model_improved_results/real_world_dataset/4.png" width="200" alt="Real World Result 4">
  </div>
</div>

## Project Structure

<pre>
face_detection/
├── Data/                      
│   ├── Test/   
│       ├── Images/                  # Test set images
│           ├── x_y.jpg
│           └── ...
│       └── Labels/                  # Test set annotations
│           ├── x_y.json
│           └── ...
│   ├── Train/  
│       ├── Images/                  # Training set images
│           ├── x_y.jpg
│           └── ...
│       └── Labels/                  # Training set annotations
│           ├── x_y.json
│           └── ...
│   ├── Validation/ 
│       ├── Images/                  # Validation set images
│           ├── x_y.jpg
│           └── ...
│       └── Labels/                  # Validation set annotations
│           ├── x_y.json
│           └── ...
│   └── Data.csv                     # Dataset metadata and specifications
│
├── feedback/                        # Feedback system
│   ├── criteria.txt                 # Feedback evaluation criteria
│   ├── feedback_data.json           # Collected feedback data
│   ├── feedback_metrics.json        # Feedback analysis metrics
│   └── verify_feedback.py           # Feedback verification tools
│
├── grid_search_results/             # Hyperparameter optimization
│   ├── combination_1.json           # Individual trial results
│   ├── grid_search_results.csv      # Results summary
│   └── grid_search.log             # Training logs
│
├── models/                          # Trained models
│   ├── face_detection_XXXXXX/       # Model versions
│       ├── best_weights.weights.h5  # Best model weights
│       ├── evaluation_results.png   # Performance visualizations
│       ├── parameters.json          # Model parameters
│       └── training_history.json    # Training metrics
│
├── results/                         # Evaluation results
│   ├── best_model_improved_results/ # RLHF-improved model results
│       ├── orignal_dataset/         # Results on original data
│       ├── real_world_dataset/      # Results on real-world tests
│       └── rlhf_dataset/           # Results on RLHF data
│   └── rlhf/                       # RLHF analysis
│       └── analysis_feedback.png    # Feedback visualizations
│
├── rlhf/                           # RLHF implementation
│   ├── data/                       # RLHF training data
│   ├── augmentation.py             # Data augmentation
│   ├── dataset_creator.py          # Dataset management
│   ├── model_improver.py           # Model improvement
│   └── utils.py                    # Utility functions
│
├── scripts/                        # Training scripts
│   ├── train_gridSearch.py         # Grid search implementation
│   └── train.py                    # Base training script
│
├── src/                           # Core implementation
│   ├── feedback/                  # Feedback collection
│   ├── gui/                      # GUI implementation
│   ├── model/                    # Model architecture
│   └── utils/                    # Utility functions
│
└── requirements.txt               # Project dependencies
</pre>

### Key Components

1. **Data Organization**
   - Structured dataset splits with images and labels
   - Comprehensive metadata tracking
   - Standardized annotation format

2. **Model Development**
   - Grid search optimization
   - Multiple model versions
   - Training and evaluation scripts
   - Performance tracking

3. **RLHF System**
   - Feedback collection and analysis
   - Model improvement pipeline
   - Results visualization
   - Data augmentation

4. **User Interface**
   - Interactive GUI application
   - Real-time detection
   - Feedback submission
   - Result visualization

This structure ensures modular development, easy maintenance, and systematic tracking of experiments and improvements.

## Dataset

The project utilizes a carefully curated dataset combining images from two renowned sources: **Labeled Faces in the Wild (LFW)** and **Jack Dataset**, creating a balanced collection of 11,985 images for face detection training.
### Dataset Distribution

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
    <thead style="background-color: #f2f2f2;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Set</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Total Images</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">% of Dataset</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Face Images</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">% Faces</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">No Face Images</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">% No Faces</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Train</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">9588</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">80.00%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">4794</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">50.00%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">4794</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">50.00%</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Test</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1197</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">9.99%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">598</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">49.96%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">599</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">50.04%</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Validation</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1200</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">10.01%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">600</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">50.00%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">600</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">50.00%</td>
        </tr>
        <tr style="background-color: #f2f2f2; font-weight: bold;">
            <td style="border: 1px solid #ddd; padding: 8px;">Total</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">11985</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100.00%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">5992</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">50.00%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">5993</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">50.00%</td>
        </tr>
    </tbody>
</table>

### Sample Images

<div align="center">
  <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
    <div>
      <h4>LFW Dataset (Face Examples)</h4>
      <img src="Data/Test/Images/1_1.jpg" width="250" height="250" alt="Face Example 1">
      <img src="Data/Test/Images/64_1.jpg" width="250" height="250" alt="Face Example 2">
    </div>
    <div>
      <h4>Jack Dataset (Non-face Examples)</h4>
      <img src="Data/Test/Images/140_0.jpg" width="250" height="250" alt="Non-face Example 1">
      <img src="Data/Test/Images/169_0.jpg" width="250" height="250" alt="Non-face Example 2">
    </div>
  </div>
</div>

### Key Characteristics

- **Image Format**: JPG
- **Dimensions**: 250x250 pixels (standardized)
- **Color Space**: RGB
- **Class Balance**: Near-perfect (50% faces, 50% non-faces)
- **Split Ratio**: 80% train, 10% validation, 10% test
- **Annotations**: Bounding box coordinates in JSON format

The dataset's balanced nature and diverse composition provide a solid foundation for training a robust face detection model, while its standardized format ensures consistent processing throughout the pipeline.

## Model Architecture

The face detection model is built using transfer learning with MobileNetV2 as the backbone, implementing a dual-head architecture for simultaneous face classification and bounding box regression. The model is designed to be efficient while maintaining high accuracy in both tasks.

### Base Model
- **Backbone**: MobileNetV2 (pre-trained on ImageNet)
- **Input Shape**: 224×224×3 (RGB images)
- **Feature Extraction**: Global Max Pooling on backbone output
- **Trainable Base**: False (frozen weights for transfer learning)

### Dual-Head Architecture

1. **Classification Branch**:
   ```python
   # First Dense Block
   Dense(1024) → BatchNorm → ReLU → Dropout
   # Second Dense Block
   Dense(512) → BatchNorm → ReLU → Dropout
   # Output
   Dense(1, sigmoid) # Face/No-Face Classification
   ```

2. **Regression Branch**:
   ```python
   # First Dense Block
   Dense(1024) → BatchNorm → ReLU → Dropout
   # Second Dense Block
   Dense(512) → BatchNorm → ReLU → Dropout
   # Output
   Dense(4, sigmoid) # Bounding Box Coordinates [x1, y1, x2, y2]
   ```

### Loss Functions

1. **Classification Loss**:
   - Binary Cross-Entropy with label smoothing (0.1)
   - Helps prevent overconfident predictions

2. **Regression Loss**:
   ```python
   def regression_loss(y_true, y_pred):
       # Coordinate difference
       delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - y_pred[:,:2]))
       # Size difference
       h_true = y_true[:,3] - y_true[:,1]
       w_true = y_true[:,2] - y_true[:,0]
       h_pred = y_pred[:,3] - y_pred[:,1]
       w_pred = y_pred[:,2] - y_pred[:,0]
       delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + 
                                tf.square(h_true - h_pred))
       return delta_coord + delta_size
   ```

### Training Configuration

```python
# Parameters
learning_rate 
batch_size
epochs
class_weight 
reg_weight 
dropout_rate 
```

### Optimization Strategy

1. **Optimizer**: Adam with learning rate decay
   ```python
   lr_schedule = learning_rate * (decay_rate ^ epoch)
   ```

2. **Regularization**:
   - L2 regularization (0.02) on dense layers
   - Dropout (0.5) after each dense layer
   - Batch Normalization for stable training

3. **Early Stopping**:
   - Monitor: validation total loss
   - Patience: 5 epochs
   - Restore best weights

### Model Callbacks

1. **Early Stopping**: Prevents overfitting
2. **Model Checkpoint**: Saves best weights
3. **Learning Rate Scheduler**: Implements decay
4. **CSV Logger**: Tracks training metrics
5. **Training Time Tracker**: Monitors duration

### Prediction Pipeline

```python
def predict(image_path, threshold=0.5):
    # Image preprocessing
    img = load_and_preprocess(image_path)
    
    # Model prediction
    class_pred, bbox_pred = model(img)
    
    # Threshold-based detection
    has_face = class_pred >= threshold
    
    return {
        'has_face': has_face,
        'confidence': class_prob,
        'bbox': bbox_coords if has_face else None
    }
```

The architecture is designed to balance accuracy and efficiency, making it suitable for real-time face detection while maintaining robust performance in both classification and localization tasks.

You're right. Let me revise the Training with Grid Search section with the correct number of combinations and include the best model results:

## Training with Grid Search

To improve the model's performance, we implemented a comprehensive grid search over key hyperparameters. The search explored 24 different combinations (2×3×1×2×2×1×1×1×1 = 24) of parameters to find the most effective configuration.

### Hyperparameter Grid

```python
param_grid = {
    'class_weight': [0.1, 0.2],           # Balance between classification and regression
    'reg_weight': [1.7, 1.8, 1.9],        # Importance of bounding box accuracy
    'learning_rate': [0.0001],            # Initial learning rate
    'batch_size': [96, 128],              # Training batch size
    'dropout_rate': [0.6, 0.7],           # Regularization strength
    'epochs': [25],                       # Maximum training epochs
    'early_stopping_patience': [7],        # Epochs before early stopping
    'reduce_lr_patience': [4],            # Epochs before LR reduction
    'lr_decay_rate': [0.9]                # Learning rate decay factor
}
```

### Best Model Configuration

The grid search identified the optimal configuration (Combination ID: 13):

```python
best_params = {
    'class_weight': 0.2,
    'reg_weight': 1.7,
    'learning_rate': 0.0001,
    'batch_size': 96,
    'dropout_rate': 0.6,
    'epochs': 25,
    'early_stopping_patience': 7,
    'reduce_lr_patience': 4,
    'lr_decay_rate': 0.9
}
```

### Performance Metrics

The best model achieved exceptional results:

1. **Classification Performance**:
   - Accuracy: 1.0000
   - Loss: 0.2028
   - Precision: 1.0000
   - Recall: 1.0000
   - F1 Score: 1.0000

2. **Regression Performance**:
   - MAE: 0.1476
   - MSE: 0.0653
   - RMSE: 0.2556
   - Total Loss: 1.4716

3. **Training Characteristics**:
   - Training Time: 7.38 minutes
   - Early Stopping: Yes (at epoch 24)
   - Final Validation Loss: 1.4733

### Key Findings

1. **Parameter Sensitivity**:
   - Higher class_weight (0.2) improved detection stability
   - Moderate reg_weight (1.7) provided best bbox accuracy
   - Lower batch size (96) offered better generalization

2. **Model Behavior**:
   - Perfect classification accuracy on validation set
   - Strong bounding box prediction (MAE: 0.1476)
   - Efficient training convergence (early stopping at 24/25 epochs)

3. **Test Set Performance**:
   - Maintained perfect classification (Accuracy: 1.0)
   - Strong regression metrics (MAE: 0.1500)
   - Robust F1 Score (1.0)

These results demonstrate the effectiveness of the grid search in finding a balanced configuration that excels in both face detection and bounding box regression tasks.

### Best Model Results
<div align="center">
  <h4>Test Set</h4>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
    <img src="results/best_model_improved_results/original_dataset/1.png" width="200" alt="Dataset Result 1">
    <img src="results/best_model_improved_results/original_dataset/2.png" width="200" alt="Dataset Result 2">
    <img src="results/best_model_improved_results/original_dataset/3.png" width="200" alt="Dataset Result 3">
    <img src="results/best_model_improved_results/original_dataset/4.png" width="200" alt="Dataset Result 4">
  </div>
</div>

Thank you for providing these details. Let me update the Analysis section with the actual metrics:

You're right. Let me revise the RLHF Implementation section to include the complete process:

## RLHF Implementation

Despite achieving excellent performance in controlled environments through transfer learning and grid search optimization (accuracy: 1.000, precision: 1.000), the model faced challenges in real-world scenarios. The implementation of Reinforcement Learning from Human Feedback (RLHF) aims to bridge this gap, creating a continuous improvement loop that adapts to real-world conditions such as varying lighting, different face angles, diverse image qualities, and occlusions.

The RLHF process follows three stages:

1. **Feedback Collection**: Through a GUI interface where users evaluate model predictions, provide correct bounding boxes, rate performance, and add comments.

2. **Feedback Analysis**: Systematic evaluation of performance patterns, failure modes, user ratings, detection confidence, and bounding box accuracy.

3. **Model Improvement**: Targeted enhancement through automatic strategy determination and priority-based training.

### Stage 1: Feedback Collection

1. **GUI Interface**:
   - Custom interface built with CustomTkinter
   - Model and image selection capabilities
   - Real-time face detection visualization
   - Interactive bounding box correction tool
   - Rating system (0-5 scale)
   - Comments section for additional feedback

<div align="center">
  <img src="results/rlhf/rlhf_1_3.png" alt="RLHF_Overview">
</div>

2. **Feedback Collection Process**:
   - Used best model from Grid Search (Combination ID: 13)
   - Selected 100 external images for evaluation
   - For each image:
     * Model makes prediction
     * User draws correct bounding box
     * Provides rating and comments
     * Feedback saved in JSON format

<div align="center">
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
    <img src="results/rlhf/rlhf_1_3.png" width="500" alt="RLHF 1">
    <img src="results/rlhf/rlhf_2_3.png" width="500" alt="RLHF 2">
    <img src="results/rlhf/rlhf_3_3.png" width="500" alt="RLHF 3">
    <img src="results/rlhf/rlhf_4_3.png" width="500" alt="RLHF 4">
  </div>
</div>

3. **Feedback Structure**:
```python
feedback_data = {
    'image_path': str,
    'model_name': str,
    'model_prediction': {
        'has_face': bool,
        'confidence': float,
        'bbox': List[float]
    },
    'human_correction': List[float],
    'rating': float,
    'comments': str,
    'timestamp': str,
    'image_size': Tuple[int, int]
}
```

<div align="center">
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
    <img src="results/rlhf/feedback1.png" width="200" alt="RLHF Preview 1">
    <img src="results/rlhf/feedback2.png" width="150" alt="RLHF Preview 2">
    <img src="results/rlhf/feedback3.png" width="180" alt="RLHF Preview 3">
    <img src="results/rlhf/feedback4.png" width="350" alt="RLHF Preview 4">
  </div>
</div>

### Stage 2: Feedback Analysis

After collecting feedback from 100 images through the GUI interface, the analysis revealed significant insights about the model's performance:

1. **Overall Performance Metrics**:
```python
metrics = {
    'total_feedback': 100,
    'average_rating': 2.0,        # Below average performance
    'average_confidence': 0.622,  # Moderate confidence
    'average_iou': 0.317         # Low IoU score
}
```

2. **IoU Distribution by Quality**:
```python
iou_ranges = {
    'excellent': 15,  # IoU >= 0.4
    'good': 24,      # 0.3 <= IoU < 0.4
    'fair': 20,      # 0.2 <= IoU < 0.3
    'poor': 10       # IoU < 0.2
}
```

3. **Classification Performance**:
```python
classification_metrics = {
    'true_positives': 39,
    'false_positives': 30,
    'false_negatives': 31,
    'precision': 0.565,
    'recall': 0.557,
    'f1_score': 0.561
}
```

### Rating Criteria

The feedback was collected using a standardized 5-point scale:

| Rating | Criteria |
|--------|----------|
| 5 | Perfect detection and bbox (100% of face) |
| 4 | Good detection, minor bbox issues (>66% of face) |
| 3 | Correct detection, noticeable bbox issues (>33% of face) |
| 2 | Correct detection, poor bbox (<33% of face) |
| 1 | Poor detection and bbox (<33% face, <50% detection) |
| 0 | Completely wrong (<25% detection, wrong bbox) |

### Performance Analysis

1. **Detection Issues**:
   - High false negative rate (31%)
   - Significant false positives (30%)
   - Balanced but low precision-recall trade-off

2. **Bounding Box Quality**:
   - Only 15% achieved excellent IoU
   - 44% good or excellent performance
   - 30% fair or poor performance

3. **Temporal Trends**:
   - Initial average rating: 2.7
   - Final average rating: 2.2
   - Declining performance on challenging cases
   - IoU fluctuation between 0.25-0.45

<div align="center">
  <img src="results/rlhf/analysis_feedback.png" alt="Analysis Feedback Results">
</div>

These findings led to specific strategy adjustments in the improvement phase, particularly focusing on:
1. Reducing false negatives
2. Improving bounding box precision
3. Enhancing confidence calibration
   
### Stage 3: Model Improvement

The RLHF implementation employs a systematic approach to improve model performance through feedback analysis and targeted training. The process consists of three main components: automatic strategy determination, phased training implementation, and performance evaluation.

1. **Automatic Strategy Determination**

The system analyzes failure patterns in the feedback data to automatically determine the optimal training strategy. It considers four potential scenarios:
- Poor bounding box performance (>40% of failures): Emphasizes regression with higher reg_weight
- Low confidence issues (>40% of failures): Focuses on classification with higher class_weight
- High confidence errors (>40% of failures): Addresses false positives with adjusted learning rate
- Balanced issues: Uses moderate parameters across all aspects

In this case, the analysis revealed a distributed pattern of issues:
```python
failure_patterns = {
    'low_confidence': 31,
    'high_confidence_wrong': 9,
    'poor_bbox': 25,
    'false_positives': 1,
    'false_negatives': 31
}
```

Since no single failure pattern exceeded the 40% threshold, the system selected a balanced approach with the following parameters:
```python
strategy = {
    'epochs': 40,
    'batch_size': 48,
    'early_stopping_patience': 12,
    'reduce_lr_patience': 5,
    'lr_decay_rate': 0.98,
    'class_weight': 0.5,    
    'reg_weight': 2.0,      
    'learning_rate': 1e-4,  
    'dropout_rate': 0.6     
}
```

2. **Two-Phase Training Implementation**

The improvement process implements a two-phase training approach to maximize the impact of feedback:

Phase 1 (Priority Training):
- Focuses on samples with ratings ≤ 2 (41 original samples)
- Applies aggressive augmentation (7 variations per sample)
- Emphasizes learning from problematic cases
- Uses 18 validation samples for performance monitoring

Phase 2 (Comprehensive Training):
- Includes all feedback samples (82 original samples)
- Applies standard augmentation (5 variations per sample)
- Ensures balanced learning from all feedback types
- Maintains consistent validation set for comparison

3. **Performance Results**

The implementation achieved significant improvements across all metrics:
```python
final_metrics = {
    # Classification Metrics
    'class_accuracy': 1.000,    
    'class_precision': 1.000,   
    'class_recall': 1.000,      
    'f1_score': 1.000,         
    
    # Regression Metrics
    'reg_mae': 0.114,          
    'reg_mse': 0.028,          
    'reg_rmse': 0.167,         
    
    # Overall Performance
    'total_loss': 3.110,       
    'class_loss': 0.240,       
    'reg_loss': 1.495         
}
```

These results demonstrate the effectiveness of our RLHF implementation in:
- Achieving perfect classification performance
- Significantly improving bounding box precision
- Maintaining balanced overall performance
- Successfully addressing identified failure patterns

The balanced strategy, automatically determined through feedback analysis, proved highly effective in improving both the classification accuracy and bounding box precision of the model.

<div align="center">
  <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
    <div>
      <h4>Before Improvement</h4>
      <img src="results/best_model_results/rlhf_dataset/1.png" width="360" alt="Result 1">
      <img src="results/best_model_results/rlhf_dataset/2.png" width="450" alt="Result 2">
    </div>
    <div>
      <h4>After Improvement</h4>
      <img src="results/best_model_improved_results/rlhf_dataset/1.png" width="360" alt="Result 1 improvement">
      <img src="results/best_model_improved_results/rlhf_dataset/2.png" width="450"  alt="Result 2 improvement">
    </div>
  </div>
</div>

