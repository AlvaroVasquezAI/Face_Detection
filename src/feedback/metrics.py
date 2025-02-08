# src/feedback/metrics.py
import numpy as np
from typing import List, Dict, Any
import tensorflow as tf

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] normalized coordinates
        box2: [x1, y1, x2, y2] normalized coordinates
    
    Returns:
        IoU score between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-7)  # Add small epsilon to avoid division by zero
    
    return float(iou)

def calculate_feedback_metrics(feedback_data: List[Dict[str, Any]], iou_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Calculate metrics from feedback data with face-specific threshold
    
    Args:
        feedback_data: List of feedback dictionaries
        iou_threshold: IoU threshold for considering a detection as correct (default: 0.3)
    
    Returns:
        Dictionary containing calculated metrics
    """
    if not feedback_data:
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "average_confidence": 0.0,
            "average_iou": 0.0,
            "iou_statistics": {
                "threshold": iou_threshold,
                "by_range": {
                    "poor": 0,
                    "fair": 0,
                    "good": 0,
                    "excellent": 0
                }
            },
            "model_performance": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        }

    # Basic metrics
    total_feedback = len(feedback_data)
    ratings = []
    confidences = []
    ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # IoU statistics
    iou_ranges = {
        "poor": 0,      # IoU < 0.2
        "fair": 0,      # 0.2 <= IoU < 0.3
        "good": 0,      # 0.3 <= IoU < 0.4
        "excellent": 0  # IoU >= 0.4
    }

    for feedback in feedback_data:
        # Collect ratings and confidences
        ratings.append(feedback['rating'])
        confidences.append(feedback['model_prediction']['confidence'])
        
        # Calculate IoU and update statistics
        if feedback['model_prediction']['has_face']:
            iou = calculate_iou(
                feedback['model_prediction']['bbox'],
                feedback['human_correction']
            )
            ious.append(iou)
            
            # Categorize IoU
            if iou < 0.2:
                iou_ranges["poor"] += 1
            elif iou < 0.3:
                iou_ranges["fair"] += 1
            elif iou < 0.4:
                iou_ranges["good"] += 1
            else:
                iou_ranges["excellent"] += 1
            
            # Update TP/FP based on threshold
            if iou >= iou_threshold:
                true_positives += 1
            else:
                false_positives += 1
        else:
            false_negatives += 1

    # Calculate averages
    average_rating = np.mean(ratings)
    average_confidence = np.mean(confidences)
    average_iou = np.mean(ious) if ious else 0.0

    # Calculate precision, recall, and F1 score
    total_predictions = true_positives + false_positives
    total_actual = true_positives + false_negatives
    
    precision = true_positives / total_predictions if total_predictions > 0 else 0.0
    recall = true_positives / total_actual if total_actual > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate temporal metrics
    temporal_metrics = calculate_temporal_metrics(feedback_data) if len(feedback_data) > 1 else {}

    metrics = {
        "total_feedback": total_feedback,
        "average_rating": float(average_rating),
        "average_confidence": float(average_confidence),
        "average_iou": float(average_iou),
        "iou_statistics": {
            "threshold": iou_threshold,
            "by_range": iou_ranges
        },
        "model_performance": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }
    }

    if temporal_metrics:
        metrics["temporal_metrics"] = temporal_metrics

    return metrics

def calculate_temporal_metrics(feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics that track changes over time.
    
    Args:
        feedback_data: List of feedback dictionaries with timestamps
    
    Returns:
        Dictionary containing temporal metrics
    """
    # Sort feedback by timestamp
    sorted_feedback = sorted(feedback_data, key=lambda x: x['timestamp'])
    
    # Calculate moving averages
    window_size = min(10, len(sorted_feedback))
    ratings_ma = []
    ious_ma = []
    
    for i in range(len(sorted_feedback) - window_size + 1):
        window = sorted_feedback[i:i + window_size]
        
        # Calculate average rating for window
        avg_rating = np.mean([fb['rating'] for fb in window])
        ratings_ma.append(float(avg_rating))
        
        # Calculate average IoU for window
        window_ious = []
        for fb in window:
            if fb['model_prediction']['has_face'] and fb['human_correction']:
                iou = calculate_iou(
                    fb['model_prediction']['bbox'],
                    fb['human_correction']
                )
                window_ious.append(iou)
        
        if window_ious:
            avg_iou = np.mean(window_ious)
            ious_ma.append(float(avg_iou))
    
    return {
        "moving_average_rating": ratings_ma,
        "moving_average_iou": ious_ma,
        "window_size": window_size
    }