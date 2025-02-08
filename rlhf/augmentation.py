# rlhf/augmentation.py
import tensorflow as tf
import numpy as np

def apply_image_augmentation(image):
    """Apply various image augmentations"""
    # Random brightness
    image = tf.image.random_brightness(image, 0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Random saturation
    image = tf.image.random_saturation(image, 0.8, 1.2)
    
    # Random hue
    image = tf.image.random_hue(image, 0.1)
    
    # Random flip with bbox adjustment
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Ensure valid pixel values
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

def apply_bbox_augmentation(bbox, jitter=0.05):
    """Apply small random variations to bounding box"""
    noise = np.random.uniform(-jitter, jitter, size=4)
    aug_bbox = [
        max(0.0, min(1.0, coord + noise[i]))
        for i, coord in enumerate(bbox)
    ]
    
    # Ensure bbox is valid
    x1, y1, x2, y2 = aug_bbox
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    return [x1, y1, x2, y2]