# rlhf/dataset_creator.py
import tensorflow as tf
from .augmentation import apply_image_augmentation, apply_bbox_augmentation

class FeedbackDatasetCreator:
    def __init__(self, feedback_data, augmentations_per_sample=5):
        self.feedback_data = feedback_data
        self.augmentations_per_sample = augmentations_per_sample
        self.n_original = len(feedback_data)
        self.n_augmented = self.n_original * augmentations_per_sample
        self.total_samples = self.n_original + self.n_augmented
        
    def create_dataset(self, batch_size):
        """Create dataset from feedback data"""
        # Create base dataset
        base_dataset = self.create_base_dataset()
        
        # Create augmented dataset
        aug_dataset = self.create_augmented_dataset()
        
        # Ensure consistent types before concatenation
        base_dataset = base_dataset.map(self._ensure_types)
        aug_dataset = aug_dataset.map(self._ensure_types)
        
        # Combine datasets
        dataset = base_dataset.concatenate(aug_dataset)
        
        # Configure dataset
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self._restructure_data)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        dataset_info = {
            'original_samples': self.n_original,
            'augmented_samples': self.n_augmented,
            'total_samples': self.total_samples
        }
        
        return dataset, dataset_info
    
    def create_base_dataset(self):
        """Create dataset from original feedback samples"""
        images = []
        classes = []
        bboxes = []
        
        for feedback in self.feedback_data:
            image = self._load_and_preprocess_image(feedback['image_path'])
            bbox = tf.cast(feedback['human_correction'], tf.float32)  # Ensure float32
            
            images.append(image)
            classes.append(tf.constant([1.0], dtype=tf.float32))
            bboxes.append(bbox)
        
        return tf.data.Dataset.from_tensor_slices((
            images,
            {
                'classification': classes,
                'bounding_box': bboxes
            }
        ))
    
    def create_augmented_dataset(self):
        """Create dataset with augmented samples"""
        aug_images = []
        aug_classes = []
        aug_bboxes = []
        
        for feedback in self.feedback_data:
            image = self._load_and_preprocess_image(feedback['image_path'])
            bbox = tf.cast(feedback['human_correction'], tf.float32)  # Ensure float32
            
            for _ in range(self.augmentations_per_sample):
                aug_image = apply_image_augmentation(image)
                aug_bbox = tf.cast(apply_bbox_augmentation(bbox), tf.float32)  # Ensure float32
                
                aug_images.append(aug_image)
                aug_classes.append(tf.constant([1.0], dtype=tf.float32))
                aug_bboxes.append(aug_bbox)
        
        return tf.data.Dataset.from_tensor_slices((
            aug_images,
            {
                'classification': aug_classes,
                'bounding_box': aug_bboxes
            }
        ))
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    @staticmethod
    def _ensure_types(images, labels):
        """Ensure consistent data types"""
        return (
            tf.cast(images, tf.float32),
            {
                'classification': tf.cast(labels['classification'], tf.float32),
                'bounding_box': tf.cast(labels['bounding_box'], tf.float32)
            }
        )
    
    @staticmethod
    def _restructure_data(images, labels):
        """Restructure data to match model's expected format"""
        return (
            images,
            (labels['classification'], labels['bounding_box'])
        )