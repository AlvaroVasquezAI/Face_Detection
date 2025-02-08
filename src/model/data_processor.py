import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, dataPath, img_size):
        self.dataPath = dataPath
        self.data = pd.read_csv(dataPath)
        self.img_size = img_size

        self.trainSet = None
        self.valSet = None
        self.testSet = None
        
    @tf.function
    def preprocess_image(self, image_path, class_label, bbox, is_training=False):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        
        image = tf.cast(image, tf.float32) / 255.0
        
        if is_training:
            image = tf.image.random_brightness(image, 0.3)  
            image = tf.image.random_contrast(image, 0.7, 1.3)  
            
            should_flip = tf.random.uniform([]) > 0.5
            image = tf.cond(
                should_flip,
                lambda: tf.image.flip_left_right(image),
                lambda: image
            )
            bbox = tf.cond(
                should_flip,
                lambda: tf.stack([1-bbox[2], bbox[1], 1-bbox[0], bbox[3]]),
                lambda: bbox
            )
            
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
            image = image + noise
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        image = tf.image.resize(image, (self.img_size, self.img_size))
        
        return image, (class_label, bbox)
    
    def create_dataset(self, dataframe, is_training=False, AUTOTUNE=tf.data.AUTOTUNE, BATCH_SIZE=32):
        image_paths = dataframe['Image_Path'].values
        classes = dataframe['class'].values
        bboxes = dataframe[['x1', 'y1', 'x2', 'y2']].values

        dataset = tf.data.Dataset.from_tensor_slices((
            image_paths,
            classes,
            bboxes
        ))

        if is_training:
            dataset = dataset.shuffle(buffer_size=len(dataframe))
        
        dataset = dataset.map(
            lambda x, y, z: self.preprocess_image(x, y, z, is_training=is_training),
            num_parallel_calls=AUTOTUNE
        )
        
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)

        return dataset
    
    def build_datasets(self, train=True, IsTrainSetTraining=True, val=True, IsValSetTraining=False, test=True, IsTestSetTraining=False):
        train_data = self.data[self.data['split'] == 'train']
        val_data = self.data[self.data['split'] == 'validation']
        test_data = self.data[self.data['split'] == 'test']
        
        with tf.device('/CPU:0'):
            if train:
                train_dataset = self.create_dataset(train_data, is_training=IsTrainSetTraining)
            if val:
                val_dataset = self.create_dataset(val_data, is_training=IsValSetTraining)
            if test:
                test_dataset = self.create_dataset(test_data, is_training=IsTestSetTraining)

        self.trainSet = train_dataset
        self.valSet = val_dataset
        self.testSet = test_dataset

        return train_dataset, val_dataset, test_dataset
    
    def dataSetsInfo(self, train=True, val=True, test=True):
        if train and self.trainSet:
            print("\nTrain Set:")
            for images, (classes, bboxes) in self.trainSet.take(1):
                print(f"Images shape: {images.shape}")
                print(f"Classes shape: {classes.shape}")
                print(f"Bounding boxes shape: {bboxes.shape}")
                print(f"Data type:")
                print(f"- Imagenes: {images.dtype}")
                print(f"- Classes: {classes.dtype}")
                print(f"- Bounding boxes: {bboxes.dtype}")
        if val and self.valSet is not None:
            print("\nValidation Set:")
            for images, (classes, bboxes) in self.valSet.take(1):
                print(f"Images shape: {images.shape}")
                print(f"Classes shape: {classes.shape}")
                print(f"Bounding boxes shape: {bboxes.shape}")
                print(f"Data type:")
                print(f"- Imagenes: {images.dtype}")
                print(f"- Classes: {classes.dtype}")
                print(f"- Bounding boxes: {bboxes.dtype}")
        if test and self.testSet is not None:
            print("\nTest Set:")
            for images, (classes, bboxes) in self.testSet.take(1):
                print(f"Images shape: {images.shape}")
                print(f"Classes shape: {classes.shape}")
                print(f"Bounding boxes shape: {bboxes.shape}")
                print(f"Data type:")
                print(f"- Imagenes: {images.dtype}")
                print(f"- Classes: {classes.dtype}")
                print(f"- Bounding boxes: {bboxes.dtype}")

    def visualize_datasets_samples(self, num_examples=4, train=True, val=True, test=True):
        with tf.device('/GPU:0'):
            if train and self.trainSet is not None:
                for images, (classes, bboxes) in self.trainSet.take(1):
                    images, classes, bboxes = self.process_batch(images, classes, bboxes)
                    self.visualize_samples(images, classes, bboxes, num_examples, "Train")
            if val and self.valSet is not None:
                for images, (classes, bboxes) in self.valSet.take(1):
                    images, classes, bboxes = self.process_batch(images, classes, bboxes)
                    self.visualize_samples(images, classes, bboxes, num_examples, "Validation")
            if test and self.testSet is not None:
                for images, (classes, bboxes) in self.testSet.take(1):
                    images, classes, bboxes = self.process_batch(images, classes, bboxes)
                    self.visualize_samples(images, classes, bboxes, num_examples, "Test")
    @tf.function
    def process_batch(self, images, classes, bboxes):
        return images, classes, bboxes
    
    def visualize_samples(self, images, classes, bboxes, num_samples, name):
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 10))
        fig.suptitle(f'{name} Dataset Samples', fontsize=16, y=1.02)

        images_np = images.numpy()
        classes_np = classes.numpy()
        bboxes_np = bboxes.numpy()
        
        if num_samples == 1:
            axes = [axes]

        for j, ax in enumerate(axes):
            if j < len(images_np):
                img = images_np[j]
                class_label = classes_np[j]
                bbox = bboxes_np[j]

                x1, y1, x2, y2 = [int(coord * self.img_size) for coord in bbox]

                if class_label == 1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 1, 0), 2)

                ax.imshow(img)
                ax.set_title(f"Class: {class_label}\nBBox: [{x1}, {y1}, {x2}, {y2}]")
                ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0)
        plt.show()
        plt.close()