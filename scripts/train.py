# scripts/train.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.data_processor import DataPreprocessor
from src.model.face_detection import FaceDetection
from src.utils.gpu_utils import setup_gpu

def train_model(
    data_path="Data/Data.csv",
    img_size=224,
    batch_size=62,
    class_weight=0.2,
    reg_weight=1.7,
    learning_rate=0.0001,
    epochs=15,
    early_stopping_patience=7,
    reduce_lr_patience=4,
    lr_decay_rate=0.9,
    dropout_rate=0.5
):
    try:
        setup_gpu()

        data = DataPreprocessor(data_path, img_size=img_size)
        print("\nData Info:")
        print("="*50)
        print(data.data.head())

        train_dataset, val_dataset, test_dataset = data.build_datasets()

        print("\nVisualizing dataset samples...")
        data.visualize_datasets_samples()

        print("\nBuilding model...")
        model = FaceDetection(
            class_weight=class_weight,
            reg_weight=reg_weight,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience,
            lr_decay_rate=lr_decay_rate
        )

        model.build_model(
            input_shape=(img_size, img_size, 3),
            use_batch_norm=True,
            dropout_rate=dropout_rate
        )

        print("\nCompiling model...")
        model.compile()

        print("\nTraining model...")
        history, model_dir = model.train(train_dataset, val_dataset)

        if history:
            print("\nModel trained successfully!")

            print("\nEvaluating model...")
            evaluation_results = model.evaluateModel(
                test_dataset,
                viualize_results=True
            )
            
            return history, model_dir, evaluation_results
        
        return None, None, None

    except Exception as e:
        print(f"\nError in training process: {e}")
        raise e

if __name__ == "__main__":
    CONFIG = {
        'data_path': "Data/Data.csv",
        'img_size': 224,
        'batch_size': 62,
        'class_weight': 0.2,
        'reg_weight': 1.7,
        'learning_rate': 0.0001,
        'epochs': 15,
        'early_stopping_patience': 7,
        'reduce_lr_patience': 4,
        'lr_decay_rate': 0.9,
        'dropout_rate': 0.5
    }

    history, model_dir, evaluation_results = train_model(**CONFIG)

    if history and model_dir:
        print("\nTraining completed successfully!")
        print(f"Model saved in: {model_dir}")
        
        if evaluation_results:
            print("\nEvaluation Results:")
            print("="*50)
            for metric, value in evaluation_results.items():
                print(f"{metric}: {value}")