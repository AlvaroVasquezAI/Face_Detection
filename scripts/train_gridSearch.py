# scripts/train_gridSearch.py

import pandas as pd
import tensorflow as tf
import os
import json
import datetime
import sys
from typing import Dict, List, Any, Optional
import logging
from itertools import product

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.data_processor import DataPreprocessor
from src.model.face_detection import FaceDetection
from src.utils.gpu_utils import setup_gpu

class GridSearch:
    """
    Class for performing grid search of hyperparameters
    """
    
    def __init__(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        image_size: int = 224,
        results_dir: Optional[str] = None
    ):
        """
        Initialize GridSearch
        
        Args:
            param_grid: Dictionary of parameters to search
            image_size: Size of input images
            results_dir: Directory to save results
        """
        self.param_grid = param_grid or {
            'class_weight': [0.1, 0.2],
            'reg_weight': [1.7, 1.8, 1.9],
            'learning_rate': [0.0001],
            'batch_size': [96, 128],
            'dropout_rate': [0.6, 0.7],
            'epochs': [25],
            'early_stopping_patience': [7],
            'reduce_lr_patience': [4],
            'lr_decay_rate': [0.9]
        }
        self.IMAGE_SIZE = image_size
        
        # Create results directory
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = results_dir or f"grid_search_results_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        self.results: List[Dict[str, Any]] = []
        self.best_model: Optional[Dict[str, Any]] = None

    def run_grid_search(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset
    ) -> Dict[str, Any]:
        """
        Execute grid search over all parameter combinations
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Dict with best model results
        """
        self.create_results_file()
        total_combinations = self.calculate_total_combinations()
        current_combination = 0
        
        best_val_loss = float('inf')
        
        try:
            # Generate all parameter combinations
            combinations = self._generate_combinations()
            
            # Iterate over combinations
            for params in combinations:
                current_combination += 1
                logging.info(f"\n{'='*50}")
                logging.info(f"Starting combination {current_combination}/{total_combinations}")
                logging.info(f"Parameters: {params}")
                
                # Update combination ID
                params['combination_id'] = current_combination
                
                try:
                    # Train model with current parameters
                    with tf.device('/GPU:0'):
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
                        
                        # Build and compile model
                        face_detection.build_model(
                            input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
                            use_batch_norm=True,
                            dropout_rate=params['dropout_rate']
                        )
                        face_detection.compile()

                        # Train model
                        logging.info("Starting training...")
                        history, model_dir = face_detection.train(
                            train_dataset,
                            val_dataset
                        )

                        # Evaluate model
                        logging.info("Evaluating model...")
                        evaluation_results = face_detection.evaluateModel(
                            test_dataset,
                            viualize_results=True
                        )
                        
                        # Collect and save results
                        results = self.collect_results(
                            history,
                            model_dir,
                            params
                        )

                        logging.info("\nTraining Results:")
                        logging.info(f"Training Metrics:")
                        logging.info(f"- Classification Accuracy: {results['final_train_class_accuracy']:.4f}")
                        logging.info(f"- Classification Loss: {results['final_train_class_loss']:.4f}")
                        logging.info(f"- Regression MAE: {results['final_train_reg_mae']:.4f}")
                        logging.info(f"- Total Loss: {results['final_train_total_loss']:.4f}")
                        
                        logging.info(f"\nValidation Metrics:")
                        logging.info(f"- Classification Accuracy: {results['final_val_class_accuracy']:.4f}")
                        logging.info(f"- Classification Loss: {results['final_val_class_loss']:.4f}")
                        logging.info(f"- Regression MAE: {results['final_val_reg_mae']:.4f}")
                        logging.info(f"- Total Loss: {results['final_val_total_loss']:.4f}")
                        
                        # Add evaluation results
                        results['evaluation'] = evaluation_results
                        self.save_results(results)
                        
                        # Update best model if necessary
                        if results['best_val_total_loss'] < best_val_loss:
                            best_val_loss = results['best_val_total_loss']
                            self.best_model = results
                            logging.info(f"\nNew Best Model Found!")
                            logging.info(f"Improvements:")
                            logging.info(f"- Classification Accuracy: {results['best_val_class_accuracy']:.4f}")
                            logging.info(f"- Classification Loss: {results['best_val_class_loss']:.4f}")
                            logging.info(f"- Regression MAE: {results['best_val_reg_mae']:.4f}")
                            logging.info(f"- Total Loss: {results['best_val_total_loss']:.4f}")
                            logging.info(f"Model saved in: {model_dir}")

                        logging.info(f"\nTraining Time: {results['training_time']:.2f} minutes")
                        logging.info(f"Epochs Trained: {results['epochs_trained']}/{params['epochs']}")
                        logging.info(f"Early Stopping: {'Yes' if results['stopped_early'] else 'No'}")
                        
                        # Clear session to free memory
                        tf.keras.backend.clear_session()
                    
                except Exception as e:
                    logging.error(f"Error in combination {current_combination}: {e}")
                    logging.error("Skipping to next combination...")
                    continue
                
            if self.best_model:
                logging.info("\n" + "="*50)
                logging.info("Grid Search Completed")
                logging.info("\nBest Model Summary:")
                logging.info(f"Combination ID: {self.best_model['combination_id']}")
                logging.info("\nParameters:")
                for param, value in {k: self.best_model[k] for k in self.param_grid.keys()}.items():
                    logging.info(f"- {param}: {value}")
                logging.info("\nBest Metrics:")
                logging.info(f"- Classification Accuracy: {self.best_model['best_val_class_accuracy']:.4f}")
                logging.info(f"- Classification Loss: {self.best_model['best_val_class_loss']:.4f}")
                logging.info(f"- Regression MAE: {self.best_model['best_val_reg_mae']:.4f}")
                logging.info(f"- Total Loss: {self.best_model['best_val_total_loss']:.4f}")
                
                return self.best_model
            else:
                logging.warning("No successful models found")
                return {}
                
        except Exception as e:
            logging.error(f"Error in grid search: {e}")
            raise
        finally:
            logging.info("\nGrid search process finished")

    def setup_logging(self) -> None:
        """Configure specific logging for this grid search"""
        log_file = os.path.join(self.results_dir, 'grid_search.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)

    def create_results_file(self) -> None:
        """Create CSV file for results"""
        columns = [
            'combination_id',
            'class_weight', 'reg_weight', 'learning_rate', 'batch_size',
            'dropout_rate', 'epochs', 'early_stopping_patience',
            'reduce_lr_patience', 'lr_decay_rate',
            'final_train_total_loss', 'final_train_class_loss', 'final_train_reg_loss',
            'final_train_class_accuracy', 'final_train_reg_mae',
            'final_val_total_loss', 'final_val_class_loss', 'final_val_reg_loss',
            'final_val_class_accuracy', 'final_val_reg_mae',
            'best_train_total_loss', 'best_train_class_loss', 'best_train_reg_loss',
            'best_train_class_accuracy', 'best_train_reg_mae',
            'best_val_total_loss', 'best_val_class_loss', 'best_val_reg_loss',
            'best_val_class_accuracy', 'best_val_reg_mae',
            'training_time', 'epochs_trained', 'stopped_early',
            'model_dir'
        ]
        
        self.results_df = pd.DataFrame(columns=columns)
        csv_path = os.path.join(self.results_dir, 'grid_search_results.csv')
        self.results_df.to_csv(csv_path, index=False)
        logging.info(f"Created results file: {csv_path}")

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        return combinations

    def collect_results(self, history: Any, model_dir: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect training results"""
        results = params.copy()

        # Training metrics
        results['final_train_total_loss'] = history.history['total_loss'][-1]
        results['final_train_class_loss'] = history.history['class_loss'][-1]
        results['final_train_reg_loss'] = history.history['reg_loss'][-1]
        results['final_train_class_accuracy'] = history.history['class_accuracy'][-1]
        results['final_train_reg_mae'] = history.history['reg_mae'][-1]

        # Validation metrics
        results['final_val_total_loss'] = history.history['val_total_loss'][-1]
        results['final_val_class_loss'] = history.history['val_class_loss'][-1]
        results['final_val_reg_loss'] = history.history['val_reg_loss'][-1]
        results['final_val_class_accuracy'] = history.history['val_class_accuracy'][-1]
        results['final_val_reg_mae'] = history.history['val_reg_mae'][-1]

        # Best metrics
        results['best_train_total_loss'] = min(history.history['total_loss'])
        results['best_train_class_loss'] = min(history.history['class_loss'])
        results['best_train_reg_loss'] = min(history.history['reg_loss'])
        results['best_train_class_accuracy'] = max(history.history['class_accuracy'])
        results['best_train_reg_mae'] = min(history.history['reg_mae'])

        results['best_val_total_loss'] = min(history.history['val_total_loss'])
        results['best_val_class_loss'] = min(history.history['val_class_loss'])
        results['best_val_reg_loss'] = min(history.history['val_reg_loss'])
        results['best_val_class_accuracy'] = max(history.history['val_class_accuracy'])
        results['best_val_reg_mae'] = min(history.history['val_reg_mae'])

        # Training info
        results['training_time'] = self.get_training_time(model_dir)
        results['epochs_trained'] = len(history.history['total_loss'])
        results['stopped_early'] = results['epochs_trained'] < params['epochs']
        results['model_dir'] = model_dir

        # Save training curves
        self.save_training_curves(history, params['combination_id'])

        return results

    def save_training_curves(self, history: Any, combination_id: int) -> None:
        """Save training curves data"""
        curves_data = {
            'epochs': list(range(1, len(history.history['total_loss']) + 1)),
            'training': {
                'total_loss': history.history['total_loss'],
                'class_loss': history.history['class_loss'],
                'reg_loss': history.history['reg_loss'],
                'class_accuracy': history.history['class_accuracy'],
                'reg_mae': history.history['reg_mae']
            },
            'validation': {
                'total_loss': history.history['val_total_loss'],
                'class_loss': history.history['val_class_loss'],
                'reg_loss': history.history['val_reg_loss'],
                'class_accuracy': history.history['val_class_accuracy'],
                'reg_mae': history.history['val_reg_mae']
            }
        }

        curves_file = os.path.join(
            self.results_dir, 
            f"curves_combination_{combination_id}.json"
        )
        with open(curves_file, 'w') as f:
            json.dump(curves_data, f, indent=4)

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results to CSV and JSON"""
        # Save to CSV
        results_df = pd.DataFrame([results])
        results_df.to_csv(
            os.path.join(self.results_dir, 'grid_search_results.csv'),
            mode='a',
            header=False,
            index=False
        )

        # Save detailed results to JSON
        with open(
            os.path.join(
                self.results_dir, 
                f"combination_{results['combination_id']}.json"
            ), 
            'w'
        ) as f:
            json.dump(results, f, indent=4)

    def get_training_time(self, model_dir: str) -> Optional[float]:
        """Get training time from file"""
        try:
            with open(os.path.join(model_dir, 'training_time.txt'), 'r') as f:
                time_str = f.read()
                return float(time_str.split(":")[1].strip().split()[0])
        except:
            return None

    def calculate_total_combinations(self) -> int:
        """Calculate total number of parameter combinations"""
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total

if __name__ == "__main__":
    try:
        # Configuration
        AUTOTUNE = tf.data.AUTOTUNE
        data = DataPreprocessor("Data/Data.csv", img_size=224)
        
        logging.info("Loading data...")
        print(data.data.head())
        
        setup_gpu()
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = data.build_datasets()
        
        # Run grid search
        grid_search = GridSearch()
        best_model = grid_search.run_grid_search(
            train_dataset,
            val_dataset,
            test_dataset
        )
        
        logging.info(f"Grid search completed. Best model: {best_model}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise