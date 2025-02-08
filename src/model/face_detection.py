import os
import time
import matplotlib.pyplot as plt
import datetime
import json
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore

@tf.keras.utils.register_keras_serializable(package="face_detection")
class FaceDetection(Model):
    def __init__(self, 
             class_weight=0.5, 
             reg_weight=1.5,
             learning_rate=0.001,
             class_loss_weight=1.0,
             bbox_loss_weight=2.0,
             epochs=20,                      
             batch_size=8,                   
             early_stopping_patience=5,       
             reduce_lr_patience=3,           
             lr_decay_rate=0.9):           
        super().__init__()
        self.base_model = None
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        self.class_loss_weight = class_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.lr_decay_rate = lr_decay_rate
        self.dropout_rate = None

        self.modelDir = None
        
        if tf.config.list_physical_devices('GPU'):
            print("Model will run on GPU")

    def build_model(self, input_shape=(224, 224, 3), use_batch_norm=True, dropout_rate=0.5):
        tf.get_logger().setLevel('ERROR')

        input_layer = Input(shape=input_shape, name='input_image')
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_layer,
            input_shape=input_shape
        )
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalMaxPooling2D(name='global_features')(x)

        class_branch = Dense(1024, 
                            kernel_regularizer=tf.keras.regularizers.l2(0.02),
                            name='class_dense1')(x)
        if use_batch_norm:
            class_branch = BatchNormalization(name='class_bn1')(class_branch)
        class_branch = Activation('relu', name='class_relu1')(class_branch)
        class_branch = Dropout(dropout_rate, name='class_dropout1')(class_branch)
        
        class_branch = Dense(512, name='class_dense2')(class_branch)
        if use_batch_norm:
            class_branch = BatchNormalization(name='class_bn2')(class_branch)
        class_branch = Activation('relu', name='class_relu2')(class_branch)
        class_branch = Dropout(dropout_rate, name='class_dropout2')(class_branch)
        
        classification_output = Dense(
            1, 
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(0.02),
            name='classification'
        )(class_branch)

        reg_branch = Dense(1024, 
                    kernel_regularizer=tf.keras.regularizers.l2(0.02),
                    name='reg_dense1')(x)
        if use_batch_norm:
            reg_branch = BatchNormalization(name='reg_bn1')(reg_branch)
        reg_branch = Activation('relu', name='reg_relu1')(reg_branch)
        reg_branch = Dropout(dropout_rate, name='reg_dropout1')(reg_branch)
        
        reg_branch = Dense(512, name='reg_dense2')(reg_branch)
        if use_batch_norm:
            reg_branch = BatchNormalization(name='reg_bn2')(reg_branch)
        reg_branch = Activation('relu', name='reg_relu2')(reg_branch)
        reg_branch = Dropout(dropout_rate, name='reg_dropout2')(reg_branch)
        
        regression_output = Dense(4, activation='sigmoid', name='bounding_box')(reg_branch)

        self.dropout_rate = dropout_rate

        self.base_model = Model(
            inputs=input_layer,
            outputs=[classification_output, regression_output],
            name='face_detection'
        )

    def compile(self, optimizer=None, classloss=None, regressloss=None):
        super().compile()  
        
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.class_loss = classloss if classloss is not None else tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        self.reg_loss = regressloss if regressloss is not None else regression_loss
        
        self.class_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.reg_mae = tf.keras.metrics.MeanAbsoluteError()

    @classmethod
    def compile_from_config(cls, config):
        optimizer = tf.keras.utils.deserialize_keras_object(config["optimizer"])
        class_loss = tf.keras.utils.deserialize_keras_object(config["class_loss"])
        reg_loss = tf.keras.utils.deserialize_keras_object(config["reg_loss"])
        return optimizer, class_loss, reg_loss
    
    def get_compile_config(self):
        return {
            "optimizer": self.optimizer,
            "class_loss": self.class_loss,
            "reg_loss": self.reg_loss,
        }
    
    def get_config(self):
        return {
            "base_model": self.base_model.get_config(),
            "class_weight": self.class_weight,
            "reg_weight": self.reg_weight,
            "optimizer": tf.keras.utils.serialize_keras_object(self.optimizer),
            "class_loss": tf.keras.utils.serialize_keras_object(self.class_loss),
            "reg_loss": tf.keras.utils.serialize_keras_object(self.reg_loss)
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        base_model = tf.keras.Model.from_config(
            config["base_model"], 
            custom_objects=custom_objects
        )
        instance = cls(
            base_model=base_model,
            class_weight=config["class_weight"],
            reg_weight=config["reg_weight"]
        )
        instance.compile(
            optimizer=tf.keras.utils.deserialize_keras_object(
                config["optimizer"], 
                custom_objects=custom_objects
            ),
            classloss=tf.keras.utils.deserialize_keras_object(
                config["class_loss"], 
                custom_objects=custom_objects
            ),
            regressloss=tf.keras.utils.deserialize_keras_object(
                config["reg_loss"], 
                custom_objects=custom_objects
            )
        )
        return instance
    
    def call(self, X):
        return self.base_model(X)
    
    class MetricsResetCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.model.class_accuracy.reset_state()
            self.model.reg_mae.reset_state()


    def create_callbacks(self, model_dir):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_total_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_weights.weights.h5'),
                monitor='val_total_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                mode='min'
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: self.learning_rate * (self.lr_decay_rate ** epoch),
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(model_dir, 'training_log.csv'),
                separator=',',
                append=True
            ),
            self.MetricsResetCallback(),
            self.TrainingTimeCallback(model_dir)
        ]
        return callbacks
    
    class TrainingTimeCallback(tf.keras.callbacks.Callback):
        def __init__(self, model_dir):
            super().__init__()
            self.model_dir = model_dir
            
        def on_train_begin(self, logs={}):
            self.start_time = time.time()
            
        def on_train_end(self, logs={}):
            training_time = time.time() - self.start_time
            print(f"\nTotal training time: {training_time/60:.2f} minutes")
            
            with open(os.path.join(self.model_dir, 'training_time.txt'), 'w') as f:
                f.write(f"Training time: {training_time/60:.2f} minutes")
    
    def train(self, train_dataset, val_dataset, model_dir=None, save_model_architecture=True):
        if model_dir is None:
            model_dir = f"models/face_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.modelDir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            print("\nInitiating training...")
            print(f"Epochs: {self.epochs}")
            print(f"Batch Size: {self.batch_size}")
            print(f"Learning Rate: {self.learning_rate}")
            print(f"Directory: {model_dir}")
            print(f"Class Weight: {self.class_weight}")
            print(f"Regression Weight: {self.reg_weight}")
            print(f"Early Stopping Patience: {self.early_stopping_patience}")
            print(f"Reduce LR Patience: {self.reduce_lr_patience}")
            print(f"LR Decay Rate: {self.lr_decay_rate}")
            print(f"Dropout Rate: {self.dropout_rate}")

            parameters = {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": float(self.learning_rate),  
                "class_weight": float(self.class_weight),
                "reg_weight": float(self.reg_weight),
                "class_loss_weight": float(self.class_loss_weight),
                "bbox_loss_weight": float(self.bbox_loss_weight),
                "early_stopping_patience": self.early_stopping_patience,
                "reduce_lr_patience": self.reduce_lr_patience,
                "lr_decay_rate": float(self.lr_decay_rate),
                "dropout_rate": float(self.dropout_rate)
            }
        
            with open(os.path.join(model_dir, 'parameters.json'), 'w') as f:
                json.dump(parameters, f, indent=4)
            
            history = self.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                initial_epoch=0,
                callbacks=self.create_callbacks(model_dir),
                verbose=1
            )
        
            final_model_path = os.path.join(model_dir, 'face_detection_final.weights.h5')
            self.save_weights(final_model_path)
         
            converted_history = {key: [float(v) for v in value] 
                            for key, value in history.history.items()}
            with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
                json.dump(converted_history, f, indent=4)

            if save_model_architecture:
                plot_model(
                    self.base_model,
                    to_file=os.path.join(model_dir, 'model_architecture.png'),
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir='TB'
                )

            #self.plot_training_history(history, model_dir)
            
            return history, model_dir
            
        except Exception as e:
            print(f"\nError during training: {e}")
            raise e
        finally:
            print("\nTraining finished.")
        
    @tf.function 
    def train_step(self, batch):
        X, y = batch
        
        y_class = tf.cast(tf.reshape(y[0], (-1, 1)), tf.float32)
        y_bbox = tf.cast(y[1], tf.float32)
        
        with tf.GradientTape() as tape:
            classes, coords = self.base_model(X, training=True)
        
            class_loss = self.class_loss(y_class, classes)
            reg_loss = self.reg_loss(y_bbox, coords)
            total_loss = (self.reg_weight * reg_loss + 
                         self.class_weight * class_loss)
        
        gradients = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

        self.class_accuracy.update_state(y_class, classes)
        self.reg_mae.update_state(y_bbox, coords)
        
        return {
            "total_loss": total_loss,
            "class_loss": class_loss,
            "reg_loss": reg_loss,
            "class_accuracy": self.class_accuracy.result(),
            "reg_mae": self.reg_mae.result()
        }
    
    @tf.function 
    def test_step(self, batch):
        X, y = batch
        
        y_class = tf.cast(tf.reshape(y[0], (-1, 1)), tf.float32)
        y_bbox = tf.cast(y[1], tf.float32)

        classes, coords = self.base_model(X, training=False)
        
        class_loss = self.class_loss(y_class, classes)
        reg_loss = self.reg_loss(y_bbox, coords)
        total_loss = (self.reg_weight * reg_loss + 
                     self.class_weight * class_loss)
        
        self.class_accuracy.update_state(y_class, classes)
        self.reg_mae.update_state(y_bbox, coords)
        
        return {
            "total_loss": total_loss,
            "class_loss": class_loss,
            "reg_loss": reg_loss,
            "class_accuracy": self.class_accuracy.result(),
            "reg_mae": self.reg_mae.result()
        }
    
    @tf.function
    def predict_step(self, data):
        return self.base_model(data, training=False)

    def predict(self, image_path, threshold=0.5):
        if isinstance(image_path, str):
            img = tf.io.read_file(image_path)
            img = tf.io.decode_jpeg(img, channels=3)
        else:
            img = image_path
        
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)

        with tf.device('/GPU:0'):
            class_pred, bbox_pred = self.predict_step(img)

        class_prob = float(class_pred[0][0])
        has_face = class_prob >= threshold
        
        if has_face:
            x1, y1, x2, y2 = [float(coord) for coord in bbox_pred[0]]
        else:
            x1 = y1 = x2 = y2 = 0.0

        return {
            'has_face': has_face,
            'confidence': class_prob,
            'bbox': [x1, y1, x2, y2] if has_face else None
        }
    
    def plot_training_history(self, history, model_dir):
        metrics = ['total_loss', 'class_loss', 'reg_loss', 'class_accuracy', 'reg_mae']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            if idx < len(axes) and metric in history.history:
                axes[idx].plot(history.history[metric], label='Training')
                axes[idx].plot(history.history[f'val_{metric}'], label='Validation')
                axes[idx].set_title(f'{metric.replace("_", " ").title()}')
                axes[idx].set_xlabel('Epoch')
                axes[idx].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_plots.png'))
        plt.show()
        plt.close()

    def evaluateModel(self, test_dataset, viualize_results=False, model_dir=None):
        if model_dir is None:
            model_dir = self.modelDir
        try:
            test_loss = tf.keras.metrics.Mean()
            test_class_loss = tf.keras.metrics.Mean()
            test_reg_loss = tf.keras.metrics.Mean()
            test_class_accuracy = tf.keras.metrics.BinaryAccuracy()
            test_class_precision = tf.keras.metrics.Precision()
            test_class_recall = tf.keras.metrics.Recall()
            test_reg_mae = tf.keras.metrics.MeanAbsoluteError()
            test_reg_mse = tf.keras.metrics.MeanSquaredError()

            for batch in test_dataset:
                X, y = batch
                y_class = tf.cast(tf.reshape(y[0], (-1, 1)), tf.float32)
                y_bbox = tf.cast(y[1], tf.float32)
                
                classes, coords = self.base_model(X, training=False)
                
                class_loss = self.class_loss(y_class, classes)
                reg_loss = self.reg_loss(y_bbox, coords)
                total_loss = (self.reg_weight * reg_loss + 
                            self.class_weight * class_loss)
                
                test_loss.update_state(total_loss)
                test_class_loss.update_state(class_loss)
                test_reg_loss.update_state(reg_loss)
                test_class_accuracy.update_state(y_class, classes)
                test_class_precision.update_state(y_class, classes)
                test_class_recall.update_state(y_class, classes)
                test_reg_mae.update_state(y_bbox, coords)
                test_reg_mse.update_state(y_bbox, coords)

            precision = test_class_precision.result()
            recall = test_class_recall.result()
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
            
            results = {
                'test_total_loss': float(test_loss.result()),
          
                'test_class_loss': float(test_class_loss.result()),
                'test_class_accuracy': float(test_class_accuracy.result()),
                'test_class_precision': float(precision),
                'test_class_recall': float(recall),
                'test_f1_score': float(f1_score),
                
                'test_reg_loss': float(test_reg_loss.result()),
                'test_reg_mae': float(test_reg_mae.result()),
                'test_reg_mse': float(test_reg_mse.result()),
                'test_reg_rmse': float(tf.sqrt(test_reg_mse.result()))
            }

            with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
                json.dump(results, f, indent=4)
            
            if viualize_results:
                self.plot_evaluation_results(results, model_dir)
            
            return results
            
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            raise e

    def plot_evaluation_results(self, results, model_dir):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        class_metrics = {
            'Accuracy': results['test_class_accuracy'],
            'Precision': results['test_class_precision'],
            'Recall': results['test_class_recall'],
            'F1 Score': results['test_f1_score']
        }
        
        bars1 = ax1.bar(class_metrics.keys(), class_metrics.values())
        ax1.set_title('Classification Metrics')
        ax1.set_ylim(0, 1)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
  
        reg_metrics = {
            'MAE': results['test_reg_mae'],
            'MSE': results['test_reg_mse'],
            'RMSE': results['test_reg_rmse']
        }
        
        bars2 = ax2.bar(reg_metrics.keys(), reg_metrics.values())
        ax2.set_title('Regression Metrics')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'evaluation_results.png'))
        plt.close()
        
def regression_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]
    h_pred = yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size