import tensorflow as tf

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Successfully configured GPU")
            return True
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
            return False
    return False
