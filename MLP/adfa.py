import tensorflow as tf
from tensorflow import keras

# Verificar se há GPU disponível
print("GPUs Disponíveis:", tf.config.list_physical_devices('GPU'))

# Criar modelo com TensorFlow (executa automaticamente na GPU se disponível)
model = keras.Sequential([
    keras.layers.Dense(12, activation='sigmoid', input_shape=(1,)),
    keras.layers.Dense(1, activation='sigmoid')
])