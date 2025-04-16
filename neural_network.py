import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Add, Input, Concatenate
from tensorflow.keras.models import Model

# Define a custom model class
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Initialize a Sequential model
        self.model = tf.keras.Sequential()
        
        # Add initial convolution and pooling
        self.model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        
        # Add a residual block by calling the function
        self._add_residual_block(filters=64, kernel_size=3, stride=1)
        
        # Add more layers after the residual block
        self.model.add(layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))  # Output layer for 10 classes
    
    def _add_residual_block(self, filters, kernel_size=3, stride=1):
        # Define the main path
        self.model.add(layers.Conv1D(filters, kernel_size, strides=stride, padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Conv1D(filters, kernel_size, strides=1, padding='same'))
        self.model.add(layers.BatchNormalization())
        
        # Define the shortcut path
        shortcut = tf.keras.Sequential()
        if stride != 1 or filters != self.model.output_shape[-1]:
            shortcut.add(layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same'))
        else:
            shortcut.add(layers.Lambda(lambda x: x))  # Identity function
        
        # Add the shortcut and the main path together
        self.model.add(layers.Lambda(lambda x: x + shortcut(x)))
        self.model.add(layers.ReLU())
    
    def call(self, inputs):
         # Split the input into real and imaginary parts
         real_part = tf.math.real(inputs)
         imag_part = tf.math.imag(inputs)
         combined_input = tf.stack([real_part, imag_part], axis=-1)
         return self.model(combined_input)
        
        # Stack the real and imaginary parts along a new axis (last dimension)
        
        # Pass the combined input to the sequential model
        