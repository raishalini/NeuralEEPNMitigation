import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1d = None
        self.conv_output = None

    def _build_layers(self, input_shape):
        l2_reg = regularizers.l2(0.01)
        self.conv1d = layers.Conv1D(
            filters=64,
            kernel_size=50,
            padding='same',
            activation='relu',
            dilation_rate=2,
            kernel_regularizer=l2_reg
        )
        
        # Residual Block 1
        self.res_conv1 = layers.Conv1D(64, kernel_size=25, padding='same', activation='relu', dilation_rate=1, kernel_regularizer=l2_reg)
        self.res_bn1 = layers.BatchNormalization()
        self.res_conv2 = layers.Conv1D(64, kernel_size=25, padding='same', dilation_rate=1, kernel_regularizer=l2_reg)
        self.res_bn2 = layers.BatchNormalization()

        # Additional Residual Block 2 (Increase Depth)
        self.res_conv3 = layers.Conv1D(64, kernel_size=15, padding='same', activation='relu', dilation_rate=1, kernel_regularizer=l2_reg)
        self.res_bn3 = layers.BatchNormalization()
        self.res_conv4 = layers.Conv1D(64, kernel_size=15, padding='same', dilation_rate=1, kernel_regularizer=l2_reg)
        self.res_bn4 = layers.BatchNormalization()

        # Additional Residual Block 3 (Further Increase Depth)
        self.res_conv5 = layers.Conv1D(64, kernel_size=9, padding='same', activation='relu', dilation_rate=1, kernel_regularizer=l2_reg)
        self.res_bn5 = layers.BatchNormalization()
        self.res_conv6 = layers.Conv1D(64, kernel_size=9, padding='same', dilation_rate=1, kernel_regularizer=l2_reg)
        self.res_bn6 = layers.BatchNormalization()

        self.extra_conv1 = layers.Conv1D(128, kernel_size=15, padding='same', activation='relu', dilation_rate=1, kernel_regularizer=l2_reg)
        self.extra_conv2 = layers.Conv1D(128, kernel_size=15, padding='same', activation='relu', dilation_rate=1, kernel_regularizer=l2_reg)

        # Output Layer
        self.conv_output = layers.Conv1D(filters=2, kernel_size=15, padding='same', activation=None)

        # Adding dropout for regularization
        self.dropout = layers.Dropout(0.2)

    def call(self, inputs, training=False):
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)
        
        combined_input = tf.stack([real_part, imag_part], axis=-1)

        if self.conv1d is None:
            self._build_layers(inputs.shape[-1])

        x = self.conv1d(combined_input)

        # Residual Block 1
        shortcut = x
        x = self.res_conv1(x)
        x = self.res_bn1(x, training=training)
        x = self.res_conv2(x)
        x = self.res_bn2(x, training=training)
        x = layers.add([x, shortcut])

        # Residual Block 2
        shortcut = x
        x = self.res_conv3(x)
        x = self.res_bn3(x, training=training)
        x = self.res_conv4(x)
        x = self.res_bn4(x, training=training)
        x = layers.add([x, shortcut])

        # Residual Block 3
        shortcut = x
        x = self.res_conv5(x)
        x = self.res_bn5(x, training=training)
        x = self.res_conv6(x)
        x = self.res_bn6(x, training=training)
        x = layers.add([x, shortcut])

        # Adding extra convolutional layers
        x = self.extra_conv1(x)
        x = self.dropout(x, training=training)
        x = self.extra_conv2(x)

        x = self.conv_output(x)

        # Split into real and imaginary parts and return as complex output
        real_output, imag_output = tf.split(x, num_or_size_splits=2, axis=-1)
        real_output = tf.squeeze(real_output, axis=-1)
        imag_output = tf.squeeze(imag_output, axis=-1)

        complex_output = tf.complex(real_output, imag_output)

        return complex_output

    @tf.function
    def call_graph(self, inputs, training=False):
        return self.call(inputs, training)
        
    def build_graph(self):
        self.call_graph = tf.function(self.call)


# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1d = None
#         self.conv_output = None

#     def _build_layers(self, input_shape):
#         self.conv1d = layers.Conv1D(
#             filters=64,
#             kernel_size=25,
#             padding='same',
#             activation='relu',
#             dilation_rate=2,
#         )
        
#         # Residual block
#         self.res_conv1 = layers.Conv1D(64, kernel_size=25, padding='same', activation='relu', dilation_rate=1)
#         self.res_bn1 = layers.BatchNormalization()
#         self.res_conv2 = layers.Conv1D(64, kernel_size=25, padding='same', dilation_rate=1)
#         self.res_bn2 = layers.BatchNormalization()
        
#         self.extra_conv1 = layers.Conv1D(128, kernel_size=15, padding='same', activation='relu', dilation_rate=1)
#         self.extra_conv2 = layers.Conv1D(128, kernel_size=15, padding='same', activation='relu', dilation_rate=1)
        
#         # Output layer
#         self.conv_output = layers.Conv1D(filters=2, kernel_size=15, padding='same', activation=None)

#         # Adding dropout for regularization
#         self.dropout = layers.Dropout(0.2)

#     def call(self, inputs, training=False):
#         real_part = tf.math.real(inputs)
#         imag_part = tf.math.imag(inputs)
        
#         combined_input = tf.stack([real_part, imag_part], axis=-1)

#         if self.conv1d is None:
#             self._build_layers(inputs.shape[-1])

#         x = self.conv1d(combined_input)

#         # Residual block
#         shortcut = x
#         x = self.res_conv1(x)
#         x = self.res_bn1(x, training=training)
#         x = self.res_conv2(x)
#         x = self.res_bn2(x, training=training)
#         x = layers.add([x, shortcut])

#         # Adding extra convolutional layers
#         x = self.extra_conv1(x)
#         x = self.dropout(x, training=training)
#         x = self.extra_conv2(x)

#         # Final output layer
#         x = self.conv_output(x)

#         real_output, imag_output = tf.split(x, num_or_size_splits=2, axis=-1)
#         real_output = tf.squeeze(real_output, axis=-1)
#         imag_output = tf.squeeze(imag_output, axis=-1)

#         complex_output = tf.complex(real_output, imag_output)

#         return complex_output

    # @tf.function
    # def call_graph(self, inputs, training=False):
    #     return self.call(inputs, training)
        
    # def build_graph(self):
    #     self.call_graph = tf.function(self.call)
   


# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1d = None
#         self.conv_output = None

#     def _build_layers(self, input_shape):
#         self.conv1d = layers.Conv1D(
#             filters=64,
#             kernel_size=50,
#             padding='same',
#             activation='relu',
#             dilation_rate=2,
#             # input_shape=(None, input_shape)
#         )
        
#         # layers for res block
#         self.res_conv1 = layers.Conv1D(64, kernel_size=50, padding='same', activation='relu', dilation_rate=2)
#         self.res_bn1 = layers.BatchNormalization()
#         self.res_relu1 = layers.ReLU()
#         self.res_conv2 = layers.Conv1D(64, kernel_size=50, padding='same', dilation_rate=2)
#         self.res_bn2 = layers.BatchNormalization()
#         self.res_relu2 = layers.ReLU()
        
#         self.conv_output = layers.Conv1D(filters=2, kernel_size=50, padding='same', activation=None, dilation_rate=2)


#     def call(self, inputs, training=False):
#         real_part = tf.math.real(inputs)
#         imag_part = tf.math.imag(inputs)
        
#         combined_input = tf.stack([real_part, imag_part], axis=-1)
#         # print("combined_input : ", combined_input)
        
#         input_shape = inputs.shape[-1]
        
#         if self.conv1d is None:
#             self._build_layers(input_shape)
        
#         x = self.conv1d(combined_input)

#         # Residual block
#         shortcut = x
#         x = self.res_conv1(x)
#         x = self.res_bn1(x, training=training)
#         x = self.res_relu1(x)
#         x = self.res_conv2(x)
#         x = self.res_bn2(x, training=training)
        
#         x = layers.add([x, shortcut])
#         x = self.res_relu2(x)
        
#         x = self.conv_output(x)
#         # print("x : ", x)
        
#         real_output, imag_output = tf.split(x, num_or_size_splits=2, axis=-1)
        
#         real_output = tf.squeeze(real_output, axis=-1)
#         imag_output = tf.squeeze(imag_output, axis=-1)
        
#         complex_output = tf.complex(real_output, imag_output)
        
#         return complex_output