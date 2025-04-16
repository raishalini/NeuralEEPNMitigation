import tensorflow as tf
import numpy as np

class PhaseNoise:
    def __init__(self):
        self.always_reseed = not True

    def __call__(self, y, linewidth, t_norm):
        with tf.name_scope("phasenoise") as name:
          input_shape = tf.shape(y)
          stddev = tf.sqrt(2 * np.pi * linewidth * t_norm)
          if self.always_reseed:
            noise = generate_noise(
                input_shape, stddev)
          else:
            noise_bins = tf.random.Generator.from_non_deterministic_state().normal(
                shape=input_shape, mean=0, stddev=stddev,
                dtype=y.dtype.real_dtype)
            # noise = start + tf.math.cumsum(noise_bins, axis=-1)
            noise = tf.math.cumsum(noise_bins, axis=-1)
          return tf.math.multiply(y, tf.math.exp(tf.complex(tf.zeros_like(noise), noise)))

def generate_noise(self, noise_shape, stddev):
  return tf.math.cumsum(
      tf.random.Generator.from_non_deterministic_state().normal(
          shape=noise_shape, mean=0, stddev=stddev,
          dtype=tf.float64),
      axis=-1)