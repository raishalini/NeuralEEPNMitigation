import tensorflow as tf
import sionna as sn

# Suggestion: flen = 31, avglen = 501, fourth_power=True
class Gardner(tf.keras.layers.Layer):
  def __init__(self, flen, avglen, span_in_symbols, shaper_type, samples_per_symbol, interpolator="sinc", roll_off=0, fourth_power=True, dtype=tf.complex64):
    super().__init__(trainable=False, name="TimingRecovery/Gardner")
    self._cdtype = tf.as_dtype(dtype)
    self._rdtype = tf.as_dtype(dtype).real_dtype
    self.avglen = int(avglen) if not isinstance(avglen, int) else avglen
    self.avglenhalf = self.avglen // 2
    self.flen = int(flen) if not isinstance(flen, int) else flen
    self.flenhalf = self.flen // 2
    self.fourth_power = fourth_power
    if not interpolator.lower() in ["rrc", "rc", "sinc"]:
      raise ValueError("Unknown interpolator")
    self.interpolator = interpolator
    assert 0 <= roll_off <= 1
    self.roll_off = roll_off
    self.stack_filter = tf.expand_dims(
        tf.eye(self.flen, self.flen, dtype=self._rdtype), axis=1)
    self.upsampler = sn.signal.Upsampling(2)
    match shaper_type.lower():
      case "rrc":
        self.pulse_shaper = sn.signal.RootRaisedCosineFilter(
            span_in_symbols=span_in_symbols,
            samples_per_symbol=2, beta=roll_off,
            dtype=self._rdtype)
      case "rc":
        self.pulse_shaper = sn.signal.RaisedCosineFilter(
            span_in_symbols=span_in_symbols,
            samples_per_symbol=2, beta=roll_off,
            dtype=self._rdtype)
      case "sinc":
        self.pulse_shaper = sn.signal.SincFilter(
            span_in_symbols=span_in_symbols,
            samples_per_symbol=2, dtype=self._rdtype)
      case _:
        raise ValueError("Pulse shaper type")

  def __call__(self, inputs):
    pre_tr = self.pulse_shaper(
        self.upsampler(inputs))[
        :, self.pulse_shaper.length // 2 - 1: -(self.pulse_shaper.length // 2)]
    timing_errors = self.estimate_timing_error(pre_tr)
    return self.compensate_timing_error(inputs, timing_errors), timing_errors

  # def timing_error(self, inputs):
  #   pre_tr = self.pulse_shaper(
  #       self.upsampler(inputs))[
  #       :, self.pulse_shaper.length // 2 - 1: -(self.pulse_shaper.length // 2)]
  #   timing_errors = self.estimate_timing_error(pre_tr)
  #   return timing_errors

  def estimate_timing_error(self, inputs):
    x = inputs
    if self.fourth_power:
      x = tf.math.square(tf.math.abs(inputs))
    err = tf.math.real((x[:, 0:-2:2] - x[:, 2::2]) * tf.math.conj(x[:, 1:-1:2]))
    err_avg = tf.nn.conv1d(
        err[..., tf.newaxis],
        tf.ones((self.avglen, 1, 1), self._rdtype),
        1, "VALID")[..., 0] / self.avglenhalf

    return err_avg

  def build_filters(self, timing_errors):
    base_time = tf.range(-self.flenhalf, self.flenhalf + 1, dtype=self._rdtype)
    time_shifts = timing_errors[..., tf.newaxis]
    used_time_shifts = base_time - time_shifts
    filters = tf.experimental.numpy.sinc(used_time_shifts)

    return filters

  def compensate_timing_error(self, inputs, errors):
    filters = tf.cast(self.build_filters(errors), self._cdtype)
    stacked_symbols = tf.complex(
        tf.nn.conv1d(
            tf.expand_dims(
                tf.math.real(inputs),
                axis=-1),
            self.stack_filter,
            1,
            "SAME"),
        tf.nn.conv1d(
            tf.expand_dims(
                tf.math.imag(inputs),
                axis=-1),
            self.stack_filter,
            1,
            "SAME"),
    )
    shifted_symbols = tf.einsum("ijk,ijk->ij",
                                stacked_symbols
                                [:, self.avglenhalf: -self.avglenhalf, :],
                                filters)[..., self.flenhalf: -self.flenhalf]
    return shifted_symbols

