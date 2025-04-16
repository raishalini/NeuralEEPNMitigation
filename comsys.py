import tensorflow as tf
from sionna.signal import Upsampling, RootRaisedCosineFilter, Downsampling
from sionna.utils import QAMSource, expand_to_rank, complex_normal
from sionna import utils as snutils
from sionna.channel.optical import SSFM
from timing_recovery import Gardner
from phasenoise import PhaseNoise

class Transmitter:
    def __init__(self, num_bits_per_symbol, batch_size, num_symbols, samples_per_symbol, beta, span_in_symbols):
        self.qam_source = QAMSource(num_bits_per_symbol)
        self.upsampling = Upsampling(samples_per_symbol)
        self.rcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
        self.batch_size = batch_size
        self.num_symbols = num_symbols
        self.samples_per_symbol = samples_per_symbol

    def generate_qam_symbols(self):
        return self.qam_source([self.batch_size, self.num_symbols])

    def upsample(self, x):
        return self.upsampling(x)

    def apply_rcf(self, x_us):
        return self.rcf(x_us)

    def pad_signal(self, x_rcf):
        sample_amt = tf.math.pow(2, tf.cast(tf.math.ceil(tf.math.log(tf.cast(x_rcf.shape[-1], tf.float32)) /            tf.math.log(2.0)) + 1, tf.int32))
        padding_amt = int(sample_amt - x_rcf.shape[-1])
        padding_left = padding_amt // 2
        padding_right = padding_amt - padding_left
        return tf.pad(x_rcf, ((0, 0), (padding_left, padding_right))), padding_left, padding_right

# class Channel:
#     def __init__(self, alpha, beta_2, f_c, length_sp, t_norm, dtype):
#         self.alpha = alpha
#         self.beta_2 = beta_2
#         self.f_c = f_c
#         self.t_norm = t_norm
#         self.dtype = dtype
#         self._length_sp = length_sp  # Internal variable for length
        
#         # Initialize SSFM objects with initial link distance
#         self.ss_fn = SSFM(
#             alpha=self.alpha, beta_2=self.beta_2, f_c=self.f_c, length=self._length_sp,
#             with_amplification=False, with_attenuation=False,
#             with_dispersion=True, with_nonlinearity=False,
#             half_window_length=100, dtype=self.dtype, t_norm=self.t_norm
#         )
#         self.ss_fn_cdc = SSFM(
#             alpha=self.alpha, beta_2=-self.beta_2, f_c=self.f_c, length=self._length_sp,
#             with_amplification=False, with_attenuation=False,
#             with_dispersion=True, with_nonlinearity=False,
#             dtype=self.dtype, t_norm=self.t_norm
#         )

#     @property
#     def length_sp(self):
#         return self._length_sp

#     @length_sp.setter
#     def length_sp(self, value):
#         self._length_sp = value
#         # Directly update the length in both SSFM objects
#         self.ss_fn._length = value
#         self.ss_fn_cdc._length = value

#     def transmit(self, x_rcf_padded):
#         return self.ss_fn(x_rcf_padded)

#     def compensate_dispersion(self, y_mf):
#         return self.ss_fn_cdc(y_mf)

class Channel:
    def __init__(self, alpha, beta_2, f_c, length_sp, t_norm, dtype):
        self.alpha = alpha  # Store alpha
        self.beta_2 = beta_2  # Store beta_2
        self.f_c = f_c  # Store f_c
        self.t_norm = t_norm  # Store t_norm
        self.dtype = dtype  # Store dtype
        self.length_sp = length_sp  # Internal variable for length
        self.ss_fn = SSFM(
            alpha=alpha, beta_2=beta_2, f_c=f_c, length=length_sp,
            with_amplification=False, with_attenuation=False,
            with_dispersion=True, with_nonlinearity=False,
            half_window_length=100, dtype=dtype, t_norm=t_norm
        )
        self.ss_fn_cdc = SSFM(
            alpha=alpha, beta_2=-beta_2, f_c=f_c, length=length_sp,
            with_amplification=False, with_attenuation=False,
            with_dispersion=True, with_nonlinearity=False,
            dtype=dtype, t_norm=t_norm
        )
    
    def transmit(self, x_rcf_padded):
        signal = self.ss_fn(x_rcf_padded)
        signal = self._add_complex_noise(signal)
        return signal

    def _add_complex_noise(self, signal):
        no = 10 ** (-14 / 10)

        x = signal

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(tf.shape(x), dtype=x.dtype)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, tf.rank(x), axis=-1)

        # Apply variance scaling
        no = tf.cast(no, self.dtype)
        noise *= tf.cast(tf.sqrt(no), noise.dtype)

        # Add noise to input
        y = x + noise

        return y
        
    def compensate_dispersion(self, y_mf):
        return self.ss_fn_cdc(y_mf)


class Receiver:
    def __init__(self, linewidth, t_norm, samples_per_symbol, rcf):
        self.linewidth = linewidth
        self.t_norm = t_norm
        self.samples_per_symbol = samples_per_symbol
        self.rcf = rcf
        self.gardner = Gardner(flen=31, avglen=501, span_in_symbols=128, shaper_type="rrc", samples_per_symbol=samples_per_symbol, interpolator="sinc", dtype=tf.complex64)
        self.phase_noise = PhaseNoise()

    # def add_phase_noise(self, y, add_phase_noise_func):
    #     return add_phase_noise_func(y, self.linewidth, self.t_norm)

    def add_phase_noise(self, y):
        # print(f"Current linewidth in add_phase_noise: {self.linewidth} Hz")
        return self.phase_noise(y, self.linewidth, self.t_norm)

    def matched_filter(self, y_pn):
        return self.rcf(y_pn)

    def low_pass_filter(self, y_cdc):
        
        return self.rcf(y_cdc)

    def downsample(self, y_cdc, padding_left, padding_right):
        ds = Downsampling(self.samples_per_symbol)
        return ds(y_cdc[..., 2*((self.rcf.length-1)//2) + padding_left:-2*((self.rcf.length-1)//2) - padding_right])

    def timing_recovery(self, y_ds):
        return self.gardner(y_ds)

    def normalize(self, symbols):
        power_per_symbol = tf.math.square(tf.math.abs(symbols))  # "P = ||^2" power of each symbol, dim: (batch, n_symbols)
        avg_power_per_batch = tf.math.reduce_mean(power_per_symbol, axis=-1, keepdims=True) # dim: (batch, 1) E[P] 
        normalized_symbols = symbols/tf.cast(tf.math.sqrt(avg_power_per_batch), symbols.dtype)
        return normalized_symbols

    def cpr(self, y_ds, x, dtype, mem_cut, filter):
        rot_vec = y_ds * tf.math.conj(x) / (x * tf.math.conj(x))
        rot_vec = snutils.expand_to_rank(rot_vec, 4, axis=-1)
        inputs_rank = tf.rank(y_ds)
        phase_correction = -tf.math.angle(tf.complex(
            tf.nn.convolution(tf.math.real(rot_vec), filter, padding="VALID"),
            tf.nn.convolution(tf.math.imag(rot_vec), filter, padding="VALID")
        ))
        phase_correction = tf.reshape(
            phase_correction, tf.slice(phase_correction.shape, [0], [inputs_rank])
        )
        return y_ds[..., mem_cut:] * tf.math.exp(tf.complex(tf.zeros_like(phase_correction), phase_correction))
