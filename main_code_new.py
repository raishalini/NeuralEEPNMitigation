import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.neighbors import KernelDensity
try:
    import sionna
except ImportError as e:
    import os
    os.system("pip install sionna")
    import sionna
import json
from sionna.channel import utils
from comsys import Transmitter, Receiver, Channel
# from phasenoise import PhaseNoise
from plots import SNRVsLinewidthPlotterNN, MSEVsLinewidthPlotterNN, ScatterPlot, EVMVsLinewidthPlotterNN, MSEXVsLinewidthPlotterNN, AIRPlotter
# from neural_network import MyModel
from neural_network_new import MyModel

class Transceiver:
    def __init__(self, gpu_num=0):
        self.set_gpu(gpu_num)
        self.dtype = tf.complex64

        # Parameters
        self.beta = 0.1
        self.span_in_symbols = 128
        self.samples_per_symbol = 10
        self.beta_2 = -21.67
        self.t_norm = 1e-12
        self.z_norm = 1e3
        self.linewidth = 200e3
        self.f_c = 193.55e12
        self.length_sp = 4000.0
        self.alpha = 0.046
        self.num_bits_per_symbol = 4
        self.batch_size = 1
        self.num_symbols = 11030

        mem = 501
        if not mem % 2:
            warnings.warn("Even number of filter taps for moving average. Expanding by 1.")
            mem = mem + (1 - mem % 2)
        self.filter = tf.ones((mem, 1, 1, 1), dtype=tf.as_dtype(self.dtype).real_dtype)
        self.mem_cut = (mem // 2) * 2
        if not self.mem_cut:
            self.mem_cut = None        

        self.transmitter = Transmitter(self.num_bits_per_symbol, self.batch_size, self.num_symbols, self.samples_per_symbol, self.beta, self.span_in_symbols)
        self.channel = Channel(self.alpha, self.beta_2, self.f_c, self.length_sp, self.t_norm, self.dtype)
        self.receiver = Receiver(self.linewidth, self.t_norm, self.samples_per_symbol, self.transmitter.rcf)
        
        # Initialize the neural network model
        self.nn_equalise = MyModel()
        self.optimizer = tf.keras.optimizers.Adam()

    def set_gpu(self, gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print(e)
        tf.get_logger().setLevel('ERROR')
    
    def run(self):
        x = self.transmitter.generate_qam_symbols()

        x_us = self.transmitter.upsample(x)

        x_rcf = self.transmitter.apply_rcf(x_us)

        x_rcf_padded, padding_left, padding_right = self.transmitter.pad_signal(x_rcf)

        y = self.channel.transmit(x_rcf_padded)
        
        # y_pn = self.receiver.add_phase_noise(y, phase_noise())
        y_pn = self.receiver.add_phase_noise(y)

        y_mf = self.receiver.matched_filter(y_pn)

        y_cdc = self.channel.compensate_dispersion(y_mf)

        y_lpf = self.receiver.low_pass_filter(y_cdc)
        
        y_ds = self.receiver.downsample(y_cdc, padding_left, padding_right)

        y_normalised = self.receiver.normalize(y_ds)

        y_tr, timing_errors = self.receiver.timing_recovery(y_normalised)

        y_cpr = self.receiver.cpr(y_tr, x[..., self.receiver.gardner.avglenhalf + self.receiver.gardner.flenhalf: -(self.receiver.gardner.avglenhalf + self.receiver.gardner.flenhalf)], self.dtype, self.mem_cut, self.filter)

        y_cpr_wo_tr = self.receiver.cpr(y_normalised, x, self.dtype, self.mem_cut, self.filter)
        
        y_cpr_normalised = self.receiver.normalize(y_cpr)
        
        return y_cpr_wo_tr, y_cpr, x, y_cpr_normalised

    def cal_air(self, x, y):
        noise_variance = tf.reduce_mean(tf.math.square(tf.abs(x - y)))
        bandwidth = 0.1
        p_y_given_x = (1 / (tf.math.sqrt(2 * np.pi * noise_variance))) * tf.math.exp(-tf.math.square(tf.abs(x - y)) / (2 * noise_variance))
        # mutual_information = tf.reduce_mean(tf.math.log(p_y_given_x + 1e-12))  
        # print("x: ", x)
        # print("y: ", y)
        x_real_imag = tf.squeeze(tf.stack((tf.math.real(x), tf.math.imag(x)), axis=-1), axis=0)
        y_real_imag = tf.squeeze(tf.stack((tf.math.real(y), tf.math.imag(y)), axis=-1), axis=0)
        # print("x_real_imag: ", x_real_imag)
        # print("y_real_imag: ", y_real_imag)
    
        kde_joint = KernelDensity(bandwidth=bandwidth)
        kde_joint.fit(np.hstack([x_real_imag, y_real_imag]))
    
        kde_y = KernelDensity(bandwidth=bandwidth)
        kde_y.fit(y_real_imag)
    
        log_joint_density = kde_joint.score_samples(np.hstack((x_real_imag, y_real_imag)))
        log_marginal_density_y = kde_y.score_samples(y_real_imag)

        mi_num = tf.cast(tf.reduce_mean(log_joint_density - log_marginal_density_y), tf.float32)
    
        air = mi_num / tf.math.log(2.0)
        return air
        
        
    def cal_mse(self, x, y):
        mse = tf.reduce_mean(tf.math.square(tf.abs(x - y)))
        signal_power = tf.reduce_mean(tf.math.square(tf.abs(x)))
        snr = 10 * tf.math.log(signal_power / mse) / tf.math.log(10.0)
        # evm = 10*tf.math.log(mse)/tf.math.log(10.0)
    
        Q_Y_values = (1 / (tf.math.sqrt(2 * np.pi) * tf.math.square(mse))) * tf.math.exp(-(tf.math.square(tf.abs(x - y))) / (2 * tf.math.square(mse)))
        entropy_reg = -tf.reduce_mean(tf.math.log(Q_Y_values))

        mse_x = mse - 2 * (tf.math.square(mse) * entropy_reg)
        # sigma2 = mse
    
        # Q_Y calculation (Gaussian assumption)
        # Q_Y_values = (1 / (tf.math.sqrt(2 * np.pi * sigma2))) * tf.math.exp(
        #     -(tf.math.square(tf.abs(x - y))) / (2 * sigma2)
        # )
        
        # Entropy regularization term
        # entropy_reg = -tf.reduce_mean(tf.math.log(Q_Y_values + 1e-12))  # Avoid log(0)
        
        # Entropy-Regularized MSE (ERMSE)
        # mse_x = mse - 2 * sigma2 * entropy_reg
        
        return mse_x, snr

    def train_and_test(self, iterations, linewidths, link_distances):

        test_results = {lw: {length: {"mse": [], "snr": [], "evm": [], "mse_x": []} for length in link_distances} for lw in linewidths}
        air_dict = {lw: {length: {"air": []} for length in link_distances} for lw in linewidths}
    
        for idx, lw in enumerate(linewidths):
            self.linewidth = lw
            self.receiver.linewidth = lw
            print(f"\nTraining with linewidth: {lw} Hz")
    
            # Load weights if not the first linewidth
            if idx > 0:
                self.nn_equalise.load_weights("final_model_weights.h5")
                print(f"Loaded model weights for continued training.")
            # print("lw in train 1:", lw)
            # print("linewidth in train 1:", self.receiver.linewidth)
            for length in link_distances:
                patience = 500
                best_loss = float('inf')
                patience_counter = 0
                best_weights = None
                best_loss_air = float('-inf')
                # self.channel.length_sp = length
                self.channel = Channel(alpha=self.channel.alpha, beta_2=self.channel.beta_2, f_c=self.channel.f_c,
                                   length_sp=length, t_norm=self.channel.t_norm, dtype=self.channel.dtype)
                # print("lw in train 2:", lw)
                # print("linewidth in train 2:", self.receiver.linewidth)
                # print(f"Link distance used in the channel: {self.channel.ss_fn._length}")
                # print(f"Link distance used in the channel: {self.channel.ss_fn_cdc._length}")
                # print("link distance inside train 1:", self.channel.length_sp)
                print(f"\nTraining with linewidth: {lw} Hz and Distance: {length} km")
                
                # Build the graph before training
                self.nn_equalise.build_graph()

                for i in range(iterations):
                    y_cpr_wo_tr, y_cpr, tx_symbols, rx_symbols = self.run()
                    tx_symbols_short = tx_symbols[..., self.receiver.gardner.avglenhalf + self.receiver.gardner.flenhalf + 500: -(self.receiver.gardner.avglenhalf + 
                                                                                                        self.receiver.gardner.flenhalf)]
                    x_train = rx_symbols
                    y_train = tx_symbols_short

                    # print("link distance inside train 2:", self.channel.length_sp)
                    with tf.GradientTape() as tape:
                        network_out = self.nn_equalise(x_train, training=True)
                        network_out = self.receiver.normalize(network_out)
                        # print("y_train: ", y_train)
                        # print("network_out: ", network_out)
                        # loss, _ , _, _ = self.cal_mse(y_train, network_out)
                        loss, _ = self.cal_mse(y_train, network_out)
                        air = self.cal_air(y_train, network_out)
        
                    gradients = tape.gradient(loss, self.nn_equalise.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.nn_equalise.trainable_variables))
                    print(f"Iteration {i+1}/{iterations}, Loss: {loss.numpy()}")

                    print(f"Iteration {i+1}/{iterations}, Loss: {loss.numpy()}, AIR: {air.numpy()}")
                    air_dict[lw][length]["air"].append(float(air.numpy()))
                    print("air_dict: ", air_dict)

                    # Early stopping criterion using both MSE and AIR
                    if air.numpy() > best_loss_air:
                        best_loss = loss.numpy()
                        best_loss_air = air.numpy()
                        best_weights = self.nn_equalise.get_weights()
                        patience_counter = 0  
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {i+1} due to no improvement in the last {patience} iterations.")
                        self.nn_equalise.set_weights(best_weights)
                        break

                # print("lw in train 3:", lw)
                # print("linewidth in train 3:", self.receiver.linewidth)
                self.nn_equalise.save_weights("final_model_weights.h5")
                print(f"Model weights saved after training and testing with linewidth {lw} Hz.\n")

                # print("link distance inside train 3:", self.channel.length_sp)
                print(f"\nTesting with linewidth: {lw} Hz and Distance: {length} km")
                # original_mse, original_snr, original_evm, original_mse_x, nn_mse, nn_snr, nn_evm, nn_mse_x, tx_symbols_arr, rx_symbols_arr, rx_symbols_nn_arr = self.test(self.nn_equalise, length, lw, num_symbols=100)
                original_mse_x, original_snr, nn_mse_x, nn_snr, tx_symbols_arr, rx_symbols_arr, rx_symbols_nn_arr = self.test(self.nn_equalise, length, lw, num_symbols=100)
                print(f"Testing MSE - Linewidth: {lw}, Link Distance: {length}, Original: {original_snr.numpy()}, Neural Network: {nn_snr.numpy()}")
                    
                # test_results[lw][length]["mse"].append({"original_mse": float(original_mse.numpy()), "nn_mse": float(nn_mse.numpy())})
                test_results[lw][length]["snr"].append({"original_snr": float(original_snr.numpy()), "nn_snr": float(nn_snr.numpy())})
                # test_results[lw][length]["evm"].append({"original_evm": original_evm.numpy(), "nn_evm": nn_evm.numpy()})
                test_results[lw][length]["mse_x"].append({"original_mse_x": float(original_mse_x.numpy()), "nn_mse_x": float(nn_mse_x.numpy())})
                # test_results[lw][length]["air"].append({"original_air": original_air.numpy(), "nn_air": nn_air.numpy()})

                plot_appender = ScatterPlot()
                plot_appender.append_scatter_plot(tf.reshape(tx_symbols_arr, [-1]), tf.reshape(rx_symbols_arr, [-1]), tf.reshape(rx_symbols_nn_arr, [-1]), link_distance=int(length), linewidth=int(lw))

                def overwrite_json(file_path, new_data):
                    with open(file_path, "w") as file:
                        json.dump(new_data, file, indent=4)
                        print("json created/updated")
                file_path = "test_results_mseloss_snr_wogn.json"
                overwrite_json(file_path, test_results)
        
        
        return test_results, tx_symbols_arr, rx_symbols_arr, rx_symbols_nn_arr, air_dict

    
    def test(self, model, length, linewidth, num_symbols):
        self.channel = Channel(alpha=self.channel.alpha, beta_2=self.channel.beta_2, f_c=self.channel.f_c,
                           length_sp=length, t_norm=self.channel.t_norm, dtype=self.channel.dtype)
        self.receiver.linewidth = linewidth
        self.linewidth = linewidth
        # print("linewidth in test 1:", self.receiver.linewidth)
        # print(f"Link distance used in the channel: {self.channel.ss_fn._length}")
        # print(f"Link distance used in the channel: {self.channel.ss_fn_cdc._length}")
        tx_symbols_arr = tf.TensorArray(dtype=tf.complex64, size=num_symbols)
        rx_symbols_arr = tf.TensorArray(dtype=tf.complex64, size=num_symbols)
        rx_symbols_nn_arr = tf.TensorArray(dtype=tf.complex64, size=num_symbols)
        for i in range(num_symbols):
            y_cpr_wo_tr, y_cpr, tx_symbols, rx_symbols = self.run()
            # print("link distance inside test 1:", self.channel.length_sp)
            tx_symbols_short = tx_symbols[..., self.receiver.gardner.avglenhalf + self.receiver.gardner.flenhalf + 500: -(self.receiver.gardner.avglenhalf + 
                                                                                    self.receiver.gardner.flenhalf) ]
            network_out = model.predict(rx_symbols)
            network_out = self.receiver.normalize(network_out)
            # print("network_out : ", network_out)
            tx_symbols_arr = tx_symbols_arr.write(i, tx_symbols_short)
            rx_symbols_arr = rx_symbols_arr.write(i, rx_symbols)
            rx_symbols_nn_arr = rx_symbols_nn_arr.write(i, network_out)
        # print("link distance inside test 2:", self.channel.length_sp)
        tx_symbols_arr = tx_symbols_arr.stack()
        # print("tx_symbols_arr : ", tx_symbols_arr)
        # print("tx_symbols_arr reshape: ", tf.reshape(tx_symbols_arr, [-1]))
        # Print the link distance used in the channel
        # print(f"Link distance used in the channel: {self.channel.ss_fn._length}")
        # print(f"Link distance used in the channel: {self.channel.ss_fn_cdc._length}")
        rx_symbols_arr = rx_symbols_arr.stack()
        # print("rx_symbols_arr : ", rx_symbols_arr)
        # print("rx_symbols_arr reshape: ", tf.reshape(rx_symbols_arr, [-1]))
        rx_symbols_nn_arr = rx_symbols_nn_arr.stack()
        # print("rx_symbols_nn_arr : ", rx_symbols_nn_arr)
        # print("rx_symbols_nn_arr reshape: ", tf.reshape(rx_symbols_nn_arr, [-1]))
        # org_mse, org_snr = self.cal_mse(tx_symbols_arr, rx_symbols_arr)
        # nn_mse, nn_snr = self.cal_mse(tx_symbols_arr, rx_symbols_nn_arr)
        # print("tx_symbols_arr: ", tx_symbols_arr)
        # print("rx_symbols_arr: ", rx_symbols_arr)
        # print("rx_symbols_nn_arr: ", rx_symbols_nn_arr)
        # org_mse, org_snr, org_evm, org_mse_x = self.cal_mse(tf.reshape(tx_symbols_arr, [-1]), tf.reshape(rx_symbols_arr, [-1]))
        # nn_mse, nn_snr, nn_evm, nn_mse_x = self.cal_mse(tf.reshape(tx_symbols_arr, [-1]), tf.reshape(rx_symbols_nn_arr, [-1]))
        org_mse_x, org_snr = self.cal_mse(tf.reshape(tx_symbols_arr, [-1]), tf.reshape(rx_symbols_arr, [-1]))
        nn_mse_x, nn_snr = self.cal_mse(tf.reshape(tx_symbols_arr, [-1]), tf.reshape(rx_symbols_nn_arr, [-1]))

        return org_mse_x, org_snr, nn_mse_x, nn_snr, tx_symbols_arr, rx_symbols_arr, rx_symbols_nn_arr

if __name__ == "__main__":    
    pipeline = Transceiver()

print("Start Train")
iterations = 2000
linewidths = [100e3, 200e3, 300e3, 400e3, 500e3, 750e3, 1000e3]
link_distances = [1e3, 2e3, 4e3, 5e3]
test_results, tx_symbols_arr, rx_symbols_arr, rx_symbols_nn_arr, air_dict = pipeline.train_and_test(iterations=iterations, linewidths = linewidths, link_distances=link_distances)
print("final air_dict: ", air_dict)
print("test_results: ", test_results)
print("tx_symbols_arr: ", tx_symbols_arr)
print("rx_symbols_arr: ", rx_symbols_arr)
print("rx_symbols_nn_arr: ", rx_symbols_nn_arr)
plotter = SNRVsLinewidthPlotterNN(pipeline, linewidths, link_distances)
plotter.plot(test_results, air_dict)
plotter4 = MSEXVsLinewidthPlotterNN(pipeline, linewidths, link_distances)
plotter4.plot(test_results)
plotter5 = AIRPlotter(pipeline, linewidths, link_distances)
plotter5.plot(air_dict)
# plotter2 = MSEVsLinewidthPlotterNN(pipeline, linewidths, link_distances)
# plotter2.plot(test_results)
# plotter3 = EVMVsLinewidthPlotterNN(pipeline, linewidths, link_distances)
# plotter3.plot(test_results)
