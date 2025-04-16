import os
try:
    import plotly
except ImportError as e:
    os.system("pip install plotly")
os.system("pip install -U kaleido")
os.system("pip install tikzplotlib")
os.system("pip install matplotlib==3.6.3")
import tikzplotlib
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
try:
    import tikzplotly
except ImportError as e:
    os.system("pip install tikzplotly")
pio.renderers.default = "jupyterlab"
# pio.templates.default = "plotly_dark"


class SNRVsLinewidthPlotter:
    def __init__(self, transceiver, linewidths, link_distances):
        self.transceiver = transceiver
        self.linewidths = linewidths
        self.link_distances = link_distances

    def plot(self, results):
        # results = self.transceiver.evaluate_snr(self.linewidths, self.link_distances)
        fig = go.Figure()

        for length in self.link_distances:
            fig.add_trace(go.Scatter(x=self.linewidths, y=results['with_tr'][length], mode='lines',
                                     name=f'With TR, Distance {length} km'))
            fig.add_trace(go.Scatter(x=self.linewidths, y=results['without_tr'][length], mode='lines',
                                     name=f'Without TR, Distance {length} km', line=dict(dash='dash')))

        fig.update_layout(xaxis_title='Linewidth (Hz)',
                          yaxis_title='SNR (dB)',
                          title='SNR vs. Linewidth for Different Link Distances',
                          legend_title='Link Distance')
        fig.write_html("snr_linewidth_linkdist.html")
        fig.show()

class ScatterPlot:
    def __init__(self, output_folder='plots'):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    def append_scatter_plot(self, tx, rx, rx_nn, link_distance, linewidth):
        # num_symbols_to_plot = 1000
        # tx = tx[:num_symbols_to_plot]
        # rx = rx[:num_symbols_to_plot]
        # rx_nn = rx_nn[:num_symbols_to_plot]

        # # Extract real and imaginary components
        # tx_real = np.real(tx)
        # tx_imag = np.imag(tx)

        # rx_real = np.real(rx)
        # rx_imag = np.imag(rx)

        # rx_nn_real = np.real(rx_nn)
        # rx_nn_imag = np.imag(rx_nn)

        # # Create subplots
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # fig.suptitle(f"Scatter Plots of Tx vs Rx and Tx vs Rx NN\n(Link Dist: {link_distance}, Linewidth: {linewidth})")

        # # Tx vs Rx
        # axs[0].scatter(tx_real, tx_imag, color='blue', label='Tx', s=6, alpha=0.8)
        # axs[0].scatter(rx_real, rx_imag, color='red', label='Rx', s=6, alpha=0.8)
        # axs[0].set_title("Tx vs Rx")
        # axs[0].set_xlabel("Real")
        # axs[0].set_ylabel("Imaginary")
        # axs[0].legend()
        # axs[0].grid(True)

        # # Tx vs Rx NN
        # axs[1].scatter(tx_real, tx_imag, color='blue', label='Tx', s=6, alpha=0.8)
        # axs[1].scatter(rx_nn_real, rx_nn_imag, color='green', label='Rx NN', s=6, alpha=0.8)
        # axs[1].set_title("Tx vs Rx NN")
        # axs[1].set_xlabel("Real")
        # axs[1].set_ylabel("Imaginary")
        # axs[1].legend()
        # axs[1].grid(True)

        # # Adjust layout
        # plt.tight_layout(rect=[0, 0, 1, 0.95])

        num_symbols_to_plot = 1000
        tx = tx[:num_symbols_to_plot]
        rx = rx[:num_symbols_to_plot]
        rx_nn = rx_nn[:num_symbols_to_plot]

        # Create a new figure
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121)  # 1 row, 2 columns, 1st subplot
        ax2 = fig.add_subplot(122)  # 1 row, 2 columns, 2nd subplot

        # Extract real and imaginary components
        tx_real = np.real(tx)
        tx_imag = np.imag(tx)

        rx_real = np.real(rx)
        rx_imag = np.imag(rx)

        rx_nn_real = np.real(rx_nn)
        rx_nn_imag = np.imag(rx_nn)

        # Add scatter plot for Tx vs Rx
        ax1.scatter(tx_real, tx_imag, color='blue', label='Tx', marker='o', s=40)
        ax1.scatter(rx_real, rx_imag, color='red', label='Rx', marker='x', s=40)

        # Add scatter plot for Tx vs Rx NN
        ax2.scatter(tx_real, tx_imag, color='blue', label='Tx', marker='o', s=40)
        ax2.scatter(rx_nn_real, rx_nn_imag, color='green', label='Rx NN', marker='x', s=40)

        # Update layout for titles and axis labels
        ax1.set_title(f"Tx vs Rx\n(Link Dist: {link_distance}, Linewidth: {linewidth})")
        ax2.set_title(f"Tx vs Rx NN\n(Link Dist: {link_distance}, Linewidth: {linewidth})")
        
        ax1.set_xlabel("Real")
        ax1.set_ylabel("Imaginary")
        ax2.set_xlabel("Real")
        ax2.set_ylabel("Imaginary")
        
        ax1.legend()
        ax2.legend()

        def tikzplotlib_fix_ncols(obj):
            """
            workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
            """
            if hasattr(obj, "_ncols"):
                obj._ncol = obj._ncols
            for child in obj.get_children():
                tikzplotlib_fix_ncols(child)

        tikzplotlib_fix_ncols(fig)

        # Generate a unique file name based on link distance and linewidth
        file_name = f"scatter_plot_ld{link_distance}_lw{linewidth}.svg"
        file_path = os.path.join(self.output_folder, file_name)

        tikz_file_path = file_path.replace('.svg', '.tex')
        # Save the figure
        tikzplotlib.save(tikz_file_path)
        fig.savefig(file_path, format='svg')
        print(f"Saved plot to {file_path}")


# class ScatterPlot:
#     def __init__(self, output_folder='plots', json_file='scatter_plot_data.json'):
#         self.output_folder = output_folder
#         os.makedirs(self.output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
#         # self.json_file = json_file
#         # # Initialize the JSON file if it doesn't exist
#         # if not os.path.exists(self.json_file):
#         #     with open(self.json_file, 'w') as f:
#         #         json.dump({}, f)  # Start with an empty dictionary

#     def append_scatter_plot(self, tx, rx, rx_nn, link_distance, linewidth):
#         num_symbols_to_plot = 1000
#         tx = tx[:num_symbols_to_plot]
#         rx = rx[:num_symbols_to_plot]
#         rx_nn = rx_nn[:num_symbols_to_plot]
#         fig = make_subplots(rows=1, cols=2, subplot_titles=("Tx vs Rx", "Tx vs Rx NN"))
#         # Extract real and imaginary components
#         tx_real = np.real(tx)
#         tx_imag = np.imag(tx)

#         rx_real = np.real(rx)
#         rx_imag = np.imag(rx)

#         rx_nn_real = np.real(rx_nn)
#         rx_nn_imag = np.imag(rx_nn)

#         # Add scatter plot for Tx vs Rx
#         fig.add_trace(go.Scatter(
#             x=tx_real, 
#             y=tx_imag, 
#             mode='markers', 
#             name='Tx',
#             marker=dict(color='blue', symbol='circle', size=6)),
#             row=1, col=1
#         )
#         fig.add_trace(go.Scatter(
#             x=rx_real, 
#             y=rx_imag, 
#             mode='markers', 
#             name='Rx',
#             marker=dict(color='red', symbol='cross', size=6)),
#             row=1, col=1
#         )

#         # Add scatter plot for Tx vs Rx NN
#         fig.add_trace(go.Scatter(
#             x=tx_real, 
#             y=tx_imag, 
#             mode='markers', 
#             name='Tx',
#             marker=dict(color='blue', symbol='circle', size=6)),
#             row=1, col=2
#         )
#         fig.add_trace(go.Scatter(
#             x=rx_nn_real, 
#             y=rx_nn_imag, 
#             mode='markers', 
#             name='Rx NN',
#             marker=dict(color='green', symbol='cross', size=6)),
#             row=1, col=2
#         )

#         # Update layout for titles and axis labels
#         fig.update_layout(
#             title=f"Scatter Plots of Tx vs Rx and Tx vs Rx NN\n(Link Dist: {link_distance}, Linewidth: {linewidth})",
#             xaxis_title="Real",
#             yaxis_title="Imaginary",
#             showlegend=True,
#             height=600, width=1200
#         )


#     # Generate a unique file name based on link distance and linewidth
#         file_name = f"scatter_plot_ld{link_distance}_lw{linewidth}.svg"
#         file_path = os.path.join(self.output_folder, file_name)

#         # Save the current figure to the specified HTML file
#         # fig.write_html(file_path)
#         fig.write_image(file_path)
#         tikzplotly.save(f"{file_path}.tex", fig)
#         print(f"Saved plot to {file_path}")

class PlotInputOutput:
    def __init__(self, y_cpr, x):
        self.y_cpr = y_cpr
        self.x = x
        
    def plot_scatter(self):
        plt.figure()
        plt.scatter(np.real(self.y_cpr), np.imag(self.y_cpr))
        plt.scatter(np.real(self.x), np.imag(self.x))
        plt.legend(["RX","TX"]);
        plt.title("Scatter plot of the transmitted and received QAM symbols")
        fig.write_html("constellation.html")
        plt.show()
        
class PlotInputOutput_NN:
    def __init__(self, y_cpr, x):
        self.y_cpr = y_cpr
        self.x = x
        
    def plot_scatter(self):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure()
        plt.scatter(np.real(self.y_cpr), np.imag(self.y_cpr))
        plt.scatter(np.real(self.x), np.imag(self.x))
        plt.legend(["RX","TX"]);
        plt.title("Scatter plot of the transmitted and received QAM symbols")
        fig.write_html("constellation_NN.html")
        plt.show()

class PlotNN:
    def __init__(self, title="Model Performance"):
        self.title = title

    def plot_evaluate(self, y_true, y_pred):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=y_true[0, :, 0], mode='lines', name='True Real'
        ))
        fig.add_trace(go.Scatter(
            y=y_pred[0, :, 0], mode='lines', name='Pred Real', line=dict(dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            y=y_true[0, :, 1], mode='lines', name='True Imag', xaxis='x2', yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            y=y_pred[0, :, 1], mode='lines', name='Pred Imag', line=dict(dash='dash'), xaxis='x2', yaxis='y2'
        ))
        
        fig.update_layout(
            title=self.title,
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            xaxis2=dict(title='Sample Index', anchor='y2'),
            yaxis2=dict(title='Amplitude', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1)
        )
        
        fig.show()


class SNRVsLinewidthPlotterNN:
    def __init__(self, transceiver, linewidths, link_distances):
        self.transceiver = transceiver
        self.linewidths = linewidths
        self.link_distances = link_distances

    def plot(self, results, air_dict):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Iterate over link distances and plot SNR values for each linewidth
        for length in self.link_distances:
            snr_with_nn = []
            snr_without_nn = []
        
            for lw in self.linewidths:
                snr_with_nn.append(results[lw][length]["snr"][0]["nn_snr"])
                snr_without_nn.append(results[lw][length]["snr"][0]["original_snr"])
        
            # Plot SNR values with and without NN
            ax.plot(self.linewidths, snr_with_nn, label=f'With NN, Distance {length} km', marker='o')
            ax.plot(self.linewidths, snr_without_nn, label=f'Without NN, Distance {length} km', marker='x')
        
        # Add labels, title, and legend
        ax.set_xlabel('Linewidth (Hz)')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('SNR vs. Linewidth for Different Link Distances')
        ax.legend(title='Link Distance')
        ax.grid(True)

        def tikzplotlib_fix_ncols(obj):
            """
            workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
            """
            if hasattr(obj, "_ncols"):
                obj._ncol = obj._ncols
            for child in obj.get_children():
                tikzplotlib_fix_ncols(child)

        tikzplotlib_fix_ncols(fig)
        # Save as SVG
        fig.savefig("snr_linewidth_linkdistNN.svg", format='svg')
        tikzplotlib.save('snr_linewidth_linkdistNN.tex')
        # Show plot
        plt.show()

 # plt.figure(figsize=(10, 6))

        # for length in self.link_distances:
        #     snr_with_nn = []
        #     snr_without_nn = []

        #     for lw in self.linewidths:
        #         snr_with_nn.append(results[lw][length]["snr"][0]["nn_snr"])
        #         snr_without_nn.append(results[lw][length]["snr"][0]["original_snr"])

        #     # Plot with NN
        #     plt.plot(self.linewidths, snr_with_nn, label=f'With NN, Distance {length} km')

        #     # Plot without NN (dashed line)
        #     plt.plot(self.linewidths, snr_without_nn, label=f'Without NN, Distance {length} km')

        # # Set plot labels and title
        # plt.xlabel('Linewidth (Hz)')
        # plt.ylabel('SNR (dB)')
        # plt.title('SNR vs. Linewidth for Different Link Distances')

        # # Add legend
        # plt.legend(title='Link Distance')

        # # Add grid
        # plt.grid(True, alpha=0.7)
# class SNRVsLinewidthPlotterNN:
#     def __init__(self, transceiver, linewidths, link_distances):
#         self.transceiver = transceiver
#         self.linewidths = linewidths
#         self.link_distances = link_distances

#     def plot(self, results):
#         fig = go.Figure()

#         for length in self.link_distances:
#             snr_with_nn = []
#             snr_without_nn = []

#             for lw in self.linewidths:
#                 snr_with_nn.append(results[lw][length]["snr"][0]["nn_snr"])
#                 snr_without_nn.append(results[lw][length]["snr"][0]["original_snr"])

#             fig.add_trace(go.Scatter(x=self.linewidths, y=snr_with_nn, mode='lines',
#                                      name=f'With NN, Distance {length} km'))
#             fig.add_trace(go.Scatter(x=self.linewidths, y=snr_without_nn, mode='lines',
#                                      name=f'Without NN, Distance {length} km', line=dict(dash='dash')))

#         fig.update_layout(
#             xaxis_title='Linewidth (Hz)',
#             yaxis_title='SNR (dB)',
#             title='SNR vs. Linewidth for Different Link Distances',
#             legend_title='Link Distance',
#         )

#         fig.write_image("snr_linewidth_linkdistNN.svg")
#         # tikzplotly.save("snr_linewidth_linkdistNN.tex", fig)
#         # fig.write_html("snr_linewidth_linkdistNN.html")
#         fig.show()

class MSEVsLinewidthPlotterNN:
    def __init__(self, transceiver, linewidths, link_distances):
        self.transceiver = transceiver
        self.linewidths = linewidths
        self.link_distances = link_distances

    def plot(self, results):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Iterate over link distances and plot SNR values for each linewidth
        for length in self.link_distances:
            mse_with_nn = []
            mse_without_nn = []
        
            for lw in self.linewidths:
                mse_with_nn.append(results[lw][length]["mse"][0]["nn_mse"])
                mse_without_nn.append(results[lw][length]["mse"][0]["original_mse"])
        
            # Plot SNR values with and without NN
            ax.plot(self.linewidths, mse_with_nn, label=f'With NN, Distance {length} km', marker='o')
            ax.plot(self.linewidths, mse_without_nn, label=f'Without NN, Distance {length} km', marker='x')
        
        # Add labels, title, and legend
        ax.set_xlabel('Linewidth (Hz)')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs. Linewidth for Different Link Distances')
        ax.legend(title='Link Distance')
        ax.grid(True)

        def tikzplotlib_fix_ncols(obj):
            """
            workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
            """
            if hasattr(obj, "_ncols"):
                obj._ncol = obj._ncols
            for child in obj.get_children():
                tikzplotlib_fix_ncols(child)

        tikzplotlib_fix_ncols(fig)
        # Save as SVG
        fig.savefig("mse_linewidth_linkdistNN.svg", format='svg')
        tikzplotlib.save('mse_linewidth_linkdistNN.tex')
        # Show plot
        plt.show()

# class MSEVsLinewidthPlotterNN:
#     def __init__(self, transceiver, linewidths, link_distances):
#         self.transceiver = transceiver
#         self.linewidths = linewidths
#         self.link_distances = link_distances

#     def plot(self, results):
#         fig = go.Figure()

#         for length in self.link_distances:
#             snr_with_nn = []
#             snr_without_nn = []

#             for lw in self.linewidths:
#                 snr_with_nn.append(results[lw][length]["mse"][0]["nn_mse"])
#                 snr_without_nn.append(results[lw][length]["mse"][0]["original_mse"])

#             fig.add_trace(go.Scatter(x=self.linewidths, y=snr_with_nn, mode='lines',
#                                      name=f'With NN, Distance {length} km'))
#             fig.add_trace(go.Scatter(x=self.linewidths, y=snr_without_nn, mode='lines',
#                                      name=f'Without NN, Distance {length} km', line=dict(dash='dash')))

#         fig.update_layout(
#             xaxis_title='Linewidth (Hz)',
#             yaxis_title='MSE',
#             title='MSE vs. Linewidth for Different Link Distances',
#             legend_title='Link Distance',
#         )

#         fig.write_html("mse_linewidth_linkdistNN.html")
#         fig.show()

class MSEXVsLinewidthPlotterNN:
    def __init__(self, transceiver, linewidths, link_distances):
        self.transceiver = transceiver
        self.linewidths = linewidths
        self.link_distances = link_distances

    def plot(self, results):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Iterate over link distances and plot MSE_X values for each linewidth
        for length in self.link_distances:
            msex_with_nn = []
            msex_without_nn = []
        
            for lw in self.linewidths:
                msex_with_nn.append(results[lw][length]["mse_x"][0]["nn_mse_x"])
                msex_without_nn.append(results[lw][length]["mse_x"][0]["original_mse_x"])
        
            # Plot SNR values with and without NN
            ax.plot(self.linewidths, msex_with_nn, label=f'With NN, Distance {length} km', marker='o')
            ax.plot(self.linewidths, msex_without_nn, label=f'Without NN, Distance {length} km', marker='x')
        
        # Add labels, title, and legend
        ax.set_xlabel('Linewidth (Hz)')
        ax.set_ylabel('MSE_X')
        ax.set_title('MSE_X vs. Linewidth for Different Link Distances')
        ax.legend(title='Link Distance')
        ax.grid(True)

        def tikzplotlib_fix_ncols(obj):
            """
            workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
            """
            if hasattr(obj, "_ncols"):
                obj._ncol = obj._ncols
            for child in obj.get_children():
                tikzplotlib_fix_ncols(child)

        tikzplotlib_fix_ncols(fig)
        # Save as SVG
        fig.savefig("msex_linewidth_linkdistNN.svg", format='svg')
        tikzplotlib.save('msex_linewidth_linkdistNN.tex')
        # Show plot
        plt.show()


class EVMVsLinewidthPlotterNN:
    def __init__(self, transceiver, linewidths, link_distances):
        self.transceiver = transceiver
        self.linewidths = linewidths
        self.link_distances = link_distances

    def plot(self, results):
        fig = go.Figure()

        for length in self.link_distances:
            evm_with_nn = []
            evm_without_nn = []

            for lw in self.linewidths:
                evm_with_nn.append(results[lw][length]["evm"][0]["nn_evm"])
                evm_without_nn.append(results[lw][length]["evm"][0]["original_evm"])

            fig.add_trace(go.Scatter(x=self.linewidths, y=evm_with_nn, mode='lines',
                                     name=f'With NN, Distance {length} km'))
            fig.add_trace(go.Scatter(x=self.linewidths, y=evm_without_nn, mode='lines',
                                     name=f'Without NN, Distance {length} km', line=dict(dash='dash')))

        fig.update_layout(
            xaxis_title='Linewidth (Hz)',
            yaxis_title='EVM',
            title='EVM vs. Linewidth for Different Link Distances',
            legend_title='Link Distance',
        )

        fig.write_html("evm_linewidth_linkdistNN.html")
        fig.show()

class AIRPlotter:
    def __init__(self, transceiver, linewidths, link_distances):
        self.transceiver = transceiver
        self.linewidths = linewidths
        self.link_distances = link_distances

    def plot(self, air_dict):
        """
        Plots AIR vs. Epoch for all combinations of linewidth and link distance.
        air_dict: Nested dictionary containing AIR values for each linewidth and link distance.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Iterate over linewidths and link distances
        for lw, distances in air_dict.items():
            for length, values in distances.items():
                air_values = values["air"]
                
                # X-axis: Epochs
                epochs = list(range(1, len(air_values) + 1))
                
                # Plot AIR vs. Epoch for each combination
                ax.plot(epochs, air_values, label=f"Linewidth: {lw} Hz, Distance: {length} km")
        
        # Add labels, title, and legend
        ax.set_xlabel("Epochs")
        ax.set_ylabel("AIR (bits)")
        ax.set_title("AIR vs. Epoch for Different Linewidth and Link Distances")
        ax.legend(title="Combinations", loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        ax.grid(True)
        
        # Adjust layout to fit legend
        plt.tight_layout()
        def tikzplotlib_fix_ncols(obj):
            """
            workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
            """
            if hasattr(obj, "_ncols"):
                obj._ncol = obj._ncols
            for child in obj.get_children():
                tikzplotlib_fix_ncols(child)

        tikzplotlib_fix_ncols(fig)
        # Save as SVG
        fig.savefig("air_plot.svg", format='svg')
        tikzplotlib.save('air_plot.tex')
        # Show plot
        plt.show()
