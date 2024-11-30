import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from scipy import optimize
import os
from tkinter import colorchooser
import scipy
import seaborn as sns

sns.set_theme(style='white')
class DataAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Analysis Tool")
        self.root.geometry("1600x900")
        
        # Data storage
        self.data_instance = None
        self.full_data = None
        self.current_data = None
        self.analysis_data = None
        
        self.create_gui_elements()
        
    def create_gui_elements(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="5")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(self.root, padding="5")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figures with custom gridspec
        self.fig = Figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=self.fig, width_ratios=[3, 1, 2.5], wspace = 0)
        
        # Time series plot
        self.ax_time = self.fig.add_subplot(gs[0, 0])
        
        # Histogram plot (rotated)
        self.ax_histogram = self.fig.add_subplot(gs[0, 1], sharey=self.ax_time)
        
        # PSD plot (narrower)
        self.ax_psd = self.fig.add_subplot(gs[0, 2])
        
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Control elements
        ttk.Label(self.control_frame, text="Data Analysis Controls", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # File loading
        ttk.Button(self.control_frame, text="Load Data File", command=self.load_data).pack(pady=5, fill=tk.X)
        
        # Plot time range frame
        plot_time_frame = ttk.LabelFrame(self.control_frame, text="Plot Time Range", padding="5")
        plot_time_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(plot_time_frame, text="Start Time (s):").pack()
        self.plot_start_time = ttk.Entry(plot_time_frame)
        self.plot_start_time.pack(fill=tk.X)
        self.plot_start_time.insert(0, "0")
        
        ttk.Label(plot_time_frame, text="Stop Time (s):").pack()
        self.plot_stop_time = ttk.Entry(plot_time_frame)
        self.plot_stop_time.pack(fill=tk.X)
        
        ttk.Button(plot_time_frame, text="Update Plot Range", command=self.update_plot_range).pack(pady=5, fill=tk.X)
        
        # Analysis time range frame
        analysis_time_frame = ttk.LabelFrame(self.control_frame, text="Analysis Time Range", padding="5")
        analysis_time_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(analysis_time_frame, text="Start Time (s):").pack()
        self.analysis_start_time = ttk.Entry(analysis_time_frame)
        self.analysis_start_time.pack(fill=tk.X)
        self.analysis_start_time.insert(0, "0")
        
        ttk.Label(analysis_time_frame, text="Stop Time (s):").pack()
        self.analysis_stop_time = ttk.Entry(analysis_time_frame)
        self.analysis_stop_time.pack(fill=tk.X)
        
        ttk.Button(analysis_time_frame, text="Update Analysis Range", command=self.update_analysis_range).pack(pady=5, fill=tk.X)
        
        # Analysis options frame
        analysis_frame = ttk.LabelFrame(self.control_frame, text="Analysis Options", padding="5")
        analysis_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(analysis_frame, text="Filter Cutoff (Hz):").pack()
        self.cutoff_freq = ttk.Entry(analysis_frame)
        self.cutoff_freq.pack(fill=tk.X)
        self.cutoff_freq.insert(0, "10")
        
        # Plotting options
        ttk.Label(analysis_frame, text="Plot Type:").pack()
        self.plot_type_var = tk.StringVar(value="Full Range")
        plot_type_options = ["Full Range", "Original", "Filtered"]
        self.plot_type_dropdown = ttk.Combobox(analysis_frame, textvariable=self.plot_type_var, values=plot_type_options, state="readonly")
        self.plot_type_dropdown.pack(fill=tk.X)
        self.plot_type_dropdown.bind("<<ComboboxSelected>>", self.update_plot_type)
        
        # Analysis buttons
        ttk.Button(analysis_frame, text="Plot Data", command=self.plot_data).pack(pady=5, fill=tk.X)
        ttk.Button(analysis_frame, text="Update PSD", command=self.update_psd).pack(pady=5, fill=tk.X)
        
        # New buttons frame
        buttons_frame = ttk.LabelFrame(self.control_frame, text="Additional Options", padding="5")
        buttons_frame.pack(pady=5, fill=tk.X)
        
        # Full Range Reset Button
        ttk.Button(buttons_frame, text="Reset to Full Range", command=self.reset_to_full_range).pack(pady=5, fill=tk.X)
        
        # Separate Windows Button
        ttk.Button(buttons_frame, text="Open Plots in Separate Windows", command=self.open_separate_windows).pack(pady=5, fill=tk.X)
        
        # RMSD display
        rmsd_frame = ttk.LabelFrame(self.control_frame, text="RMSD Result", padding="5")
        rmsd_frame.pack(pady=5, fill=tk.X)
        self.rmsd_text = tk.Text(rmsd_frame, height=1, width=20)
        self.rmsd_text.pack(pady=5, fill=tk.X)

    def reset_to_full_range(self):
        """Reset plot range to full data length"""
        if self.data_instance is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        # Set plot start and stop times to full data range
        time_length = len(self.full_data) * 0.00001
        self.plot_start_time.delete(0, tk.END)
        self.plot_start_time.insert(0, "0")
        self.plot_stop_time.delete(0, tk.END)
        self.plot_stop_time.insert(0, f"{time_length:.2f}")
        
        # Replot with full range
        self.plot_data()
    
    def open_separate_windows(self):
        """Open each plot in a separate window"""
        if self.full_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        # Time Series Plot
        time_fig, time_ax = plt.subplots(figsize=(12, 4))
        start = float(self.plot_start_time.get())
        stop = float(self.plot_stop_time.get())
        plot_data = self.data_instance.select(self.full_data, start, stop)
        plot_data /= np.mean(plot_data)
        
        # Filtering
        cutoff = float(self.cutoff_freq.get())
        nyq = 0.5 * 100000  # sampling rate
        normal_cutoff = cutoff / nyq
        b, a = butter(4, normal_cutoff, 'low')
        filtered_data = filtfilt(b, a, plot_data)
        
        time_step = 0.00001
        time_variable = np.arange(start=0, stop=len(plot_data), step=1) * time_step
        
        # Plot both original and filtered
        time_ax.plot(time_variable, plot_data, linewidth=0.5, alpha=0.5, label='Raw Data')
        time_ax.plot(time_variable, filtered_data, linewidth=1, alpha=1, label=f'{cutoff:.0f} Hz Filter')
        time_ax.set_xlabel('Time [s]')
        time_ax.set_ylabel('Normalized Transmission [V]')
        time_ax.set_xlim(0, time_variable[-1])
        time_ax.legend(frameon=False)
        
        # Histogram Plot
        hist_fig, hist_ax = plt.subplots(figsize=(2, 4))
        #counts, bins = np.histogram(plot_data, bins='auto', density=True)
        #bin_centers = (bins[:-1] + bins[1:]) / 2
        #hist_ax.plot(counts/max(counts), bin_centers, alpha=1)
        #hist_ax.fill_between(counts/max(counts), bin_centers, alpha=0.3)
        #hist_ax.set_xlabel('Norm. Counts [a.u.]')
        #hist_ax.set_ylabel('Normalized Transmission [V]')
        
        # Histogram plot (horizontal)
        sns.histplot(y=plot_data, kde=True, stat='density', bins=25, ax = self.ax_histogram, alpha=0.3, color ='k')            
        self.ax_histogram.set_xlabel('Density')
        
        # Remove y-axis label from histogram as it shares y-axis with time series
        self.ax_histogram.set_ylabel('')
        
        # Hide histogram y-tick labels
        plt.setp(self.ax_histogram.get_yticklabels(), visible=False)


        # PSD Plot
        psd_fig, psd_ax = plt.subplots(figsize=(5, 4))
        
        # Calculate PSD
        n = len(self.full_data)
        dt = 0.00001
        data_fft = scipy.fftpack.fft(self.full_data)
        data_psd = np.abs(data_fft)**2 / (n*dt)
        freq = scipy.fftpack.fftfreq(n, dt)
        i = freq > 0 
        freq = freq[i]
        data_psd = data_psd[i]
        
        # Plot PSD
        psd_ax.loglog(freq, data_psd)
        
        # Fit Lorentzian
        def func(x, A, fc):
            return A/(x**2 + fc**2)
        
        num_delete = int(2/0.2)  # Delete first few points
        x = freq[num_delete:]
        y = data_psd[num_delete:]
        
        try:
            popt, _ = optimize.curve_fit(func, x, y)
            A, fc = popt
            fc = abs(fc)
            
            psd_ax.loglog(freq, A/(freq**2 + fc**2), 'r-', linewidth =4)
            psd_ax.set_xlabel('Frequency [Hz]')
            psd_ax.set_ylabel('Power Spectral Density [$V^2$/Hz]')
            psd_ax.text(0.25, 0.000001, f'$f_c$: {abs(fc):.0f} Hz')
            psd_ax.set_xlim(0.2, 4000)

        except Exception as e:
            print(f"Failed to fit Lorentzian: {str(e)}")
        
        # Show all figures
        time_fig.tight_layout()
        hist_fig.tight_layout()
        psd_fig.tight_layout()
        plt.show()

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.data_instance = Get_data(file_path)
                self.full_data = self.data_instance.data
                self.current_data = self.full_data
                self.analysis_data = self.current_data
                
                # Set stop time to full data length
                time_length = len(self.full_data) * 0.00001
                self.plot_stop_time.delete(0, tk.END)
                self.plot_stop_time.insert(0, f"{time_length:.2f}")
                self.analysis_stop_time.delete(0, tk.END)
                self.analysis_stop_time.insert(0, f"{time_length:.2f}")
                
                self.plot_data()
                self.update_psd()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def update_plot_range(self):
        if self.data_instance is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            start = float(self.plot_start_time.get())
            stop = float(self.plot_stop_time.get())
            self.plot_data()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid time values!")
    
    def update_analysis_range(self):
        if self.data_instance is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            start = float(self.analysis_start_time.get())
            stop = float(self.analysis_stop_time.get())
            self.analysis_data = self.data_instance.select(self.full_data, start, stop)
            self.update_psd()
            self.calculate_rmsd()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid time values!")
    
    def plot_data(self):
        if self.full_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            start = float(self.plot_start_time.get())
            stop = float(self.plot_stop_time.get())
            plot_data = self.data_instance.select(self.full_data, start, stop)
            plot_data /= np.mean(plot_data)
            # Clear the axes
            self.ax_time.clear()
            self.ax_histogram.clear()
            
            # Get time array for the selected data
            time_step = 0.00001
            time_variable = np.arange(start=0, stop=len(plot_data), step=1) * time_step
            
            # Determine plot type
            plot_type = self.plot_type_var.get()
            
            # Filter cutoff frequency
            cutoff = float(self.cutoff_freq.get())
            
            # Filtering
            nyq = 0.5 * 100000  # sampling rate
            normal_cutoff = cutoff / nyq
            b, a = butter(4, normal_cutoff, 'low')
            filtered_data = filtfilt(b, a, plot_data)
            
            # Time series plot
            if plot_type == "Full Range":
                # Plot both original and filtered data
                self.ax_time.plot(time_variable, plot_data, linewidth=0.5, alpha=0.5, label='Raw Data', color = 'k')
                self.ax_time.plot(time_variable, filtered_data, linewidth=1, alpha=1, label=f'{cutoff:.0f} Hz Filter', color='k')
                self.ax_time.legend(frameon=False)
            elif plot_type == "Original":
                # Plot only original data
                self.ax_time.plot(time_variable, plot_data, linewidth=0.5, color = 'k')
            else:  # Filtered
                # Plot only filtered data
                self.ax_time.plot(time_variable, filtered_data, linewidth=1, color = 'k')
            
            # Set time series plot labels and limits
            self.ax_time.set_xlabel('Time [s]')
            self.ax_time.set_ylabel('Normalized Transmission [V]')
            self.ax_time.set_xlim(0, time_variable[-1])
            

            # comppute histogram

            #counts, bins = np.histogram(plot_data, bins = 50, density = True)
            #bin_centers = (bins[:-1] + bins[1:]) / 2
            # Histogram plot (horizontal)
            #self.ax_histogram.plot(counts/max(counts), bin_centers, alpha=1, color='k')
            #self.ax_histogram.fill_between(counts/max(counts), bin_centers, alpha=0.3, color='k')
            #self.ax_histogram.set_xlabel('Norm. Counts [a.u.]')
            #self.ax_histogram.set_ylim(0.85, 1.15)
            #self.ax_histogram.set_xlim(0, 1.1)

            # Remove y-axis label from histogram as it shares y-axis with time series
            #self.ax_histogram.set_ylabel('')
            
            # Hide histogram y-tick labels
            #plt.setp(self.ax_histogram.get_yticklabels(), visible=False)
            
            sns.histplot(y=plot_data, kde=True, stat='density', bins=25, ax = self.ax_histogram, alpha=0.3, color ='k')            
            self.ax_histogram.set_xlabel('Density')
            
            # Remove y-axis label from histogram as it shares y-axis with time series
            self.ax_histogram.set_ylabel('')
            
            # Hide histogram y-tick labels
            plt.setp(self.ax_histogram.get_yticklabels(), visible=False)


            self.canvas.draw()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid time values and cutoff frequency!")
    
    def update_plot_type(self, event=None):
        # Automatically update plot when plot type is changed
        self.plot_data()
    
    def update_psd(self):
        if self.analysis_data is None:
            messagebox.showwarning("Warning", "Please select data first!")
            return
            
        self.ax_psd.clear()
        
        # Calculate PSD
        n = len(self.analysis_data)
        dt = 0.00001

        data_fft = scipy.fftpack.fft(self.analysis_data)
        data_psd = np.abs(data_fft)**2 / (n*dt)

        freq = scipy.fftpack.fftfreq(n, dt)
        i = freq > 0 

        freq = freq[i]
        data_psd = data_psd[i]
        
        # Plot PSD
        self.ax_psd.loglog(freq, data_psd, color = 'k', alpha = 0.5)
        
        # Fit and plot Lorentzian
        def func(x, A, fc):
            return A/(x**2 + fc**2)
        
        num_delete = int(2/0.2)  # Delete first few points
        x = freq[num_delete:]
        y = data_psd[num_delete:]
        
        try:
            popt, _ = optimize.curve_fit(func, x, y)
            A, fc = popt
            fc = abs(fc)
            
            self.ax_psd.loglog(freq, A/(freq**2 + fc**2), 'r-', linewidth=4)
            self.ax_psd.set_xlabel('Frequency [Hz]')
            self.ax_psd.yaxis.tick_right()
            self.ax_psd.set_ylabel('Power Spectral Density [$V^2$/Hz]')
            self.ax_psd.text(0.25, 0.000001, f'$f_c$: {abs(fc):.0f} Hz')
            self.ax_psd.set_xlim(0.2, 4000)
            self.ax_psd.yaxis.set_label_position('right')



        except Exception as e:
            print(f"Failed to fit Lorentzian: {str(e)}")
        
        self.canvas.draw()
    
    def calculate_rmsd(self):
        if self.analysis_data is None:
            return
            
        window_size = 5000
        N = len(self.analysis_data)
        num_window = int(N/window_size)
        RMSD_set = []
        data_ave = np.average(self.analysis_data)
        
        for i in range(num_window):
            start_p = i*window_size
            stop_p = (i+1)*window_size
            X = self.analysis_data[start_p:stop_p]
            X_ave = np.average(X)
            RMSD_temp = np.sqrt(np.sum((X-X_ave)**2)/window_size)/data_ave
            RMSD_set.append(RMSD_temp)
        
        RMSD = np.average(RMSD_set)
        self.rmsd_text.delete(1.0, tk.END)
        self.rmsd_text.insert(1.0, f"{RMSD:.6f}")

class Get_data:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, start=1):
                if line.strip() and line.strip()[0].isdigit():
                    skip_lines = i - 1
                    break
        self.data = np.genfromtxt(file_path, skip_header=skip_lines)
        self.sampling_rate = 100000
        self.time_length = len(self.data) / self.sampling_rate

    def select(self, data, start_time, stop_time):
        dt = 0.00001
        start_p = int(start_time / dt)
        stop_p = int(stop_time / dt)
        return data[start_p:stop_p]

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisGUI(root)
    root.mainloop()
    # Call this to apply the modifications
