import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from scipy import optimize
import os

class DataAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Analysis Tool")
        self.root.geometry("1400x900")
        
        # Data storage
        self.data_instance = None
        self.current_data = None
        self.analysis_data = None
        
        self.create_gui_elements()
        
    def create_gui_elements(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="5")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(self.root, padding="5")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figures
        self.fig = Figure(figsize=(12, 8))
        self.ax_time = self.fig.add_subplot(211)
        self.ax_psd = self.fig.add_subplot(212)
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
        self.plot_stop_time.insert(0, "5")
        
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
        self.analysis_stop_time.insert(0, "5")
        
        ttk.Button(analysis_time_frame, text="Update Analysis Range", command=self.update_analysis_range).pack(pady=5, fill=tk.X)
        
        # Analysis options frame
        analysis_frame = ttk.LabelFrame(self.control_frame, text="Analysis Options", padding="5")
        analysis_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(analysis_frame, text="Filter Cutoff (Hz):").pack()
        self.cutoff_freq = ttk.Entry(analysis_frame)
        self.cutoff_freq.pack(fill=tk.X)
        self.cutoff_freq.insert(0, "10")
        
        # Analysis buttons
        ttk.Button(analysis_frame, text="Plot Filtered Data", command=self.plot_filtered_data).pack(pady=5, fill=tk.X)
        ttk.Button(analysis_frame, text="Update PSD", command=self.update_psd).pack(pady=5, fill=tk.X)
        
        # RMSD display
        rmsd_frame = ttk.LabelFrame(self.control_frame, text="RMSD Result", padding="5")
        rmsd_frame.pack(pady=5, fill=tk.X)
        self.rmsd_text = tk.Text(rmsd_frame, height=1, width=20)
        self.rmsd_text.pack(pady=5, fill=tk.X)
        
    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.data_instance = Get_data(file_path)
                self.current_data = self.data_instance.data
                self.analysis_data = self.current_data
                self.update_plot_range()
                self.update_psd()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def update_plot_range(self):
        if self.data_instance is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            start = float(self.plot_start_time.get())
            stop = float(self.plot_stop_time.get())
            plot_data = self.data_instance.select(self.data_instance.data, start, stop)
            self.plot_time_series(plot_data)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid time values!")
    
    def update_analysis_range(self):
        if self.data_instance is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            start = float(self.analysis_start_time.get())
            stop = float(self.analysis_stop_time.get())
            self.analysis_data = self.data_instance.select(self.data_instance.data, start, stop)
            self.update_psd()
            self.calculate_rmsd()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid time values!")
    
    def plot_time_series(self, data):
        self.ax_time.clear()
        time_step = 0.00001
        num_data = len(data)
        time_variable = np.arange(start=0, stop=num_data, step=1) * time_step
        self.ax_time.plot(time_variable, data, linewidth=0.5)
        self.ax_time.set_xlabel('Time [s]')
        self.ax_time.set_ylabel('Normalized Transmission')
        self.canvas.draw()
    
    def plot_filtered_data(self):
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please select data first!")
            return
            
        try:
            cutoff = float(self.cutoff_freq.get())
            self.ax_time.clear()
            
            # Filter the data
            nyq = 0.5 * 100000  # sampling rate
            normal_cutoff = cutoff / nyq
            b, a = butter(4, normal_cutoff, 'low')
            filtered_data = filtfilt(b, a, self.current_data)
            
            # Plot both original and filtered data
            time_step = 0.00001
            num_data = len(self.current_data)
            time_variable = np.arange(start=0, stop=num_data, step=1) * time_step
            
            self.ax_time.plot(time_variable, self.current_data, linewidth=0.5, alpha=0.5)
            self.ax_time.plot(time_variable, filtered_data, linewidth=1, alpha=1)
            self.ax_time.set_xlabel('time/s')
            self.ax_time.set_ylabel('normalized APD voltage')
            self.canvas.draw()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid cutoff frequency!")
    
    def update_psd(self):
        if self.analysis_data is None:
            messagebox.showwarning("Warning", "Please select data first!")
            return
            
        self.ax_psd.clear()
        
        # Calculate PSD
        n = len(self.analysis_data)
        dt = 0.00001
        data_fft = np.abs(np.fft.fft(self.analysis_data)[:int(n/2)])
        data_psd = (1/(n*dt))*(np.abs(data_fft))**2
        freq = np.fft.fftfreq(n, dt)[:int(n/2)]
        freq = freq[1:10000]
        data_psd = data_psd[1:10000]
        
        # Plot PSD
        self.ax_psd.loglog(freq, data_psd, '.', markersize=2.5)
        
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
            
            self.ax_psd.loglog(freq, A/(freq**2 + fc**2), 'r-')
            self.ax_psd.set_xlabel('Frequency (Hz)')
            self.ax_psd.set_ylabel('Power Spectral Density ($V^2$/Hz)')
            self.ax_psd.set_title(f'Corner frequency: {round(fc, 2)} Hz')
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