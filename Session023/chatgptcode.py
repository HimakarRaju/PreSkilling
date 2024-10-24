import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import threading
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import seaborn as sns


class DataAnalyzerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Analysis App")

        # Frame for file selection and action buttons
        self.entry_frame = tk.Frame(master)
        self.entry_frame.pack(fill=tk.X)

        # Button to select files
        self.select_button = tk.Button(self.entry_frame, text="Select Files", command=self.select_files)
        self.select_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Button to analyze files
        self.analyze_button = tk.Button(self.entry_frame, text="Analyze Files", command=self.analyze_files)
        self.analyze_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create a horizontal PanedWindow for the tree view and the plot/error area
        self.paned_window = tk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Treeview for file display
        self.tree = ttk.Treeview(self.paned_window)
        self.paned_window.add(self.tree)

        # Scrollbar for treeview
        self.tree_scroll = tk.Scrollbar(self.tree, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a vertical PanedWindow for plots and error display
        self.plot_error_paned_window = tk.PanedWindow(self.paned_window, orient=tk.VERTICAL)
        self.paned_window.add(self.plot_error_paned_window)

        # Frame for plots
        self.plot_frame = tk.Frame(self.plot_error_paned_window, bg='lightgray', bd=2, relief=tk.SUNKEN)
        self.plot_error_paned_window.add(self.plot_frame)

        # Textbox for error messages
        self.error_display = ScrolledText(self.plot_error_paned_window, height=10, wrap=tk.WORD)
        self.plot_error_paned_window.add(self.error_display)

        # Queue for threading
        self.queue = queue.Queue()
        self.master.after(100, self.process_queue)

        # Bind double-click event on treeview
        self.tree.bind("<Double-1>", self.on_node_double_click)

        # Store the file paths associated with tree items
        self.file_paths = {}
        self.processed_files = set()

    def select_files(self):
        filetypes = [
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
            ("All files", "*.*")
        ]

        selected_files = filedialog.askopenfilenames(filetypes=filetypes)

        for file_path in selected_files:
            file_name = os.path.basename(file_path)
            self.tree.insert('', 'end', file_name, text=file_name)
            self.file_paths[file_name] = file_path

    def analyze_files(self):
        if not self.file_paths:
            self.queue.put(("", "No files selected for analysis."))
            return

        threads = []
        for file_name, file_path in self.file_paths.items():
            if file_path not in self.processed_files:
                thread = threading.Thread(target=self.analyze_file, args=(file_path, file_name))
                thread.start()
                threads.append(thread)
                self.processed_files.add(file_path)

        for thread in threads:
            thread.join()  # Wait for all threads to finish

    def analyze_file(self, file_path, file_name):
        node = file_name

        try:
            # Load data without context manager
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file type.")

            shape_info = f"Shape: {df.shape}"
            self.queue.put((node, shape_info))

            column_info = {col: str(df[col].dtype) for col in df.columns}
            for col, dtype in column_info.items():
                self.queue.put((node, f"Column: {col}, Type: {dtype}"))

            # Detailed Column Statistics
            self.calculate_detailed_statistics(df, node)

            # Plotting the data directly in the Tkinter window
            self.plot_data(df, node)

        except Exception as e:
            self.queue.put((node, f"Error: {str(e)}"))

    def calculate_detailed_statistics(self, df, node):
        """ Calculate and display detailed statistics for numerical columns. """
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numerical_cols.empty:
            stats = df[numerical_cols].describe().T
            for col in numerical_cols:
                mean = df[col].mean()
                median = df[col].median()
                mode = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
                std_dev = df[col].std()
                self.queue.put((node,
                                f"Statistics for {col}: Mean={mean:.2f}, Median={median:.2f}, Mode={mode}, Std Dev={std_dev:.2f}"))

    def plot_data(self, df, file_name):
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        try:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            fig = plt.figure(figsize=(10, 10))

            # Create subplots and plot histograms, box plots, and scatter plots
            ax1 = fig.add_subplot(221)
            for col in numerical_cols[:4]:
                sns.histplot(df[col], bins=30, kde=True, ax=ax1, color='blue', alpha=0.7)
            ax1.set_title('Histograms of Numerical Columns')

            ax2 = fig.add_subplot(222)
            if numerical_cols.size > 0:
                sns.boxplot(x=df[numerical_cols[0]], ax=ax2, color='lightblue')
                ax2.set_title(f'Box Plot of {numerical_cols[0]}')

            ax3 = fig.add_subplot(223)
            if numerical_cols.size > 1:
                sns.scatterplot(x=df[numerical_cols[0]], y=df[numerical_cols[1]], ax=ax3, color='orange')
                ax3.set_title(f'Scatter Plot of {numerical_cols[0]} vs {numerical_cols[1]}')

            if numerical_cols.size > 2:
                ax4 = fig.add_subplot(224, projection='3d')
                ax4.scatter(df[numerical_cols[0]], df[numerical_cols[1]], df[numerical_cols[2]], color='green')
                ax4.set_title('3D Scatter Plot')

            # Display plots
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            self.queue.put((file_name, f"Error while plotting: {str(e)}"))

    def process_queue(self):
        while not self.queue.empty():
            node, message = self.queue.get()
            self.tree.insert(node, 'end', text=message)
            self.error_display.insert(tk.END, message + '\n')

        self.master.after(100, self.process_queue)

    def on_node_double_click(self, event):
        item = self.tree.selection()[0]
        file_name = self.tree.item(item)['text']
        file_path = self.file_paths.get(file_name)

        if file_path and file_path not in self.processed_files:
            threading.Thread(target=self.analyze_file, args=(file_path, file_name)).start()
            self.processed_files.add(file_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalyzerApp(root)
    root.mainloop()
