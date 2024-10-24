import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
from threading import Thread
import psutil  # For CPU and RAM usage
from IPython.terminal.embed import InteractiveShellEmbed
import io


# Create the main Tkinter app window
class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EDA Tool - Multiple Datasets")
        self.root.geometry("1000x600")

        # Create a frame to hold resource usage on the top-left, below the close button
        self.resource_frame = tk.Frame(root)
        self.resource_frame.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cpu_label = tk.Label(self.resource_frame, text="CPU: 0%")
        self.cpu_label.pack(side=tk.LEFT, padx=5)
        self.ram_label = tk.Label(self.resource_frame, text="RAM: 0%")
        self.ram_label.pack(side=tk.LEFT)
        self.update_resource_usage()

        # Load dataset button moved to the menu
        self.load_button = ttk.Button(
            root, text="Load Dataset(s)", command=self.load_datasets
        )
        self.load_button.grid(row=0, column=1, padx=5, pady=5, sticky="e")

        # File explorer panel to list datasets
        self.file_explorer_frame = tk.Frame(root)
        self.file_explorer_frame.grid(row=1, column=0, sticky="nsew")
        self.file_explorer = tk.Listbox(self.file_explorer_frame, height=15)
        self.file_explorer.pack(fill=tk.BOTH, expand=True)
        self.file_explorer.bind("<Double-Button-1>", self.on_dataset_double_click)

        # Status bar with scroll bar for updates
        self.status_frame = tk.Frame(root)
        self.status_frame.grid(row=2, column=0, columnspan=3, sticky="nsew")
        self.status_bar = scrolledtext.ScrolledText(self.status_frame, height=8)
        self.status_bar.pack(fill=tk.BOTH, expand=True)

        # Main data view panel (read-only, with scrollbar)
        self.data_view_frame = tk.Frame(root)
        self.data_view_frame.grid(row=1, column=1, sticky="nsew")
        self.data_view = tk.Text(self.data_view_frame, wrap=tk.NONE, state=tk.DISABLED)
        self.data_view.pack(fill=tk.BOTH, expand=True)

        # Scrollbars for the main data view
        self.data_view_scroll_y = tk.Scrollbar(
            self.data_view_frame, orient=tk.VERTICAL, command=self.data_view.yview
        )
        self.data_view_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_view_scroll_x = tk.Scrollbar(
            self.data_view_frame, orient=tk.HORIZONTAL, command=self.data_view.xview
        )
        self.data_view_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_view.config(
            yscrollcommand=self.data_view_scroll_y.set,
            xscrollcommand=self.data_view_scroll_x.set,
        )

        # IPython interactive shell at the bottom
        self.ipython_output = scrolledtext.ScrolledText(root, height=10)
        self.ipython_output.grid(
            row=3, column=0, columnspan=3, sticky="nsew", padx=5, pady=5
        )

        # Setting grid column and row weights to make widgets auto-resize
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(3, weight=1)

        # Keep track of loaded datasets
        self.datasets = {}

        # Initialize IPython shell for operations
        self.shell = InteractiveShellEmbed()
        self.setup_ipython_redirection()

    def setup_ipython_redirection(self):
        """Redirect IPython output to the text box."""
        ipython_io = io.StringIO()
        self.shell.push(
            {"print": lambda msg: self.ipython_output.insert(tk.END, msg + "\n")}
        )

    def update_resource_usage(self):
        """Updates the CPU and RAM usage on the resource panel."""
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent
        self.cpu_label.config(text=f"CPU: {cpu_percent}%")
        self.ram_label.config(text=f"RAM: {ram_percent}%")
        self.root.after(1000, self.update_resource_usage)  # Update every second

    def log_status(self, message):
        """Logs the message to the status bar."""
        self.status_bar.insert(tk.END, message + "\n")
        self.status_bar.see(tk.END)

    def load_datasets(self):
        """Loads one or more datasets and adds them to the explorer."""
        file_paths = filedialog.askopenfilenames(
            title="Select dataset(s)", filetypes=[("CSV Files", "*.csv")]
        )
        if file_paths:
            self.log_status("Loading datasets...")
            thread = Thread(target=self.load_data_thread, args=(file_paths,))
            thread.start()

    def load_data_thread(self, file_paths):
        """Thread for loading datasets in the background."""
        for file_path in file_paths:
            try:
                dataset_name = file_path.split("/")[-1]
                df = pd.read_csv(file_path)
                self.datasets[dataset_name] = df
                self.file_explorer.insert(tk.END, dataset_name)  # Add to file explorer
                self.log_status(
                    f"Loaded: {dataset_name} ({df.shape[0]} rows, {df.shape[1]} columns)"
                )
            except Exception as e:
                self.log_status(f"Error loading {file_path}: {e}")

    def on_dataset_double_click(self, event):
        """Handles double-click on a dataset in the file explorer to load it."""
        selected_idx = self.file_explorer.curselection()
        if selected_idx:
            dataset_name = self.file_explorer.get(selected_idx)
            self.update_data_view(dataset_name)
            self.shell.run_cell(f"active_dataset = '{dataset_name}'")
            self.log_status(f"Dataset '{dataset_name}' is now active in IPython.")

    def update_data_view(self, dataset_name):
        """Displays the head of the selected dataset in a tabular, read-only format."""
        df = self.datasets.get(dataset_name)
        if df is not None:
            self.data_view.config(state=tk.NORMAL)  # Allow changes
            self.data_view.delete(1.0, tk.END)  # Clear previous content
            self.data_view.insert(
                tk.END, df.head().to_string()
            )  # Insert head of DataFrame
            self.data_view.config(state=tk.DISABLED)  # Make it read-only again
            self.log_status(f"Displaying head of '{dataset_name}'.")

# Running the app
if __name__ == "__main__":
    root = tk.Tk()
    app = EDAApp(root)
    root.mainloop()
