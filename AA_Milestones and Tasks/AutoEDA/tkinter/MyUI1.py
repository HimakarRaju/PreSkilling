import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
from threading import Thread
import IPython
from IPython.terminal.embed import InteractiveShellEmbed
import io


# Create the main Tkinter app window
class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EDA Tool - Multiple Datasets")
        self.root.geometry("1000x600")

        # Dropdown for loaded datasets
        self.dataset_dropdown = ttk.Combobox(root, state="readonly")
        self.dataset_dropdown.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Load datasets button
        self.load_button = ttk.Button(
            root, text="Load Dataset(s)", command=self.load_datasets
        )
        self.load_button.grid(row=0, column=1, padx=5, pady=5)

        # Status bar with scroll bar for updates
        self.status_frame = tk.Frame(root)
        self.status_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        self.status_bar = scrolledtext.ScrolledText(self.status_frame, height=8)
        self.status_bar.pack(fill=tk.BOTH, expand=True)

        # VS Code-like file explorer panel (placeholder)
        self.file_explorer = tk.Listbox(root, height=15)
        self.file_explorer.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        # Main data view panel (placeholder for displaying data)
        self.data_view = tk.Text(root, wrap=tk.NONE)
        self.data_view.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

        # IPython interactive shell at the bottom
        self.ipython_output = scrolledtext.ScrolledText(root, height=10)
        self.ipython_output.grid(
            row=3, column=0, columnspan=3, sticky="nsew", padx=5, pady=5
        )

        # Setting grid column and row weights to make widgets auto-resize
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)

        # Keep track of loaded datasets
        self.datasets = {}

        # Initialize IPython shell for operations
        self.shell = InteractiveShellEmbed()

        # Redirect IPython output to the app's text box
        self.setup_ipython_redirection()

    def setup_ipython_redirection(self):
        # Redirect stdout to display in the IPython output area
        ipython_io = io.StringIO()
        self.shell.push(
            {"print": lambda msg: self.ipython_output.insert(tk.END, msg + "\n")}
        )

    def log_status(self, message):
        """Logs the message to the status bar."""
        self.status_bar.insert(tk.END, message + "\n")
        self.status_bar.see(tk.END)

    def load_datasets(self):
        """Loads one or more datasets and adds them to the dropdown menu."""
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
                self.log_status(
                    f"Loaded: {dataset_name} ({df.shape[0]} rows, {df.shape[1]} columns)"
                )
                # Update the dropdown menu
                self.dataset_dropdown["values"] = list(self.datasets.keys())
            except Exception as e:
                self.log_status(f"Error loading {file_path}: {e}")

    def update_data_view(self, dataset_name):
        """Updates the main data view panel with the dataset content."""
        df = self.datasets.get(dataset_name)
        if df is not None:
            self.data_view.delete(1.0, tk.END)
            self.data_view.insert(tk.END, df.head().to_string())

    def execute_ipython_command(self, command):
        """Executes an IPython command and prints output."""
        self.shell.run_cell(command)


# Running the app
if __name__ == "__main__":
    root = tk.Tk()
    app = EDAApp(root)
    root.mainloop()
