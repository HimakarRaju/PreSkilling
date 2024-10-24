import pandas as pd
from tkinter import filedialog, messagebox

from utils import log_status  # Ensure you import your log_status function here

class DataHandler:
    def __init__(self):
        self.dataset = {}
        self.current_dataset_name = None

    def load_multiple_datasets(self, dataset_selector, text_area):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            log_status("Loading datasets...")
            for path in file_paths:
                dataset_name = path.split("/")[-1]
                try:
                    self.dataset[dataset_name] = pd.read_csv(path)
                    self.current_dataset_name = dataset_name
                    dataset_selector["values"] = list(self.dataset.keys())
                    dataset_selector.set(self.current_dataset_name)
                    self.update_columns(dataset_selector)
                    log_status(f"Loaded {dataset_name}")
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"Failed to load {dataset_name}: {str(e)}"
                    )
                    log_status(f"Failed to load {dataset_name}")

    def update_columns(self, dataset_selector, x_dropdown, y_dropdown):
        if self.current_dataset_name:
            data = self.dataset[self.current_dataset_name]
            columns = list(data.columns)
            x_dropdown["values"] = columns
            y_dropdown["values"] = columns

    def show_data(self, text_area):
        if self.dataset:
            text_area.delete(1.0, tk.END)
            for name, data in self.dataset.items():
                text_area.insert(tk.END, f"{name}:\n{data.head()}\n\n")
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def show_statistics(self, text_area):
        if self.dataset:
            text_area.delete(1.0, tk.END)
            for name, data in self.dataset.items():
                text_area.insert(tk.END, f"{name} Summary:\n{data.describe()}\n\n")
        else:
            messagebox.showerror("Error", "No dataset loaded!")
