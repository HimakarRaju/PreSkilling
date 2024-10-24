import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from IPython.terminal.embed import InteractiveShellEmbed
from threading import Thread
import pandas as pd

from data_handler import DataHandler
from plot_manager import PlotManager
from utils import log_status, open_cleaning_window, execute_ipython_command


class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exploratory Data Analysis Tool")
        self.root.geometry("1000x1200")
        self.style = ttk.Style("darkly")
        self.data_handler = DataHandler()
        self.plot_manager = PlotManager(self.data_handler)

        # Menu
        menubar = tk.Menu(self.root)
        self.setup_menus(menubar)
        self.root.config(menu=menubar)

        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.setup_frames(main_frame)

    def setup_menus(self, menubar):
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Load Datasets", command=self.load_multiple_datasets
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        modelling_menu = tk.Menu(menubar, tearoff=0)
        modelling_menu.add_command(
            label="Logistic Regression", command=self.run_logistic_regression
        )
        modelling_menu.add_command(
            label="Random Forest", command=self.run_random_forest
        )
        menubar.add_cascade(label="Modelling", menu=modelling_menu)
        menubar.add_command(label="Status Window", command=self.open_status_window)

    def setup_frames(self, main_frame):
        # Dataset Information Frame
        self.dataset_info_frame = ttk.Labelframe(
            main_frame, text="Dataset Info", padding=10
        )
        self.dataset_info_frame.grid(row=0, column=0, sticky="nsew")
        self.text_area = tk.Text(self.dataset_info_frame, height=20, width=60)
        self.text_area.pack()

        self.dataset_selector = ttk.Combobox(self.dataset_info_frame)
        self.dataset_selector.pack(padx=5, pady=5)
        self.dataset_selector.bind("<<ComboboxSelected>>", self.update_columns)

        # Plot Options Frame
        self.plot_manager.setup_plot_options_frame(main_frame)

        # Dataset Actions Frame
        self.setup_action_frame(main_frame)

        # IPython Frame
        self.setup_ipython_frame(main_frame)

    def setup_action_frame(self, main_frame):
        self.action_frame = ttk.Labelframe(main_frame, text="Actions", padding=10)
        self.action_frame.grid(row=1, column=0, columnspan=2, pady=20, sticky="nsew")
        show_button = ttk.Button(
            self.action_frame,
            text="Show Dataset Head",
            bootstyle=INFO,
            command=lambda: Thread(target=self.show_data).start(),
        )
        show_button.grid(row=0, column=0, padx=10)
        stats_button = ttk.Button(
            self.action_frame,
            text="Show Summary Statistics",
            bootstyle=INFO,
            command=lambda: Thread(target=self.show_statistics).start(),
        )
        stats_button.grid(row=0, column=1, padx=10)
        corr_button = ttk.Button(
            self.action_frame,
            text="Show Correlation Heatmap",
            bootstyle=INFO,
            command=lambda: Thread(target=self.generate_corr_heatmap).start(),
        )
        corr_button.grid(row=0, column=2, padx=10)
        clean_button = ttk.Button(
            self.action_frame,
            text="Open Cleaning Tools",
            bootstyle=WARNING,
            command=open_cleaning_window,
        )
        clean_button.grid(row=0, column=3, padx=10)

    def setup_ipython_frame(self, main_frame):
        self.ipython_frame = ttk.Labelframe(
            main_frame, text="IPython Terminal", padding=10
        )
        self.ipython_frame.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew"
        )
        self.ipython_entry = ttk.Entry(self.ipython_frame, width=100)
        self.ipython_entry.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.ipython_entry.bind("<Return>", execute_ipython_command)
        self.ipython_output = scrolledtext.ScrolledText(self.ipython_frame, height=10)
        self.ipython_output.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

    def load_multiple_datasets(self):
        self.data_handler.load_multiple_datasets(self.dataset_selector, self.text_area)

    def update_columns(self, event=None):
        self.data_handler.update_columns(
            self.dataset_selector, self.x_dropdown, self.y_dropdown
        )

    def show_data(self):
        self.data_handler.show_data(self.text_area)

    def show_statistics(self):
        self.data_handler.show_statistics(self.text_area)

    def generate_corr_heatmap(self):
        self.plot_manager.generate_corr_heatmap(self.root)

    def generate_plot(self):
        self.plot_manager.generate_plot(
            self.root, self.plot_type, self.x_column, self.y_column
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = EDAApp(root)
    root.mainloop()
