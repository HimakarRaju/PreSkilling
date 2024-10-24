import tkinter as tk
from tkinter import ttk, messagebox
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.decomposition import PCA
from threading import Thread

from utils import log_status  # Ensure you import your log_status function here


class PlotManager:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def setup_plot_options_frame(self, main_frame):
        plot_options_frame = ttk.Labelframe(main_frame, text="Plot Options", padding=10)
        plot_options_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        plot_type_label = ttk.Label(plot_options_frame, text="Select Plot Type:")
        plot_type_label.grid(row=0, column=0, padx=5, pady=5)
        plot_type_menu = ttk.Combobox(
            plot_options_frame,
            textvariable=plot_type,
            values=["pairplot", "scatterplot", "heatmap", "histogram", "PCA"],
        )
        plot_type_menu.grid(row=0, column=1, padx=5, pady=5)
        column_label = ttk.Label(plot_options_frame, text="Select Columns for X and Y:")
        column_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        x_dropdown = ttk.Combobox(plot_options_frame, textvariable=x_column)
        x_dropdown.grid(row=2, column=0, padx=5, pady=5)
        y_dropdown = ttk.Combobox(plot_options_frame, textvariable=y_column)
        y_dropdown.grid(row=2, column=1, padx=5, pady=5)
        plot_button = ttk.Button(
            plot_options_frame,
            text="Generate Plot",
            bootstyle=SUCCESS,
            command=lambda: Thread(target=self.generate_plot).start(),
        )
        plot_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

    def generate_corr_heatmap(self, root):
        if self.data_handler.dataset:
            plot_window = tk.Toplevel(root)
            plot_window.title("Correlation Heatmap")
            plot_window.geometry("600x600")
            fig, ax = plt.subplots()
            data = self.data_handler.dataset[self.data_handler.current_dataset_name]
            numerical_cols = data.select_dtypes(include="number")
            sns.heatmap(numerical_cols.corr(), annot=True, fmt=".2f", ax=ax)
            plt.tight_layout()
            log_status(
                f"Generated correlation heatmap for {self.data_handler.current_dataset_name}"
            )
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            canvas.get_tk_widget().pack()

    def generate_plot(self, root, plot_type, x_column, y_column):
        plot_type = plot_type.get()
        x_col = x_column.get()
        y_col = y_column.get()
        if not x_col or not y_col:
            messagebox.showerror("Error", "Please select both X and Y columns!")
            return
        data = self.data_handler.dataset[self.data_handler.current_dataset_name]
        plot_window = tk.Toplevel(root)
        plot_window.title(f"{plot_type} Plot")
        plot_window.geometry("800x800")
        fig, ax = plt.subplots()
        if plot_type == "scatterplot":
            sns.scatterplot(x=x_col, y=y_col, data=data, ax=ax)
        elif plot_type == "pairplot":
            sns.pairplot(data[[x_col, y_col]], ax=ax)
        elif plot_type == "heatmap":
            sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax)
        elif plot_type == "PCA":
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data.select_dtypes(include="number"))
            data["PCA1"] = pca_result[:, 0]
            data["PCA2"] = pca_result[:, 1]
            sns.scatterplot(x="PCA1", y="PCA2", data=data, ax=ax)
        else:
            messagebox.showerror("Error", f"Plot type {plot_type} not supported!")
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack()
        log_status(
            f"Generated {plot_type} for {self.data_handler.current_dataset_name}"
        )
