import tkinter
from pandastable import Table
import ttkbootstrap as ttk
from matplotlib import pyplot as plt
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading

pd.set_option("display.max_columns", None)


class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exploratory Data Analysis Tool")
        self.root.geometry("1000x1200")

        # Set modern theme
        self.style = ttk.Style("darkly")
        self.dataset = {}
        self.current_dataset_name = None

        # Menu
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Load Datasets", command=self.load_multiple_datasets
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Dataset Information Frame
        self.dataset_info_frame = ttk.Labelframe(
            main_frame, text="Dataset Info", padding=10
        )
        self.dataset_info_frame.grid(row=0, column=0, sticky="nsew")

        self.text_area = tk.Text(self.dataset_info_frame, height=20, width=60)
        self.text_area.pack()

        # Dataset Selector
        self.dataset_selector = ttk.Combobox(self.dataset_info_frame)
        self.dataset_selector.pack(padx=5, pady=5)
        self.dataset_selector.bind("<<ComboboxSelected>>", self.update_columns)

        # Plot Options Frame
        self.plot_options_frame = ttk.Labelframe(
            main_frame, text="Plot Options", padding=10
        )
        self.plot_options_frame.grid(row=0, column=1, padx=20, pady=20)

        self.plot_type = tk.StringVar(value="scatterplot")
        plot_type_label = ttk.Label(self.plot_options_frame, text="Select Plot Type:")
        plot_type_label.grid(row=0, column=0, padx=5, pady=5)

        plot_type_menu = ttk.Combobox(
            self.plot_options_frame,
            textvariable=self.plot_type,
            values=["pairplot", "scatterplot", "heatmap", "histogram", "PCA"],
        )
        plot_type_menu.grid(row=0, column=1, padx=5, pady=5)

        column_label = ttk.Label(
            self.plot_options_frame, text="Select Columns for X and Y:"
        )
        column_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.x_column = tk.StringVar()
        self.x_dropdown = ttk.Combobox(
            self.plot_options_frame, textvariable=self.x_column
        )
        self.x_dropdown.grid(row=2, column=0, padx=5, pady=5)

        self.y_column = tk.StringVar()
        self.y_dropdown = ttk.Combobox(
            self.plot_options_frame, textvariable=self.y_column
        )
        self.y_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Generate Plot Button
        plot_button = ttk.Button(
            self.plot_options_frame,
            text="Generate Plot",
            bootstyle=SUCCESS,
            command=self.generate_plot,
        )
        plot_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        # Dataset Actions Frame
        self.action_frame = ttk.Labelframe(main_frame, text="Actions", padding=10)
        self.action_frame.grid(row=1, column=0, columnspan=2, pady=20)

        show_button = ttk.Button(
            self.action_frame,
            text="Show Dataset Head",
            bootstyle=INFO,
            command=self.show_data,
        )
        show_button.grid(row=0, column=0, padx=10)

        stats_button = ttk.Button(
            self.action_frame,
            text="Show Summary Statistics",
            bootstyle=INFO,
            command=self.show_statistics,
        )
        stats_button.grid(row=0, column=1, padx=10)

        corr_button = ttk.Button(
            self.action_frame,
            text="Show Correlation Heatmap",
            bootstyle=INFO,
            command=self.generate_corr_heatmap,
        )
        corr_button.grid(row=0, column=2, padx=10)

        clean_button = ttk.Button(
            self.action_frame,
            text="Open Cleaning Tools",
            bootstyle=WARNING,
            command=self.open_cleaning_window,
        )
        clean_button.grid(row=0, column=3, padx=10)

        # Status Label (larger size)
        self.status_label = ttk.Label(
            main_frame,
            text="No dataset loaded.",
            relief=tk.SUNKEN,
            anchor="w",
            font=("Arial", 12),
            padding=10,
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="we")

        # Configure grid weights
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def load_multiple_datasets(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            self.status_label.config(text="Loading datasets...")
            threading.Thread(target=self.process_datasets, args=(file_paths,)).start()

    def process_datasets(self, file_paths):
        for path in file_paths:
            dataset_name = path.split("/")[-1]
            try:
                self.dataset[dataset_name] = pd.read_csv(path)
                self.current_dataset_name = dataset_name
                self.dataset_selector["values"] = list(self.dataset.keys())
                self.dataset_selector.set(self.current_dataset_name)
                self.update_columns()
                self.status_label.config(text=f"Loaded {dataset_name}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to load {dataset_name}: {str(e)}"
                )
                self.status_label.config(text=f"Failed to load {dataset_name}")

    def update_columns(self, event=None):
        if self.current_dataset_name:
            data = self.dataset[self.current_dataset_name]
            columns = list(data.columns)
            self.x_dropdown["values"] = columns
            self.y_dropdown["values"] = columns

    def show_data(self):
        if self.dataset:
            self.text_area.delete(1.0, tk.END)
            for name, data in self.dataset.items():
                self.text_area.insert(tk.END, f"{name}:\n{data.head()}\n\n")
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def show_statistics(self):
        if self.dataset:
            self.text_area.delete(1.0, tk.END)
            for name, data in self.dataset.items():
                self.text_area.insert(tk.END, f"{name} Summary:\n{data.describe()}\n\n")
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def generate_corr_heatmap(self):
        if self.dataset:
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Correlation Heatmap")
            plot_window.geometry("1000x1000")
            fig, ax = plt.subplots()
            data = self.dataset[self.current_dataset_name]
            numerical_cols = data.select_dtypes(include="number")
            sns.heatmap(numerical_cols.corr(), annot=True, fmt=".2f", ax=ax)
            plt.tight_layout()
            self.status_label.config(
                text=f"Generated heatmap for {self.current_dataset_name}"
            )

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def open_cleaning_window(self):
        if self.dataset:
            clean_window = tk.Toplevel(self.root)
            clean_window.title("Data Cleaning Tools")
            clean_window.geometry("800x600")

            # Display table using pandastable
            frame = ttk.Frame(clean_window)
            frame.pack(fill=tk.BOTH, expand=1)
            pt = Table(frame, dataframe=self.dataset[self.current_dataset_name])
            pt.show()

            # Add preprocessing buttons (Example: Fill missing values, drop columns, etc.)
            fill_button = ttk.Button(
                clean_window,
                text="Fill Missing Values",
                bootstyle=DANGER,
                command=self.fill_missing,
            )
            fill_button.pack(padx=10, pady=5)

        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def fill_missing(self):
        if self.dataset:
            data = self.dataset[self.current_dataset_name]
            data.fillna(data.mean(), inplace=True)
            self.status_label.config(
                text=f"Filled missing values in {self.current_dataset_name}"
            )

    def generate_plot(self):
        if self.dataset:
            plot_type = self.plot_type.get()
            x_col = self.x_column.get()
            y_col = self.y_column.get()

            plot_window = tk.Toplevel(self.root)
            plot_window.title(f"{plot_type.capitalize()} Plot")
            plot_window.geometry("800x600")

            fig, ax = plt.subplots()

            data = self.dataset[self.current_dataset_name]
            if plot_type == "PCA":
                self.generate_pca(plot_window)
            elif plot_type == "pairplot":
                sns.pairplot(data)
            elif plot_type == "scatterplot":
                if x_col and y_col:
                    sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)
            elif plot_type == "histogram":
                if x_col:
                    sns.histplot(data[x_col], ax=ax)
            elif plot_type == "heatmap":
                sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax)
            self.status_label.config(
                text=f"Generated plot for {self.current_dataset_name}"
            )

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def generate_pca(self, plot_window):
        data = self.dataset[self.current_dataset_name].select_dtypes(
            include=[np.number]
        )
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)

        df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

        fig = px.scatter(df_pca, x="PC1", y="PC2")
        fig.show()


if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = EDAApp(root)
    root.mainloop()
