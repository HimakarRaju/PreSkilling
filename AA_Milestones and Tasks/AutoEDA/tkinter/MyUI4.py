import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk
from matplotlib import pyplot as plt
import plotly.express as px
from ttkbootstrap.constants import *
from pandastable import Table
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from IPython.terminal.embed import InteractiveShellEmbed
from threading import Thread

pd.set_option("display.max_columns", None)


class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exploratory Data Analysis Tool")
        self.root.geometry("1000x1200")

        self.style = ttk.Style("darkly")
        self.dataset = {}
        self.current_dataset_name = None

        # IPython Terminal Embed
        self.ipython_shell = InteractiveShellEmbed()

        # Menu
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Load Datasets", command=self.load_multiple_datasets
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Modelling Menu
        modelling_menu = tk.Menu(menubar, tearoff=0)
        modelling_menu.add_command(
            label="Logistic Regression", command=self.run_logistic_regression
        )
        modelling_menu.add_command(
            label="Random Forest", command=self.run_random_forest
        )
        menubar.add_cascade(label="Modelling", menu=modelling_menu)

        # Status Bar Window Menu
        menubar.add_command(label="Status Window", command=self.open_status_window)
        self.root.config(menu=menubar)

        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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
        self.plot_options_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

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
            command=lambda: Thread(target=self.generate_plot).start(),
        )
        plot_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        # Dataset Actions Frame
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
            command=self.open_cleaning_window,
        )
        clean_button.grid(row=0, column=3, padx=10)

        # IPython Frame
        self.ipython_frame = ttk.Labelframe(
            main_frame, text="IPython Terminal", padding=10
        )
        self.ipython_frame.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew"
        )

        self.ipython_entry = ttk.Entry(self.ipython_frame, width=100)
        self.ipython_entry.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.ipython_entry.bind("<Return>", self.execute_ipython_command)

        self.ipython_output = scrolledtext.ScrolledText(self.ipython_frame, height=10)
        self.ipython_output.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Initialize status list to capture all status logs
        self.status_logs = []

    def load_multiple_datasets(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            self.log_status("Loading datasets...")
            for path in file_paths:
                dataset_name = path.split("/")[-1]
                try:
                    self.dataset[dataset_name] = pd.read_csv(path)
                    self.current_dataset_name = dataset_name
                    self.dataset_selector["values"] = list(self.dataset.keys())
                    self.dataset_selector.set(self.current_dataset_name)
                    self.update_columns()
                    self.log_status(f"Loaded {dataset_name}")
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"Failed to load {dataset_name}: {str(e)}"
                    )
                    self.log_status(f"Failed to load {dataset_name}")

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
            plot_window.geometry("600x600")
            fig, ax = plt.subplots()
            data = self.dataset[self.current_dataset_name]
            numerical_cols = data.select_dtypes(include="number")
            sns.heatmap(numerical_cols.corr(), annot=True, fmt=".2f", ax=ax)
            plt.tight_layout()
            self.log_status(
                f"Generated correlation heatmap for {self.current_dataset_name}"
            )

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            canvas.get_tk_widget().pack()

    def generate_plot(self):
        plot_type = self.plot_type.get()
        x_col = self.x_column.get()
        y_col = self.y_column.get()

        if not x_col or not y_col:
            messagebox.showerror("Error", "Please select both X and Y columns!")
            return

        data = self.dataset[self.current_dataset_name]
        plot_window = tk.Toplevel(self.root)
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

        self.log_status(f"Generated {plot_type} for {self.current_dataset_name}")

    def generate_profile_report(self):
        if self.dataset:
            self.log_status(
                f"Generating profile report for {self.current_dataset_name}..."
            )
            data = self.dataset[self.current_dataset_name]
            profile = ProfileReport(
                data,
                title=f"Profiling Report for {self.current_dataset_name}",
                explorative=True,
            )
            profile.to_file(f"{self.current_dataset_name}_profile_report.html")
            self.log_status(
                f"Generated profile report for {self.current_dataset_name} (saved as HTML)"
            )
            print(
                f"Generated profile report for {self.current_dataset_name} (saved as HTML)"
            )
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def run_logistic_regression(self):
        if self.dataset:
            data = self.dataset[self.current_dataset_name]
            try:
                X = data.select_dtypes(include=[np.number]).dropna()
                y = data[
                    "target"
                ]  # Adjust the target column name based on your dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.log_status(
                    f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}"
                )
                self.log_status(
                    f"Classification Report:\n{classification_report(y_test, y_pred)}"
                )
            except KeyError:
                messagebox.showerror("Error", "Target column not found in the dataset.")
            except Exception as e:
                self.log_status(f"Failed to run Logistic Regression: {str(e)}")
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def run_random_forest(self):
        if self.dataset:
            data = self.dataset[self.current_dataset_name]
            try:
                X = data.select_dtypes(include=[np.number]).dropna()
                y = data[
                    "target"
                ]  # Adjust the target column name based on your dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.log_status(
                    f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)}"
                )
                self.log_status(
                    f"Classification Report:\n{classification_report(y_test, y_pred)}"
                )
            except KeyError:
                messagebox.showerror("Error", "Target column not found in the dataset.")
            except Exception as e:
                self.log_status(f"Failed to run Random Forest: {str(e)}")
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def execute_ipython_command(self, event=None):
        command = self.ipython_entry.get()
        self.ipython_output.insert(
            tk.END, f"In [{len(self.status_logs)+1}]: {command}\n"
        )
        try:
            result = self.ipython_shell.run_cell(command)
            self.ipython_output.insert(
                tk.END, f"Out [{len(self.status_logs)+1}]: {result}\n"
            )
            self.ipython_entry.delete(0, tk.END)
        except Exception as e:
            self.ipython_output.insert(tk.END, f"Error: {str(e)}\n")

    def log_status(self, message):
        self.status_logs.append(message)
        print(f"[Status Log] {message}")

    def open_cleaning_window(self):
        messagebox.showinfo("Cleaning Tools", "Coming Soon!")

    def open_status_window(self):
        status_window = tk.Toplevel(self.root)
        status_window.title("Status Logs")
        status_window.geometry("400x400")
        log_text = tk.Text(status_window, wrap=tk.WORD)
        log_text.pack(fill="both", expand=True)

        for log in self.status_logs:
            log_text.insert(tk.END, log + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = EDAApp(root)
    root.mainloop()
