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

pd.set_option("display.max_columns", None)


class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exploratory Data Analysis Tool")
        self.root.geometry("1000x1200")

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
            self.log_status(f"Generated heatmap for {self.current_dataset_name}")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def open_cleaning_window(self):
        if self.dataset:
            clean_window = tk.Toplevel(self.root)
            clean_window.title("Data Cleaning Tools")
            clean_window.geometry("800x800")

            # Add Data Profiling Button (using ydata-profiling)
            profile_button = ttk.Button(
                clean_window,
                text="Generate Data Profile",
                bootstyle=PRIMARY,
                command=self.generate_profile_report,
            )
            profile_button.pack(padx=10, pady=5)

            # Add pandastable to visualize data
            frame = ttk.Frame(clean_window)
            frame.pack(fill=tk.BOTH, expand=True)
            pt = Table(frame, dataframe=self.dataset[self.current_dataset_name])
            pt.show()

            # Add preprocessing buttons (e.g., fill missing values)
            fill_button = ttk.Button(
                clean_window,
                text="Fill Missing Values",
                bootstyle=DANGER,
                command=self.fill_missing,
            )
            fill_button.pack(padx=10, pady=5)

        else:
            messagebox.showerror("Error", "No dataset loaded!")

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

    def fill_missing(self):
        if self.dataset:
            data = self.dataset[self.current_dataset_name]
            data.fillna(data.mean(), inplace=True)
            self.log_status(f"Filled missing values in {self.current_dataset_name}")

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
            self.log_status(f"Generated plot for {self.current_dataset_name}")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def generate_pca(self, plot_window):
        data = self.dataset[self.current_dataset_name].select_dtypes(
            include=[np.number]
        )
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)

        df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

        feature_names = data.columns

        fig = px.scatter(df_pca, x="PC1", y="PC2")

        for i, feature_name in enumerate(feature_names):
            fig.add_annotation(
                x=df_pca["PC1"][i],
                y=df_pca["PC2"][i],
                text=feature_name,
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
            )
        fig.show()

    def open_status_window(self):
        status_window = tk.Toplevel(self.root)
        status_window.title("Status Logs")
        status_window.geometry("800x600")

        status_area = scrolledtext.ScrolledText(status_window, wrap=tk.WORD)
        status_area.pack(expand=True, fill="both")

        # Load all current logs into the status window
        for log in self.status_logs:
            status_area.insert(tk.END, f"{log}\n")

        # Scroll to the latest log
        status_area.see(tk.END)

    def log_status(self, message):
        self.status_logs.append(message)


if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = EDAApp(root)
    root.mainloop()
