import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exploratory Data Analysis Tool")
        self.root.geometry("1000x700")

        self.dataset = None

        # Menu
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Dataset", command=self.load_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Main Frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Dataset Information Frame
        self.dataset_info_frame = tk.LabelFrame(
            main_frame, text="Dataset Info", padx=10, pady=10
        )
        self.dataset_info_frame.grid(row=0, column=0, sticky="nsew")

        self.text_area = tk.Text(self.dataset_info_frame, height=10, width=50)
        self.text_area.pack()

        # Plot Options Frame
        self.plot_options_frame = tk.LabelFrame(
            main_frame, text="Plot Options", padx=10, pady=10
        )
        self.plot_options_frame.grid(row=0, column=1, padx=20, pady=20)

        self.plot_type = tk.StringVar(value="pairplot")
        plot_type_label = tk.Label(self.plot_options_frame, text="Select Plot Type:")
        plot_type_label.grid(row=0, column=0, padx=5, pady=5)

        plot_type_menu = ttk.Combobox(
            self.plot_options_frame,
            textvariable=self.plot_type,
            values=["pairplot", "scatterplot", "heatmap", "histogram"],
        )
        plot_type_menu.grid(row=0, column=1, padx=5, pady=5)

        column_label = tk.Label(
            self.plot_options_frame, text="Select Columns for X and Y (Optional):"
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
        plot_button = tk.Button(
            self.plot_options_frame, text="Generate Plot", command=self.generate_plot
        )
        plot_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        # Dataset Actions Frame
        self.action_frame = tk.LabelFrame(main_frame, text="Actions", padx=10, pady=10)
        self.action_frame.grid(row=1, column=0, columnspan=2, pady=20)

        show_button = tk.Button(
            self.action_frame, text="Show Dataset Head", command=self.show_data
        )
        show_button.grid(row=0, column=0, padx=10)

        stats_button = tk.Button(
            self.action_frame,
            text="Show Summary Statistics",
            command=self.show_statistics,
        )
        stats_button.grid(row=0, column=1, padx=10)

        corr_button = tk.Button(
            self.action_frame,
            text="Show Correlation Heatmap",
            command=self.generate_corr_heatmap,
        )
        corr_button.grid(row=0, column=2, padx=10)

        # Status Label
        self.status_label = tk.Label(
            main_frame, text="No dataset loaded.", relief=tk.SUNKEN, anchor="w"
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="we")

        # Configure grid weights
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset Loaded Successfully!")
                self.update_columns()
                self.status_label.config(
                    text=f"Dataset loaded: {file_path.split('/')[-1]}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def show_data(self):
        if self.dataset is not None:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, str(self.dataset.head()))
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def show_statistics(self):
        if self.dataset is not None:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, str(self.dataset.describe()))
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def update_columns(self):
        columns = list(self.dataset.columns)
        self.x_dropdown["values"] = columns
        self.y_dropdown["values"] = columns

    def generate_plot(self):
        if self.dataset is not None:
            plot_type = self.plot_type.get()
            x_col = self.x_column.get()
            y_col = self.y_column.get()

            plot_window = tk.Toplevel(self.root)
            plot_window.title(f"{plot_type.capitalize()} Plot")
            plot_window.geometry("600x500")

            fig, ax = plt.subplots()

            if plot_type == "pairplot":
                sns.pairplot(self.dataset)
            elif plot_type == "scatterplot":
                if x_col and y_col:
                    sns.scatterplot(data=self.dataset, x=x_col, y=y_col, ax=ax)
            elif plot_type == "histogram":
                if x_col:
                    sns.histplot(self.dataset[x_col], ax=ax)
            elif plot_type == "heatmap":
                sns.heatmap(self.dataset.corr(), annot=True, fmt=".2f", ax=ax)

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            messagebox.showerror("Error", "No dataset loaded!")

    def generate_corr_heatmap(self):
        if self.dataset is not None:
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Correlation Heatmap")
            plot_window.geometry("600x500")

            fig, ax = plt.subplots()
            sns.heatmap(self.dataset.corr(), annot=True, fmt=".2f", ax=ax)

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            messagebox.showerror("Error", "No dataset loaded!")


if __name__ == "__main__":
    root = tk.Tk()
    app = EDAApp(root)
    root.mainloop()
