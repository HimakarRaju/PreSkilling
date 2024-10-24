import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import concurrent.futures
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Global variable to hold thread statuses
thread_statuses = []

# Function to update the GUI with thread statuses
def update_status_gui(tree):
    for status in thread_statuses:
        thread_name = status['name']
        thread_action = status['action']
        thread_state = status['state']
        # Insert or update in the treeview
        existing_item = tree.exists(thread_name)
        if existing_item:
            tree.item(thread_name, values=(thread_name, thread_state, thread_action))
        else:
            tree.insert("", "end", thread_name, values=(thread_name, thread_state, thread_action))
    tree.after(1000, update_status_gui, tree)  # Refresh every second

# Automatically select the target column for regression
def auto_select_target(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.empty:
        raise ValueError("No numeric columns found in the DataFrame.")
    
    potential_target_columns = numeric_columns[numeric_columns != "Record ID"]
    target_column_name = potential_target_columns[0] if not potential_target_columns.empty else numeric_columns[0]

    print(f"Automatically selected numeric target column for regression: {target_column_name}")
    return target_column_name

# Function to preprocess data
def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert columns to numeric, forcing errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    return X, y

# Function to create a 3D scatter plot in a tab
def plot_3d_scatter(df, target_column, dataset_name, tab):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_columns) < 3:
        messagebox.showwarning("Warning", "Not enough numeric columns for 3D plotting.")
        return
    
    # Select the first three numeric columns for 3D plotting
    x_col, y_col, z_col = numeric_columns[:3]
    
    ax.scatter(df[x_col], df[y_col], df[z_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f"{dataset_name}: 3D Scatter Plot")
    
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Function to create a single figure for all plots in a tab
def plot_all_in_tab(df, target_column, dataset_name, tab):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Set up subplots for 2D plots
    num_columns = len(numeric_columns) - 1  # Exclude target column
    num_rows = (num_columns + 1) // 2  # Two plots per row
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    # Ensure we don't exceed the number of axes available
    for idx, col in enumerate(numeric_columns):
        if col != target_column and idx < len(axes):
            sns.scatterplot(data=df, x=col, y=target_column, ax=axes[idx])
            axes[idx].set_title(f"{dataset_name}: {col} vs {target_column}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel(target_column)
            axes[idx].grid()
    
    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    # Create a canvas to embed the figure into the tkinter tab
    canvas = FigureCanvasTkAgg(fig, master=tab)  # Create canvas for matplotlib figure
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Function to analyze data and train model
def analyze_data(file_path):
    # Update thread status
    thread_statuses.append({'name': file_path, 'action': 'Loading data', 'state': 'In Progress'})
    
    try:
        df = pd.read_csv(file_path)  # Load your dataset here
        thread_statuses.append({'name': file_path, 'action': 'Data loaded', 'state': 'Completed'})

        print(f"Dataset loaded from {file_path} with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Summarize dataset
        print("Dataset summary:")
        print(df.describe())
        print("\nData Types:")
        print(df.dtypes)

        # Automatically select the target column
        target_column = auto_select_target(df)
        X, y = preprocess_data(df, target_column)

        # Drop columns with all NaN values
        X = X.dropna(axis=1, how='all')

        # Separate numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        print("Numerical columns:", numerical_cols)
        print("Categorical columns:", categorical_cols)

        # Create transformers based on the availability of categorical columns
        transformers = [('num', SimpleImputer(strategy='mean'), numerical_cols)]

        # Only add the categorical transformer if there are categorical columns
        if categorical_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

        # Create a ColumnTransformer for preprocessing
        preprocessor = ColumnTransformer(transformers=transformers)

        # Create a pipeline with preprocessing and the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', LinearRegression())])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = pipeline.predict(X_test)

        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print model performance metrics
        print("\nModel Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R² Score: {r2}")

        return (df, target_column, os.path.basename(file_path))  # Return the necessary data for plotting
    except Exception as e:
        thread_statuses.append({'name': file_path, 'action': f'Error: {str(e)}', 'state': 'Error'})
        print(f"{file_path} generated an exception: {e}")

# Function to process a folder of files
def process_folder(folder_path):
    # Get all CSV files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"Found {len(files)} files to analyze.")

    # Determine thread pool size
    thread_pool_size = min(len(files), 10)  # Limit to 10 threads
    print(f"Using a thread pool of size: {thread_pool_size}")

    # Create a new tkinter window
    root = tk.Tk()
    root.title("Thread Status")

    # Create a treeview to display thread statuses
    tree = ttk.Treeview(root, columns=("Thread Name", "Status", "Action"), show="headings")
    tree.heading("Thread Name", text="Thread Name")
    tree.heading("Status", text="Status")
    tree.heading("Action", text="Action")
    tree.pack(expand=True, fill="both")

    # Start the GUI thread status updates
    update_status_gui(tree)

    # Create a notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH)

    # Start thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
        future_to_file = {executor.submit(analyze_data, os.path.join(folder_path, f)): f for f in files}
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                df, target_column, dataset_name = future.result()
                # Create a new tab for each dataset processed
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=dataset_name)

                # Plot all 2D graphs in the tab
                plot_all_in_tab(df, target_column, dataset_name, tab)

                # Create a 3D scatter plot in the same tab
                plot_3d_scatter(df, target_column, dataset_name, tab)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    root.mainloop()

# Entry point to the application
if __name__ == "__main__":
    folder_path = simpledialog.askstring("Input", "Enter the path to the folder containing CSV files:")
    if folder_path:
        process_folder(folder_path)
