import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import threading
from queue import Queue


# Function to measure time taken by a function
def time_function(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    print(f"Time taken for {func.__name__}: {end_time - start_time:.2f} seconds")
    return result


# Automatically select target column
def auto_select_target(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.empty:
        raise ValueError("No numeric columns found in the DataFrame.")
    potential_target_columns = numeric_columns[numeric_columns != "Record ID"]
    target_column_name = potential_target_columns[0] if not potential_target_columns.empty else numeric_columns[0]
    print(f"Automatically selected numeric target column for regression: {target_column_name}")
    return target_column_name


# Preprocess data
def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    X = X.select_dtypes(include=[np.number])
    return X, y


# Plot 2D scatter plots
def plot_2d(df, target_column, plot_queue):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if col != target_column:
            plot_queue.put((df, col, target_column))  # Pass df along with plotting params


# Plot 3D scatter plots
def plot_3d(df, target_column, plot_queue):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) >= 3:
        for i in range(len(numeric_columns) - 2):
            for j in range(i + 1, len(numeric_columns) - 1):
                x_col = numeric_columns[i]
                y_col = numeric_columns[j]
                z_col = target_column
                plot_queue.put((df, x_col, y_col, z_col))  # Pass df along with plotting params


# Create plots from queue
def create_plots(plot_queue):
    while not plot_queue.empty():
        plot_data = plot_queue.get()
        if len(plot_data) == 3:  # 2D plot
            df, col, target_column = plot_data
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=col, y=target_column)
            plt.title(f"2D Scatter Plot: {col} vs {target_column}")
            plt.xlabel(col)
            plt.ylabel(target_column)
            plt.grid()
            plt.show()
        elif len(plot_data) == 4:  # 3D plot
            df, x_col, y_col, z_col = plot_data
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[x_col], df[y_col], df[z_col], alpha=0.7)
            ax.set_title(f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            plt.show()


# Analyze data and train model
def analyze_data(file_path, plot_queue):
    df = pd.read_csv(file_path)  # Load the dataset
    print(f"\nAnalyzing {file_path} with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Time the data analysis process
    time_function(analyze_and_plot, df, plot_queue)


def analyze_and_plot(df, plot_queue):
    print("Dataset summary:")
    print(df.describe())
    print("\nData Types:")
    print(df.dtypes)

    target_column = auto_select_target(df)
    X, y = preprocess_data(df, target_column)
    X = X.dropna(axis=1, how='all')
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print("Numerical columns:", numerical_cols)
    print("Categorical columns:", categorical_cols)

    transformers = [('num', SimpleImputer(strategy='mean'), numerical_cols)]
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    feature_names = numerical_cols + (
        pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
            categorical_cols).tolist() if categorical_cols else [])
    feature_importance = pd.DataFrame(pipeline.named_steps['model'].coef_, index=feature_names, columns=["Coefficient"])
    print("Feature importance:\n", feature_importance)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")

    # Collect plot data in the queue
    plot_2d(df, target_column, plot_queue)
    plot_3d(df, target_column, plot_queue)


# Process multiple files
def process_files_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv') or f.endswith('.xlsx')]
    threads = []
    plot_queue = Queue()

    for file in files:
        file_path = os.path.join(directory, file)
        thread = threading.Thread(target=time_function, args=(analyze_data, file_path, plot_queue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Create plots after all threads have finished
    create_plots(plot_queue)


# Main function to run analysis
def main(directory):
    start_time = time.time()
    print(f"Analyzing files in directory: {directory}")
    process_files_in_directory(directory)
    end_time = time.time()
    print(f"\nTotal time taken for analysis: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    directory = r"C:\Users\HimakarRaju\Desktop\PreSkilling\Python\Data_Science_And_Visualization\Datasets\RawDatasets"  # Update with your directory path
    main(directory)
