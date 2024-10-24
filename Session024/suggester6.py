import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from fpdf import FPDF
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report


def load_dataset(file_path):
    """Load the dataset from a CSV or Excel file."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(
            "File format not supported. Please provide a .csv or .xlsx file."
        )


def detect_sequential_columns(df):
    """Detect columns that contain sequential numbers (e.g., ID or serial number)."""
    sequential_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if the column is sequential by comparing the differences
            if (df[col].diff().dropna().unique() == 1).all():
                sequential_columns.append(col)
    return sequential_columns


def generate_report(df, file_path):
    """Generate a comprehensive PDF report for the dataset."""
    pdf = FPDF()
    pdf.add_page()

    # Page 1: Dataset Information
    pdf.set_font("Arial", "B", 16)
    dataset_name = os.path.basename(file_path).split(".")[0]
    pdf.cell(200, 10, f"{dataset_name}", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Dataset Name: {dataset_name}", ln=True)
    pdf.cell(200, 10, f"Shape: {df.shape[0]} rows, {df.shape[1]} columns", ln=True)

    # Missing values table
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Missing Values in Dataset:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    missing_values = df.isnull().sum()

    # Table Header
    pdf.cell(90, 10, "Column Name", border=1)
    pdf.cell(90, 10, "Missing Values", border=1, ln=True)

    # Table Rows
    for col, missing in missing_values.items():
        pdf.cell(90, 10, col, border=1)
        pdf.cell(90, 10, str(missing), border=1, ln=True)

    # Insert a blank page for Page 2
    pdf.add_page()

    # Descriptive Statistics Table
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Dataset Description:", ln=True)
    pdf.ln(5)

    describe_data = df.describe().transpose()  # Transpose for better formatting

    # Splitting the descriptive statistics table across pages
    num_columns_per_page = 4
    num_stats = describe_data.shape[0]
    num_pages = (
        num_stats + num_columns_per_page - 1
    ) // num_columns_per_page  # Ceiling division

    for page in range(num_pages):
        # Table Header
        pdf.cell(40, 10, "Statistic", border=1)
        for col in range(num_columns_per_page):
            index = page * num_columns_per_page + col
            if index < num_stats:
                pdf.cell(30, 10, describe_data.index[index], border=1)
        pdf.ln()  # Move to next line after header

        # Table Rows for each statistic
        for stat in describe_data.columns:
            pdf.cell(40, 10, stat, border=1)  # Print statistic name
            for col in range(num_columns_per_page):
                index = page * num_columns_per_page + col
                if index < num_stats:
                    value = describe_data[stat].iloc[index]
                    pdf.cell(
                        30,
                        10,
                        (
                            f"{value:.2f}"
                            if isinstance(value, (int, float))
                            else str(value)
                        ),
                        border=1,
                    )
            pdf.ln()  # Move to the next line after all values for a statistic

        # Add a new page if there are more statistics to display
        if (page + 1) < num_pages:
            pdf.add_page()

    # Insert a blank page for Page 3 (correlation plot and model suggestions)
    pdf.add_page()

    # Correlation Plot
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, "Correlation Plot:", ln=True)
    plot_correlation_matrix(df, dataset_name)  # Generate and save the plot
    pdf.image(f"{dataset_name}_corr.png", x=10, y=pdf.get_y(), w=190)

    # Suggested optimization and models
    pdf.ln(100)
    optimization_report = suggest_optimization(df)
    pdf.multi_cell(0, 10, optimization_report)

    # Save the report as a PDF file
    pdf_output = f"{dataset_name}_report.pdf"
    pdf.output(pdf_output)
    print(f"Report saved as {pdf_output}")


def plot_correlation_matrix(df, dataset_name):
    """Generate and save a correlation matrix plot as PNG."""
    plt.figure(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(f"Correlation Matrix of {dataset_name}")
    plt.savefig(f"{dataset_name}_corr.png")
    plt.close()


def suggest_optimization(df):
    """Analyze the dataset, suggest relevant optimization techniques and suitable models."""
    report = "\n\n\n\nSuggested Optimization Techniques and Models:\n"

    # Determine model suitability based on column types
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    if numeric_columns:
        report += "\nSuggested Optimization Techniques:\n"
        # Linear Programming
        report += "1. Linear Programming:\n"
        report += "   - Suitable for maximizing or minimizing a linear objective function subject to linear equality and inequality constraints.\n"
        # Run linear programming
        target_variable = numeric_columns[
            -1
        ]  # Use the last numeric column as the target
        constraint_columns = numeric_columns[
            :-1
        ]  # Use all other numeric columns as constraints
        optimization_result = run_linear_programming(
            df, constraint_columns, target_variable
        )
        report += "\nLinear Programming Result:\n" + optimization_result

        report += "\nSuggested Models based on Numeric Columns:\n"
        if len(numeric_columns) > 1:
            report += "   - Regression Analysis: Use when predicting a continuous outcome based on one or more predictors.\n"
            regression_result = run_regression(df, target_variable)
            report += "\nRegression Analysis Result:\n" + regression_result

        if len(categorical_columns) > 0:
            report += "   - Classification Models: Consider logistic regression for binary/multiclass outcomes.\n"
            classification_report_str = run_classification(df, categorical_columns[0])
            report += "\nClassification Report:\n" + classification_report_str

    return report


def run_regression(df, target_variable):
    """Run linear regression."""
    # Assume that the first numeric column will be used as a feature
    feature_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if target_variable not in feature_columns:
        return "Target variable not found in feature columns."

    feature_columns.remove(target_variable)

    X = df[feature_columns]
    y = df[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate and return regression report
    return f"Regression Coefficients: {model.coef_}, Intercept: {model.intercept_}"


def run_classification(df, target_variable):
    """Run a simple logistic regression classification."""
    report = ""

    # Assume that the first numeric column will be used as a feature
    feature_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if not feature_columns:
        return "No numeric feature columns available for classification."

    X = df[feature_columns].drop(columns=[target_variable])
    y = df[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create a logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate and return classification report
    report = classification_report(y_test, y_pred)
    return report


def run_linear_programming(df, constraint_columns, target_variable):
    """Run linear programming on the dataset."""
    # Define the objective function
    c = df[target_variable].values  # Coefficients for the objective function

    # Select the appropriate constraint data
    A = df[constraint_columns].values  # Coefficients for the constraints
    b = df[constraint_columns].sum(axis=0).values  # Right-hand side of the constraints

    # Check if dimensions are valid
    if A.ndim != 2 or A.shape[1] != len(c):
        return "Invalid dimensions for linear programming. Check the constraints and objective function."

    # Run linear programming using scipy.optimize.linprog
    res = linprog(c, A_ub=A, b_ub=b, method="highs")

    if res.success:
        return f"Optimal value: {res.fun}, Optimal solution: {res.x}"
    else:
        return "Linear programming did not find a solution."


def main():
    file_path = input("Enter the path to your CSV or Excel file: ")
    df = load_dataset(file_path)

    # Detect and drop sequential columns
    sequential_columns = detect_sequential_columns(df)
    if sequential_columns:
        df = df.drop(columns=sequential_columns)
        print(f"Dropped sequential columns: {sequential_columns}")

    # Generate a comprehensive report
    generate_report(df, file_path)


if __name__ == "__main__":
    main()
