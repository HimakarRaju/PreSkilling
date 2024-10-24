import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
from fpdf import FPDF
import os


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

    report += "\nOptimization Techniques:\n"
    report += "1. Linear Programming:\n"
    report += "   - Linear programming is used to find the best outcome in a mathematical model with linear relationships.\n"

    report += "2. Integer Programming:\n"
    report += "   - Integer programming restricts some or all of the decision variables to be integers, useful for discrete choices.\n"

    # Determine model suitability based on column types
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    if numeric_columns:
        report += "\nSuggested Models based on Numeric Columns:\n"
        if len(numeric_columns) > 1:
            report += "   - Regression Analysis: Use when predicting a continuous outcome based on one or more predictors.\n"
        if len(categorical_columns) > 0:
            report += "   - Classification Models: Consider logistic regression or decision trees for binary/multiclass outcomes.\n"

    if categorical_columns:
        report += "\nSuggested Models based on Categorical Columns:\n"
        report += (
            "   - If predicting categorical outcomes, use classification techniques.\n"
        )

    report += "\nBased on the dataset's characteristics, suitable models could be:\n"
    report += " - Linear Regression for continuous numeric outcomes.\n"
    report += " - Logistic Regression for binary outcomes.\n"
    report += (
        " - Decision Trees or Random Forests for both regression and classification.\n"
    )

    return report


def run_integer_programming_with_pulp(df, target_variable):
    """Example of applying integer programming using pulp."""
    report = "\nInteger Programming based on Target Variable:\n"

    # Create a linear programming problem
    prob = LpProblem("Maximize_Target_Variable", LpMaximize)

    # Define decision variables
    decision_vars = {
        i: LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(len(df))
    }

    # Objective function: maximize the target variable
    prob += (
        lpSum(decision_vars[i] * df[target_variable].iloc[i] for i in range(len(df))),
        "Objective",
    )

    # Add constraints
    prob += lpSum(decision_vars[i] for i in range(len(df))) <= 10, "Sum_Constraint"

    # Solve the problem
    prob.solve()

    # Collect results
    report += f"Status: {LpStatus[prob.status]}\n"

    decision_vars_report = []
    for var in decision_vars.values():
        if var.varValue > 0.1:
            decision_vars_report.append(f"{var.name}: {var.varValue}\n")
        elif (
            var.varValue == 0.00
        ):  # If it's 0.00, we will add only the last decision variable
            last_var_report = f"{var.name}: {var.varValue}\n"

    if decision_vars_report:
        report += "Decision Variables (exceeding 0.1):\n" + "".join(
            decision_vars_report
        )
    if "last_var_report" in locals():  # Check if last_var_report is defined
        report += f"Last decision variable (0.00): {last_var_report}"

    return report


def run_linear_programming(df, x, y, target_variable):
    """Example of applying linear programming based on correlation."""
    report = "\nLinear Programming based on Correlation:\n"

    # Coefficients based on correlation values
    c = -df[
        target_variable
    ].values  # Objective function (maximize by minimizing -target)
    A = df[[x, y]].values  # Constraint coefficients
    b = [1.5, 2]  # Example constraint coefficients

    # Run linear programming using scipy
    res = linprog(c, A_ub=A, b_ub=b, method="highs")

    if res.success:
        report += f"Linear programming result: {res.x}\n"
    else:
        report += f"Linear programming failed: {res.message}\n"

    return report


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
