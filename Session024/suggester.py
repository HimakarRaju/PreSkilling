import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
import pulp
from fpdf import FPDF
import os
import io


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

    # Set up the PDF document
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

    # Insert a blank page for Page 3 (correlation plot and optimization suggestions)
    pdf.add_page()

    # Correlation Plot
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, "Correlation Plot:", ln=True)
    plot_correlation_matrix(df, dataset_name)  # Generate and save the plot
    pdf.image(f"{dataset_name}_corr.png", x=10, y=pdf.get_y(), w=190)

    # Suggested optimization and equations
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
    """Analyze the dataset, suggest and apply relevant optimization techniques."""

    report = "\n\n\n\nSuggested Optimization Techniques:\n"

    report += "\nOptimization Techniques:\n"

    report += "1. Linear Programming:\n"
    report += "   - Linear programming is used to find the best outcome in a mathematical model with linear relationships. The objective function and constraints are defined, and the solution maximizes or minimizes the objective function while satisfying the constraints.\n"
    report += "   - Results indicate the optimal values of the decision variables.\n"
    
    report += "2. Integer Programming:\n"
    report += "   - Integer programming is similar to linear programming, but it restricts some or all of the decision variables to be integers. This is useful for discrete choices.\n"

    report += "3. Optimization with PuLP:\n"
    report += "   - PuLP is a Python library for formulating and solving linear and integer programming problems. It allows you to define objectives, constraints, and decision variables easily, and it can solve these problems using various algorithms.\n"

    # Get a list of numeric columns
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Use correlation to find the most correlated variables
    corr_matrix = df.corr()
    if len(numeric_columns) >= 2:
        # Find the two most correlated columns
        corr_pairs = corr_matrix.unstack().sort_values(
            kind="quicksort", ascending=False
        )
        corr_pairs = corr_pairs[corr_pairs < 1.0]  # Remove self-correlation
        top_pair = corr_pairs.idxmax()
        x, y = top_pair

        # Define a list of common target variable names
        common_targets = ["target", "label", "y", "price", "cost", "estimate"]

        # Select the most correlated variable with y as the target variable
        target_variable = corr_matrix[y].drop(y).idxmax()

        # Check if the target_variable is in common_targets
        if target_variable not in common_targets:
            print(
                f"Suggested target variable: {target_variable}. Is this acceptable? (y/n)"
            )
            user_input = input().strip().lower()
            if user_input != "y":
                target_variable = input("Please specify your desired target variable: ")

        report += f"\nChosen y variable for optimization: {y} (correlation: {corr_matrix.loc[x, y]:.2f})\n"
        report += f"Selected target variable for optimization: {target_variable} (correlation with {y}: {corr_matrix.loc[target_variable, y]:.2f})\n"

        # Example: Linear programming using these variables
        report += run_linear_programming_with_pulp(df, x, y, target_variable)
        # Example: Maximize using minimize() from scipy
        report += run_maximize_with_minimize(df, x, y, target_variable)
        # Example: Integer programming with pulp
        report += run_integer_programming_with_pulp(df, target_variable)

    return report


def run_integer_programming_with_pulp(df, target_variable):
    """Example of applying integer programming using pulp."""
    from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus

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


def run_maximize_with_minimize(df, x, y, target_variable):
    """Maximization using scipy.optimize.minimize (maximize by minimizing negative objective)."""
    report = "\nMaximization using Scipy's Minimize:\n"

    # Objective function to maximize (minimize the negative)
    def objective(vars):
        return -(vars[0] + vars[1])  # Example: simple sum of x and y

    initial_guess = [df[x].mean(), df[y].mean()]

    # Run minimization
    result = minimize(objective, initial_guess, method="BFGS")

    if result.success:
        report += f"Maximization result: {result.x}\n"
    else:
        report += f"Maximization failed: {result.message}\n"

    return report


def run_linear_programming_with_pulp(df, x, y, target_variable):
    """Example of applying linear programming based on correlation."""
    report = "\nLinear Programming based on Correlation:\n"

    # Objective function: minimize the negative of the target variable (maximize target_variable)
    c = -df[
        target_variable
    ].values  # Objective function (minimize -target to maximize the target)

    # Create constraint matrix A_ub based on the correlated variables x and y
    A = df[[x, y]].values  # Constraint coefficients

    # Ensure that A has the correct number of columns to match the size of c
    if A.ndim != 2 or A.shape[1] != len(c):
        report += "Error: Dimensions of A_ub and c do not match.\n"
        return report

    # Generate dummy upper bounds (you can modify these to fit your problem)
    b = [1.5] * A.shape[0]  # Make sure the size of b matches the number of rows in A_ub

    try:
        # Run linear programming using scipy's linprog
        res = linprog(c, A_ub=A, b_ub=b, method="highs")

        if res.success:
            report += f"Linear programming result: {res.x}\n"
        else:
            report += f"Linear programming failed: {res.message}\n"
    except ValueError as e:
        report += f"Error during linear programming: {str(e)}\n"

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
