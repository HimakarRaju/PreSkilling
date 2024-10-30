import pandas as pd
import glob
import os
import sys
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Select, Div
from bokeh.plotting import figure
from bokeh.layouts import column

# Check for command line argument
if len(sys.argv) < 2:
    print("Usage: python pybokeh.py <path_to_csv_directory>")
    # sys.exit(1)
else:
    path_to_csv_directory = input("enter a file path: ")

# Get the directory from command line arguments
if sys.argv<2:
    csv_directory = path_to_csv_directory
else:
    csv_directory = sys.argv[1]

# Use glob to find all CSV files in the specified directory
datasets = {
    os.path.basename(file): file
    for file in glob.glob(os.path.join(csv_directory, "*.csv"))
}

if not datasets:
    print("No CSV files found in the specified directory.")
    sys.exit(1)

# Create a blank plot
plot = figure(
    title="Data Visualization",
    x_axis_label="X-Axis",
    y_axis_label="Y-Axis",
    height=400,
    width=700,
)

# Layout elements
dataset_selector = Select(
    title="Select Dataset:",
    value=list(datasets.keys())[0],
    options=list(datasets.keys()),
)
plot_type_selector = Select(
    title="Select Plot Type:",
    value="Line",
    options=[
        "Line",
        "Scatter",
        "Bar",
        "HBar",
        "Circle",
        "Square",
        "Triangle",
        "Area",
        "Step",
        "VBar",
        "VArea",
        "Patch",
        "Wedge",
    ],
)
x_column_selector = Select(title="Select X Column:", value="", options=[])
y_column_selector = Select(title="Select Y Column:", value="", options=[])
output_text = Div(text="Select a dataset to visualize.")
source = ColumnDataSource(data=dict(x=[], y=[]))


# Function to update column options based on selected dataset
def update_column_options():
    selected_dataset = datasets[dataset_selector.value]
    df = pd.read_csv(selected_dataset)

    # Update X and Y column selectors
    columns = df.columns.tolist()
    x_column_selector.options = columns
    y_column_selector.options = columns
    if columns:
        x_column_selector.value = columns[0]
        y_column_selector.value = columns[1] if len(columns) > 1 else ""


# Function to load and plot the selected dataset
def load_dataset(attr, old, new):
    selected_dataset = datasets[dataset_selector.value]

    try:
        # Read the selected CSV file
        df = pd.read_csv(selected_dataset)
        if df.shape[1] < 2:
            output_text.text = "Please upload a CSV with at least two columns."
            return

        # Update column options
        update_column_options()

        # Plot the data based on the selected columns and plot type
        plot_data(df)

    except Exception as e:
        output_text.text = f"Error loading file: {str(e)}"


# Function to plot the data based on selected columns and plot type
def plot_data(df):
    x_col = x_column_selector.value
    y_col = y_column_selector.value

    if x_col not in df.columns or y_col not in df.columns:
        output_text.text = "Please select valid columns."
        return

    source.data = {"x": df[x_col], "y": df[y_col]}  # Update the data source

    # Remove existing renderers before adding new ones
    plot.renderers = []

    # Draw the selected plot type
    if plot_type_selector.value == "Line":
        plot.line(
            "x",
            "y",
            source=source,
            line_width=2,
            color="blue",
            legend_label="Data Line",
        )
    elif plot_type_selector.value == "Scatter":
        plot.circle(
            "x", "y", source=source, size=5, color="red", legend_label="Data Points"
        )
    elif plot_type_selector.value == "Bar":
        plot.vbar(
            x="x",
            top="y",
            source=source,
            width=0.9,
            color="green",
            legend_label="Data Bars",
        )
    elif plot_type_selector.value == "HBar":
        plot.hbar(
            y="x",
            right="y",
            source=source,
            height=0.4,
            color="orange",
            legend_label="Horizontal Bars",
        )
    elif plot_type_selector.value == "Circle":
        plot.circle(
            "x",
            "y",
            source=source,
            size=10,
            color="blue",
            alpha=0.5,
            legend_label="Circle Markers",
        )
    elif plot_type_selector.value == "Square":
        plot.square(
            "x",
            "y",
            source=source,
            size=10,
            color="green",
            alpha=0.5,
            legend_label="Square Markers",
        )
    elif plot_type_selector.value == "Triangle":
        plot.triangle(
            "x",
            "y",
            source=source,
            size=10,
            color="purple",
            alpha=0.5,
            legend_label="Triangle Markers",
        )
    elif plot_type_selector.value == "Area":
        plot.varea(
            x="x",
            y1="y",
            y2=0,
            source=source,
            fill_color="lightgreen",
            legend_label="Area Plot",
        )
    elif plot_type_selector.value == "Step":
        plot.step(
            "x",
            "y",
            source=source,
            line_width=2,
            color="orange",
            legend_label="Step Plot",
        )
    elif plot_type_selector.value == "VBar":
        plot.vbar(
            x="x",
            top="y",
            source=source,
            width=0.9,
            color="cyan",
            legend_label="Vertical Bars",
        )
    elif plot_type_selector.value == "VArea":
        plot.varea(
            x="x",
            y1="y",
            y2=0,
            source=source,
            fill_color="lightblue",
            legend_label="Vertical Area",
        )
    elif plot_type_selector.value == "Patch":
        # Placeholder for patch plot
        plot.patches(
            "x",
            "y",
            source=source,
            fill_color="pink",
            line_color="black",
            legend_label="Patches",
        )
    elif plot_type_selector.value == "Wedge":
        # Placeholder for wedge plot
        plot.wedge(
            x="x",
            y="y",
            radius=0.5,
            start_angle=0,
            end_angle=3.14,
            source=source,
            fill_color="purple",
            legend_label="Wedge",
        )

    output_text.text = f"Data Loaded: {df.shape[0]} rows and {df.shape[1]} columns."


# Set up event listeners for the dataset selector, plot type selector, and column selectors
dataset_selector.on_change("value", load_dataset)
plot_type_selector.on_change(
    "value",
    lambda attr, old, new: plot_data(pd.read_csv(datasets[dataset_selector.value])),
)
x_column_selector.on_change(
    "value",
    lambda attr, old, new: plot_data(pd.read_csv(datasets[dataset_selector.value])),
)
y_column_selector.on_change(
    "value",
    lambda attr, old, new: plot_data(pd.read_csv(datasets[dataset_selector.value])),
)

# Initial load
load_dataset(None, None, None)

# Arrange layout
layout = column(
    dataset_selector,
    plot_type_selector,
    x_column_selector,
    y_column_selector,
    output_text,
    plot,
)

# Add layout to current document
curdoc().add_root(layout)
