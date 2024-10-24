import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image
import warnings

def analyze_dataset(file_path, content, styles):
    # Load the dataset
    print(f"Loading dataset: {file_path}")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)

    # Add a title for the dataset
    dataset_name = os.path.basename(file_path)
    content.append(PageBreak())  # Start a new page for each dataset
    content.append(Paragraph(f"Analysis of {dataset_name}", styles['Title']))

    # Prepare to collect equations for this dataset
    equations = []

    # Basic statistics
    stats = df.describe()
    
    # Prepare data for table
    data = [['Statistic', 'Value']] + list(stats.items())

    # Create a table and add it to the content
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    content.append(table)
    
    # Mathematical analysis and plotting
    equation, steps = analyze_and_plot(df, content, styles, file_path)  # Now returns steps as well
    if equation:
        equations.append((equation, steps))  # Store the equation and steps for the summary

    # Add bullet points for equations specific to this dataset
    if equations:
        content.append(Paragraph("Inferred Equations:", styles['Normal']))
        for eq, step in equations:
            # Make the equation bold and red, and include the steps
            equation_paragraph = Paragraph(f"• <b><font color='red'>{eq}</font></b><br/>" + step, styles['Normal'])
            content.append(equation_paragraph)

    content.append(Paragraph("<br/>", styles['Normal']))  # Add space between datasets

def analyze_and_plot(df, content, styles, file_path): 
    # Example analysis: Linear regression on the first two numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        content.append(Paragraph("Not enough numerical data for analysis.", styles['Normal']))
        return None, None

    x = df[numeric_cols[0]].values
    y = df[numeric_cols[1]].values
    
    # Check variability before fitting
    if np.std(x) == 0 or np.std(y) == 0:
        content.append(Paragraph("One or both columns have no variability, cannot perform linear regression.", styles['Normal']))
        return None, None

    # Suppress RankWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", np.RankWarning)
        coeffs = np.polyfit(x, y, 1)  # Linear fit
        
        # Check if a warning was triggered
        if len(w) > 0:
            content.append(Paragraph("Warning: Polyfit may be poorly conditioned.", styles['Normal']))
        
        equation = f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f} (for {numeric_cols[0]} vs {numeric_cols[1]})"
    
    # Generate plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, np.polyval(coeffs, x), color='red', label='Fitted line')
    plt.title(f'Linear Regression: {equation}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.legend()
    
    # Save plot as image
    plot_filename = f"plot_{os.path.basename(file_path)}.png"
    plt.savefig(plot_filename)
    plt.close()

    # Add equation and plot to PDF
    content.append(Paragraph(f"Linear Regression Equation: <b><font color='red'>{equation}</font></b>", styles['Normal']))
    content.append(Image(plot_filename, width=5*inch, height=3*inch))

    # Generate steps for forming the equation
    steps = (f"1. Collected data from '{numeric_cols[0]}' and '{numeric_cols[1]}'.<br/>"
             f"2. Applied linear regression: <br/>"
             f"   y = mx + b, where:<br/>"
             f"   m = slope (computed as {coeffs[0]:.2f}),<br/>"
             f"   b = y-intercept (computed as {coeffs[1]:.2f}).")
    
    return equation, steps  # Return the equation and the steps for storage

def analyze_and_generate_report(folder_path):
    content = []
    table_of_contents = []
    
    # Define the output PDF filename
    output_pdf_filename = "consolidated_report.pdf"
    
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist. Please provide a valid path.")
        return

    # Create PDF document with landscape orientation
    pdf = SimpleDocTemplate(output_pdf_filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    
    # Append title
    content.append(Paragraph("Dataset Analysis", styles['Title']))

    # Iterate through the files in the provided directory
    for file_name in os.listdir(folder_path):
        # Process only CSV and Excel files
        if file_name.endswith('.csv') or file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Analyzing dataset: {file_path}")
            analyze_dataset(file_path, content, styles)

            # Add entry to the table of contents
            dataset_name = os.path.basename(file_name)
            page_number = len(content)  # Current content length gives the page number
            table_of_contents.append((dataset_name, page_number))

    # Add table of contents header
    content.insert(1, PageBreak())  # Ensure there's a page break before the TOC
    content.insert(2, Paragraph("Table of Contents", styles['Title']))

    # Add the TOC entries
    for index, (dataset_name, page_number) in enumerate(table_of_contents):
        content.append(Paragraph(f"{index + 1}. {dataset_name} .......... Page {page_number}", styles['Normal']))

    # Build the PDF
    print("Generating PDF report...")
    pdf.build(content)
    print(f"Report generated: {os.path.abspath(output_pdf_filename)}")

# Get user input for the folder path
folder_path = input("Enter the folder path containing datasets: ")

# New approach: First, analyze datasets and then create TOC
datasets = []
for file_name in os.listdir(folder_path):
    # Process only CSV and Excel files
    if file_name.endswith('.csv') or file_name.endswith('.xlsx'):
        datasets.append(file_name)

# Create PDF document with landscape orientation
pdf = SimpleDocTemplate("consolidated_report.pdf", pagesize=landscape(letter))
styles = getSampleStyleSheet()
content = []

# Append title
content.append(Paragraph("Dataset Analysis", styles['Title']))
content.append(PageBreak())  # Page break after title

# Add table of contents header
content.append(Paragraph("Table of Contents", styles['Title']))

# Add entries to the TOC
for index, dataset_name in enumerate(datasets):
    content.append(Paragraph(f"{index + 1}. {dataset_name}", styles['Normal']))

content.append(PageBreak())  # Add a page break after the TOC

# Now analyze datasets and generate the report
for file_name in datasets:
    file_path = os.path.join(folder_path, file_name)
    print(f"Analyzing dataset: {file_path}")
    analyze_dataset(file_path, content, styles)

# Build the PDF
print("Generating PDF report...")
pdf.build(content)
print(f"Report generated: {os.path.abspath('consolidated_report.pdf')}")
