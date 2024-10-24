# pip install ydata-profiling

import os
import pandas as pd
from ydata_profiling import ProfileReport


def profile_datasets(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all files in the given input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            print(f"Generating profile for: {file_path}")

            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)

                # Generate a profile report
                profile = ProfileReport(df, title=f"Profiling Report for {filename}", explorative=True)

                # Save the report as an HTML file in the output directory
                report_file = os.path.join(output_directory, f"profile_report_{filename[:-4]}.html")
                profile.to_file(report_file)
                print(f"Report saved to: {report_file}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    # Specify the input and output directories
    input_dir = r'C:\Users\HimakarRaju\Desktop\PreSkilling\Python\Data_Science_And_Visualization\Datasets\RawDatasets'
    output_dir = r'C:\Users\HimakarRaju\Desktop\PreSkilling\Python\Data_Science_And_Visualization\Session023\outputs'

    # Call the profiling function
    profile_datasets(input_dir, output_dir)
