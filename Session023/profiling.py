import os
import pandas as pd
from ydata_profiling import ProfileReport
import threading

def generate_profile_report(file_path, output_directory):
    try:
        # Determine the file type
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return

        # Generate profile report
        profile = ProfileReport(data, title=os.path.basename(file_path), explorative=True)

        # Save the report as an HTML file in the output directory
        output_file = os.path.join(output_directory, os.path.basename(file_path).replace('.csv', '').replace('.xlsx',
                                                                                                             '') + '_profile_report.html')
        profile.to_file(output_file)
        print(f"Generated report: {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main(input_directory, output_directory):
    threads = []

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List all files in the input directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)

        # Check if the file is a CSV or XLSX
        if os.path.isfile(file_path) and (filename.endswith('.csv') or filename.endswith('.xlsx')):
            thread = threading.Thread(target=generate_profile_report, args=(file_path, output_directory))
            threads.append(thread)
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    # Specify the input directory containing the CSV/XLSX files
    input_directory = r'C:\Users\HimakarRaju\Desktop\PreSkilling\Python\Data_Science_And_Visualization\Datasets\RawDatasets'  # Update this path
    output_directory = './outputs'  # Update this path
    main(input_directory, output_directory)
