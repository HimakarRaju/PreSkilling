import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from googlesearch import search

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def fetch_numerical_data(keyword):
    # Search for the query and get 15 results
    urls = search(keyword, num_results=15)
    numerical_data = []

    for url in urls:
        print(f"Fetching data from: {url}")
        content = fetch_data(url)
        if content:
            # Extract numerical data from content
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
            numerical_data.extend(numbers)

    return list(set(numerical_data))  # Return unique numerical values

def extract_data(numerical_data):
    extracted_info = {
        "years": [],
        "sales": [],
        "market": [],
        "production": []
    }

    for value in numerical_data:
        str_value = str(value)  # Convert to string to handle all cases

        # Check for year (four-digit number)
        if str_value.isdigit() and len(str_value) == 4 and 2000 <= int(str_value) <= 2024:
            extracted_info["years"].append(str_value)

        # Check for sales (numbers typically greater than a certain threshold)
        elif str_value.replace('.', '', 1).isdigit() and float(str_value) >= 1000:  # Example threshold for sales
            extracted_info["sales"].append(str_value)

        # Check for market share (assume percentage representation)
        elif str_value.isdigit() and 0 <= int(str_value) <= 100:  # Example for market share percentage
            extracted_info["market"].append(str_value)

        # Check for production (assume larger numbers are production counts)
        elif str_value.isdigit() and int(str_value) > 1000:  # Example threshold for production
            extracted_info["production"].append(str_value)

    return extracted_info

def main():
    # Define a keyword for fetching numerical data
    keyword = input("Enter a product or query you want to search for: ")
    
    # Step 1: Fetch numerical data based on the keyword
    numerical_data = fetch_numerical_data(keyword)

    # Step 2: Extract Data
    extracted_info = extract_data(numerical_data)

    # Step 3: Convert extracted_info to DataFrame for further processing
    df_years = pd.DataFrame(extracted_info['years'], columns=['Years'])
    df_sales = pd.DataFrame(extracted_info['sales'], columns=['Sales'])
    df_market = pd.DataFrame(extracted_info['market'], columns=['Market'])
    df_production = pd.DataFrame(extracted_info['production'], columns=['Production'])

    # Combine all into a single DataFrame if needed
    combined_df = pd.concat([df_years, df_sales, df_market, df_production], axis=1)

    # Print the combined DataFrame
    print("Extracted Data:")
    print(combined_df)

if __name__ == "__main__":
    main()
