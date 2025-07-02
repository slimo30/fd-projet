import requests
import json
import pandas as pd
import os
import time
from pprint import pprint

# Base URL for the API
BASE_URL = "http://127.0.0.1:5000"

def test_upload_data(file_path):
    """Test the upload-data endpoint."""
    print("\n=== Testing Upload Data ===")
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload-data", files=files)
    
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        pprint(data)
        
        if response.status_code == 200:
            return data.get('path')
        return None
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")
        return None

def test_get_data_head(path):
    """Test the data/head endpoint."""
    print("\n=== Testing Get Data Head ===")
    response = requests.get(f"{BASE_URL}/data/head", params={'path': path})
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        pprint(data)
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")

def test_get_data_columns(path):
    """Test the data/columns endpoint."""
    print("\n=== Testing Get Data Columns ===")
    response = requests.get(f"{BASE_URL}/data/columns", params={'path': path})
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        pprint(data)
        if response.status_code == 200:
            return data.get('columns', [])
        return []
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")
        return []

def test_select_columns(path, columns):
    """Test the data/select-columns endpoint."""
    if not columns or len(columns) < 2:
        print("\n=== Skipping Select Columns (not enough columns) ===")
        return
        
    print("\n=== Testing Select Columns ===")
    data = {'path': path, 'columns': columns[:5]}  # Select first 2 columns for testing
    response = requests.post(f"{BASE_URL}/data/select-columns", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")

def test_get_statistics(path):
    """Test the data/statistics endpoint."""
    print("\n=== Testing Get Statistics ===")
    response = requests.get(f"{BASE_URL}/data/statistics", params={'path': path})
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")

def test_fill_missing(path, columns):
    """Test the data/fill-missing endpoint."""
    if not columns:
        print("\n=== Skipping Fill Missing Values (no columns) ===")
        return
        
    print("\n=== Testing Fill Missing Values ===")
    
    # Create a fill strategy for each column based on its name/position
    fill_strategies = {}
    for i, col in enumerate(columns[:2]):  # Use only first 2 columns for simplicity
        strategy = 'mean' if i % 3 == 0 else 'mode'  # Skipping median for categorical columns
        fill_strategies[col] = strategy
    
    data = {'path': path, 'fill': fill_strategies}
    response = requests.post(f"{BASE_URL}/data/fill-missing", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")

def test_normalize_data(path, columns):
    """Test the data/normalize endpoint."""
    if not columns:
        print("\n=== Skipping Normalize Data (no columns) ===")
        return
        
    print("\n=== Testing Normalize Data ===")
    
    # Find numeric columns for normalization
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(pd.Series(col))]
    
    if not numeric_cols:
        print("No suitable numeric columns found for normalization test")
        return
    
    # Test Z-score normalization
    data = {'path': path, 'method': 'zscore', 'columns': numeric_cols[:1]}
    print("\nTesting Z-Score Normalization")
    response = requests.post(f"{BASE_URL}/data/normalize", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")
    
    # Test Min-Max normalization
    data = {'path': path, 'method': 'minmax', 'columns': numeric_cols[:1]}
    print("\nTesting Min-Max Normalization")
    response = requests.post(f"{BASE_URL}/data/normalize", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")

def test_scatter_plot(path, columns):
    """Test the plot/scatter endpoint."""
    if not columns or len(columns) < 2:
        print("\n=== Skipping Scatter Plot (not enough columns) ===")
        return
        
    print("\n=== Testing Scatter Plot ===")
    
    # Make sure the columns are converted to string type for plotting
    x_col = columns[0] if len(columns) > 0 else None
    y_col = columns[1] if len(columns) > 1 else None
    
    if x_col and y_col:
        data = {'path': path, 'x': x_col, 'y': y_col}
        response = requests.post(f"{BASE_URL}/plot/scatter", json=data)
        print(f"Status Code: {response.status_code}")
        try:
            pprint(response.json())
        except json.JSONDecodeError:
            print(f"Response content (not JSON): {response.text}")
    else:
        print("Not enough columns for scatter plot testing")

def test_box_plot(path, columns):
    """Test the plot/box endpoint."""
    if not columns:
        print("\n=== Skipping Box Plot (no columns) ===")
        return
        
    print("\n=== Testing Box Plot ===")
    
    # Use first 2 columns for box plot
    cols_to_use = columns[:2] if len(columns) >= 2 else columns
    
    data = {'path': path, 'columns': cols_to_use}
    response = requests.post(f"{BASE_URL}/plot/box", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")

def test_categorical_to_numerical(path, columns):
    """Test the data/categorical-to-numerical endpoint."""
    if not columns:
        print("\n=== Skipping Categorical to Numerical Conversion (no columns) ===")
        return
        
    print("\n=== Testing Categorical to Numerical Conversion ===")
    
    # Find columns that might be categorical
    categorical_cols = [col for col in columns if col.lower() in ['status', 'category', 'type']]
    
    if not categorical_cols and len(columns) >= 1:
        categorical_cols = columns[:1]
    
    if not categorical_cols:
        print("No suitable categorical columns found for testing")
        return
    
    # Test label encoding
    data = {'path': path, 'method': 'label', 'columns': categorical_cols}
    print("\nTesting Label Encoding")
    response = requests.post(f"{BASE_URL}/data/categorical-to-numerical", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")
    
    # Test one-hot encoding (using the original path as it might have been modified)
    data = {'path': path, 'method': 'onehot', 'columns': categorical_cols}
    print("\nTesting One-Hot Encoding")
    response = requests.post(f"{BASE_URL}/data/categorical-to-numerical", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        pprint(response.json())
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")

def create_sample_data():
    """Create a sample CSV for testing."""  
    print("\n=== Creating Sample Data for Testing ===")
    data = {
        'id': range(1, 101),
        'category': ['A', 'B', 'C', 'D', 'E'] * 20,
        'value': [float(i) for i in range(1, 101)],
        'status': ['active', 'inactive', 'pending', 'unknown'] * 25,
        'rating': [i % 5 + 1 for i in range(100)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add missing values randomly
    for col in df.columns:
        df.loc[df.sample(frac=0.1).index, col] = None
    
    sample_file = 'sample_data.csv'
    df.to_csv(sample_file, index=False)
    print(f"Sample data created at {sample_file}")
    return sample_file

def handle_test_function(func, *args, **kwargs):
    """Execute a test function and handle any exceptions."""  
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {str(e)}")
        return None
def test_save_data(new_path):
    """Test the data/save endpoint."""
    print("\n=== Testing Save Data ===")
    data = {'new_path': new_path}
    response = requests.post(f"{BASE_URL}/data/save", json=data)
    
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        pprint(data)
        if response.status_code == 200:
            return data.get('path')
        return None
    except json.JSONDecodeError:
        print(f"Response content (not JSON): {response.text}")
        return None

def main():
    """Main function to run all tests."""  
    try:
        # Test with sample data
        sample_file = create_sample_data()
        
        # Upload the data and get the path
        data_path = handle_test_function(test_upload_data, sample_file)
        
        if data_path:
            # Test various endpoints
            handle_test_function(test_get_data_head, data_path)
            columns = handle_test_function(test_get_data_columns, data_path) or []
            handle_test_function(test_select_columns, data_path, columns)
            handle_test_function(test_get_statistics, data_path)
            handle_test_function(test_fill_missing, data_path, columns)
            handle_test_function(test_normalize_data, data_path, columns)
            handle_test_function(test_scatter_plot, data_path, columns)
            handle_test_function(test_box_plot, data_path, columns)
            handle_test_function(test_categorical_to_numerical, data_path, columns)
            test_save_data(data_path)
            print("\n=== All tests completed ===")
        else:
            print("Failed to upload data. Tests cannot continue.")
    except Exception as e:
        print(f"Error in main test execution: {str(e)}")

if __name__ == "__main__":
    # Make sure the Flask app is running before executing tests
    print("Ensure the Flask API is running at http://127.0.0.1:5000")
    print("Press Enter to continue with testing...")
    input()
    
    main()
