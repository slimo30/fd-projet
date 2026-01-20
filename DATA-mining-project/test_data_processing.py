import requests
import json
import pandas as pd
import os
import time
from pprint import pprint

# Base URL for the API
BASE_URL = "http://localhost:5001"

def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running at {BASE_URL}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {BASE_URL}. Is the server running?")
        print("Run 'python run.py' in a separate terminal.")
        return False

def test_upload_data(file_path):
    """Test the upload-data endpoint."""
    print("\n" + "="*80)
    print("📤 Testing Upload Data")
    print("="*80)
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload-data", files=files)
    
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        if response.status_code == 200:
            print(f"✅ Upload successful")
            print(f"📊 Columns: {', '.join(data.get('columns', []))}")
            return data.get('path')
        else:
            print(f"❌ Upload failed: {data.get('error', 'Unknown error')}")
            return None
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")
        return None

def test_get_data_head(path):
    """Test the data/head endpoint."""
    print("\n" + "="*80)
    print("👀 Testing Get Data Head")
    print("="*80)
    response = requests.get(f"{BASE_URL}/data/head", params={'path': path})
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        if response.status_code == 200:
            print(f"✅ Retrieved {len(data)} rows")
        else:
            print(f"❌ Failed: {data.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")

def test_get_data_columns(path):
    """Test the data/columns endpoint."""
    print("\n" + "="*80)
    print("📋 Testing Get Data Columns")
    print("="*80)
    response = requests.get(f"{BASE_URL}/data/columns", params={'path': path})
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        if response.status_code == 200:
            columns = data.get('columns', [])
            print(f"✅ Found {len(columns)} columns: {', '.join(columns)}")
            return columns
        else:
            print(f"❌ Failed: {data.get('error', 'Unknown error')}")
            return []
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")
        return []

def test_select_columns(path, columns):
    """Test the data/select-columns endpoint."""
    if not columns or len(columns) < 2:
        print("\n⏭️  Skipping Select Columns (not enough columns)")
        return
        
    print("\n" + "="*80)
    print("✂️  Testing Select Columns")
    print("="*80)
    selected = columns[:3] if len(columns) >= 3 else columns
    data = {'path': path, 'columns': selected}
    response = requests.post(f"{BASE_URL}/data/select-columns", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        result = response.json()
        if response.status_code == 200:
            print(f"✅ Selected columns: {', '.join(selected)}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")

def test_get_statistics(path):
    """Test the data/statistics endpoint."""
    print("\n" + "="*80)
    print("📊 Testing Get Statistics")
    print("="*80)
    response = requests.get(f"{BASE_URL}/data/statistics", params={'path': path})
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        if response.status_code == 200:
            print(f"✅ Statistics retrieved for {len(data)} columns")
            # Show sample stats
            for col_name in list(data.keys())[:2]:
                stats = data[col_name]
                print(f"  • {col_name}: missing={stats.get('missing', 0)}")
        else:
            print(f"❌ Failed: {data.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")

def test_fill_missing(path, columns):
    """Test the data/fill-missing endpoint."""
    if not columns:
        print("\n⏭️  Skipping Fill Missing Values (no columns)")
        return
        
    print("\n" + "="*80)
    print("🔧 Testing Fill Missing Values")
    print("="*80)
    
    # Create a fill strategy for each column based on its name/position
    fill_strategies = {}
    for i, col in enumerate(columns[:2]):
        strategy = 'mean' if i % 2 == 0 else 'mode'
        fill_strategies[col] = strategy
    
    data = {'path': path, 'fill': fill_strategies}
    response = requests.post(f"{BASE_URL}/data/fill-missing", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        result = response.json()
        if response.status_code == 200:
            print(f"✅ Missing values filled with strategies: {fill_strategies}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")

def test_normalize_data(path, columns):
    """Test the data/normalize endpoint."""
    if not columns:
        print("\n⏭️  Skipping Normalize Data (no columns)")
        return
        
    print("\n" + "="*80)
    print("📏 Testing Normalize Data")
    print("="*80)
    
    # Find numeric-sounding columns
    numeric_cols = [col for col in columns if any(word in col.lower() for word in ['value', 'rating', 'id', 'age', 'price', 'amount'])]
    
    if not numeric_cols:
        numeric_cols = [columns[0]] if columns else []
    
    if numeric_cols:
        # Test Z-score normalization
        data = {'path': path, 'method': 'zscore', 'columns': numeric_cols[:1]}
        print("\n  Testing Z-Score Normalization")
        response = requests.post(f"{BASE_URL}/data/normalize", json=data)
        print(f"  Status Code: {response.status_code}")
        try:
            result = response.json()
            if response.status_code == 200:
                print(f"  ✅ Z-score normalized: {numeric_cols[:1]}")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        except json.JSONDecodeError:
            print(f"  ❌ Response not JSON")
        
        # Test Min-Max normalization
        data = {'path': path, 'method': 'minmax', 'columns': numeric_cols[:1]}
        print("\n  Testing Min-Max Normalization")
        response = requests.post(f"{BASE_URL}/data/normalize", json=data)
        print(f"  Status Code: {response.status_code}")
        try:
            result = response.json()
            if response.status_code == 200:
                print(f"  ✅ Min-max normalized: {numeric_cols[:1]}")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        except json.JSONDecodeError:
            print(f"  ❌ Response not JSON")

def test_scatter_plot(path, columns):
    """Test the plot/scatter endpoint."""
    if not columns or len(columns) < 2:
        print("\n⏭️  Skipping Scatter Plot (not enough columns)")
        return
        
    print("\n" + "="*80)
    print("📈 Testing Scatter Plot")
    print("="*80)
    
    x_col = columns[0]
    y_col = columns[1]
    
    data = {'path': path, 'x': x_col, 'y': y_col}
    response = requests.post(f"{BASE_URL}/plot/scatter", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        result = response.json()
        if response.status_code == 200:
            print(f"✅ Scatter plot created: {x_col} vs {y_col}")
            print(f"   Plot URL: {result.get('plot_url', 'N/A')}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")

def test_box_plot(path, columns):
    """Test the plot/box endpoint."""
    if not columns:
        print("\n⏭️  Skipping Box Plot (no columns)")
        return
        
    print("\n" + "="*80)
    print("📦 Testing Box Plot")
    print("="*80)
    
    cols_to_use = columns[:2] if len(columns) >= 2 else columns
    
    data = {'path': path, 'columns': cols_to_use}
    response = requests.post(f"{BASE_URL}/plot/box", json=data)
    print(f"Status Code: {response.status_code}")
    try:
        result = response.json()
        if response.status_code == 200:
            print(f"✅ Box plot created for: {', '.join(cols_to_use)}")
            print(f"   Plot URL: {result.get('plot_url', 'N/A')}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")

def test_categorical_to_numerical(path, columns):
    """Test the data/categorical-to-numerical endpoint."""
    if not columns:
        print("\n⏭️  Skipping Categorical to Numerical Conversion (no columns)")
        return
        
    print("\n" + "="*80)
    print("🔢 Testing Categorical to Numerical Conversion")
    print("="*80)
    
    # Find columns that might be categorical
    categorical_cols = [col for col in columns if col.lower() in ['status', 'category', 'type', 'gender', 'class']]
    
    if not categorical_cols and len(columns) >= 1:
        categorical_cols = [columns[0]]
    
    if not categorical_cols:
        print("⏭️  No suitable categorical columns found")
        return
    
    # Test label encoding
    data = {'path': path, 'method': 'label', 'columns': categorical_cols}
    print("\n  Testing Label Encoding")
    response = requests.post(f"{BASE_URL}/data/categorical-to-numerical", json=data)
    print(f"  Status Code: {response.status_code}")
    try:
        result = response.json()
        if response.status_code == 200:
            print(f"  ✅ Label encoded: {', '.join(categorical_cols)}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"  ❌ Response not JSON")
    
    # Test one-hot encoding
    data = {'path': path, 'method': 'onehot', 'columns': categorical_cols}
    print("\n  Testing One-Hot Encoding")
    response = requests.post(f"{BASE_URL}/data/categorical-to-numerical", json=data)
    print(f"  Status Code: {response.status_code}")
    try:
        result = response.json()
        if response.status_code == 200:
            print(f"  ✅ One-hot encoded: {', '.join(categorical_cols)}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    except json.JSONDecodeError:
        print(f"  ❌ Response not JSON")

def create_sample_data():
    """Create a sample CSV for testing."""  
    print("\n" + "="*80)
    print("🔨 Creating Sample Data for Testing")
    print("="*80)
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
    import numpy as np
    np.random.seed(42)
    for col in df.columns:
        df.loc[df.sample(frac=0.1, random_state=42).index, col] = None
    
    sample_file = 'test_sample_data.csv'
    df.to_csv(sample_file, index=False)
    print(f"✅ Sample data created at {sample_file}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return sample_file

def handle_test_function(func, *args, **kwargs):
    """Execute a test function and handle any exceptions."""  
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"❌ Error in {func.__name__}: {str(e)}")
        return None

def test_save_data(new_path):
    """Test the data/save endpoint."""
    print("\n" + "="*80)
    print("💾 Testing Save Data")
    print("="*80)
    data = {'new_path': new_path}
    response = requests.post(f"{BASE_URL}/data/save", json=data)
    
    print(f"Status Code: {response.status_code}")
    try:
        result = response.json()
        if response.status_code == 200:
            print(f"✅ Data saved to: {result.get('path', new_path)}")
            return result.get('path')
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            return None
    except json.JSONDecodeError:
        print(f"❌ Response content (not JSON): {response.text}")
        return None

def main():
    """Main function to run all tests."""
    print("="*80)
    print("🚀 DATA PROCESSING API TESTS")
    print("="*80)
    print(f"📡 Server URL: {BASE_URL}")
    print("="*80)
    
    # Check if server is running
    if not check_server():
        return
    
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
            # handle_test_function(test_save_data, data_path)
            
            print("\n" + "="*80)
            print("✅ ALL TESTS COMPLETED")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ Failed to upload data. Tests cannot continue.")
            print("="*80)
    except Exception as e:
        print(f"\n❌ Error in main test execution: {str(e)}")

if __name__ == "__main__":
    main()
