import requests
import json
import pandas as pd
import numpy as np
import os
import time
from tabulate import tabulate
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://localhost:5001"
TEST_DATA_PATH = "test_sample_data_copy.csv"

# Global variable to store uploaded file path
UPLOADED_FILE_PATH = None

# Create test data if it doesn't exist
def create_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Creating test data at {TEST_DATA_PATH}")
        
        # Generate three clusters in 2D
        np.random.seed(42)
        n_samples = 300
        
        # Cluster 1
        cluster1 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
        
        # Cluster 2
        cluster2 = np.random.randn(n_samples, 2) * 0.5 + np.array([-2, -2])
        
        # Cluster 3
        cluster3 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, -2])
        
        # Combine clusters
        data = np.vstack([cluster1, cluster2, cluster3])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['feature1', 'feature2'])
        
        # Save data
        df.to_csv(TEST_DATA_PATH, index=False)
        print(f"Test data created with {len(df)} samples and {df.shape[1]} features")
    else:
        print(f"Test data already exists at {TEST_DATA_PATH}")

def print_response(response, title):
    """Pretty print API response"""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    try:
        if response.status_code in [200, 201]:
            print(f"✅ Status: {response.status_code}")
            try:
                data = response.json()
                
                # Print message if available
                if 'message' in data:
                    print(f"Message: {data['message']}")
                
                # Print plot URL if available
                if 'plot_url' in data:
                    print(f"Plot URL: {data['plot_url']}")
                
                # Print image URL if available
                if 'image_url' in data:
                    print(f"Image URL: {data['image_url']}")
                
                # Print file path if available (for upload)
                if 'path' in data:
                    print(f"File Path: {data['path']}")

                # Print performance metrics if available
                if 'performance' in data:
                    perf = data['performance']
                    print("\nPerformance Metrics:")
                    metrics = []
                    for metric, value in perf.items():
                        if value is not None:
                            metrics.append([metric, f"{value:.4f}"])
                        else:
                            metrics.append([metric, "N/A"])
                    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid"))
                
                # Print comparison if available
                if 'comparison' in data:
                    comp = data['comparison']
                    print("\nComparison Metrics:")
                    algorithms = list(comp.keys())
                    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
                    
                    comparison_data = []
                    for algo in algorithms:
                        row = [algo]
                        for metric in metrics:
                            if algo in comp and isinstance(comp[algo], dict):
                                value = comp[algo].get(metric)
                                row.append(f"{value:.4f}" if value is not None else "N/A")
                            else:
                                row.append("N/A")
                        comparison_data.append(row)
                    
                    print(tabulate(comparison_data, 
                          headers=["Algorithm"] + [m.replace('_', ' ').title() for m in metrics], 
                          tablefmt="grid"))
            except ValueError:
                print("Response is not JSON")
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        try:
            print(f"Raw response: {response.text[:500]}")
        except:
            pass

def test_home():
    """Test Home endpoint"""
    print("\n🔍 Testing home endpoint")
    try:
        response = requests.get(f"{API_URL}/")
        # Since the main app might not have a root route defined in what we saw, 
        # but usually does, we print result but don't fail hard if 404
        print_response(response, "Home Endpoint") 
    except Exception as e:
        print(f"Skipping home test: {e}")

def test_upload():
    """Test data upload"""
    global UPLOADED_FILE_PATH
    print("\n🔍 Testing upload endpoint")
    
    with open(TEST_DATA_PATH, 'rb') as f:
        files = {'file': (TEST_DATA_PATH, f, 'text/csv')}
        response = requests.post(f"{API_URL}/upload-data-clustering", files=files)
        
    print_response(response, "Upload Data")
    
    if response.status_code == 200:
        UPLOADED_FILE_PATH = response.json().get('path')
        return True
    return False

def test_select_algorithm(algo):
    """Test algorithm selection"""
    print(f"\n🔍 Testing select algorithm: {algo}")
    payload = {
        'path': UPLOADED_FILE_PATH,
        'algorithm': algo
    }
    response = requests.post(f"{API_URL}/clustering/select", json=payload)
    print_response(response, f"Select {algo}")

def test_elbow(algo):
    """Test Elbow method"""
    print(f"\n🔍 Testing Elbow method for {algo}")
    payload = {
        'path': UPLOADED_FILE_PATH,
        'algorithm': algo,
        'max_k': 5
    }
    response = requests.post(f"{API_URL}/clustering/elbow", json=payload)
    print_response(response, f"Elbow {algo}")

def test_dendrogram(method='ward'):
    """Test Dendrogram"""
    print(f"\n🔍 Testing Dendrogram with method {method}")
    payload = {
        'path': UPLOADED_FILE_PATH,
        'method': method,
        'max_clusters': 5
    }
    response = requests.post(f"{API_URL}/clustering/dendrogram", json=payload)
    print_response(response, f"Dendrogram {method}")

def test_run_clustering(algo, params):
    """Test running a specific clustering algorithm"""
    print(f"\n🔍 Testing Run {algo.upper()}")
    payload = {
        'path': UPLOADED_FILE_PATH,
        'algorithm': algo,
        **params
    }
    response = requests.post(f"{API_URL}/clustering/run", json=payload)
    print_response(response, f"Run {algo.upper()}")

def test_dbscan():
    """Test DBSCAN"""
    print(f"\n🔍 Testing DBSCAN")
    payload = {
        'path': UPLOADED_FILE_PATH,
        'eps': 0.5,
        'min_samples': 5
    }
    response = requests.post(f"{API_URL}/clustering/dbscan", json=payload)
    print_response(response, "Run DBSCAN")

def test_comparison():
    """Test Comparison endpoint"""
    print(f"\n🔍 Testing Comparison")
    response = requests.get(f"{API_URL}/clustering/comparison")
    print_response(response, "Comparison Results")

def run_all_tests():
    """Execute all tests sequentially"""
    create_test_data()
    
    # 1. Base Tests
    # test_home() # Optional
    
    # 2. Upload (Critical)
    if not test_upload():
        print("❌ Upload failed. Aborting tests.")
        return

    # 3. Exploratory Methods
    test_select_algorithm('kmeans')
    test_elbow('kmeans')
    test_elbow('kmedoids')
    test_dendrogram('ward')
    
    # 4. Partitioning Algorithms
    test_run_clustering('kmeans', {'k': 3})
    test_run_clustering('kmedoids', {'k': 3})
    
    # 5. Hierarchical Algorithms
    test_run_clustering('agnes', {'n_clusters': 3, 'method': 'ward'})
    test_run_clustering('diana', {'n_clusters': 3, 'method': 'ward'}) # Note: using same impl as agnes currently
    
    # 6. Density Based
    test_dbscan()
    
    # 7. Comparison
    test_comparison()

if __name__ == "__main__":
    print("🚀 Starting Clustering Logic Tests...")
    try:
        # Check if server is running basically
        requests.get(API_URL)
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {API_URL}. Is the server running?")
        print("Run 'python run.py' in a separate terminal.")