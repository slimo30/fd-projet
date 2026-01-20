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
        if response.status_code == 200:
            print(f"✅ Status: {response.status_code}")
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
                        value = comp[algo].get(metric)
                        row.append(f"{value:.4f}" if value is not None else "N/A")
                    comparison_data.append(row)
                
                print(tabulate(comparison_data, 
                      headers=["Algorithm"] + [m.replace('_', ' ').title() for m in metrics], 
                      tablefmt="grid"))
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        print(f"Raw response: {response.text[:500]}")

def test_api():
    """Test all endpoints of the clustering API"""
    
    # Test 1: Home endpoint
    print("\n🔍 Testing home endpoint")
    response = requests.get(f"{API_URL}/")
    print_response(response, "Home Endpoint")
    
    # Test 2: Upload data
    print("\n🔍 Testing upload endpoint")

    with open(TEST_DATA_PATH, 'rb') as f:
        files = {'file': (TEST_DATA_PATH, f, 'text/csv')}
        response = requests.post(f"{API_URL}/upload-data-clustering", files=files)

    print_response(response, "Upload Data")
    
    # Save the file path from the response
    if response.status_code == 200:
        file_path = response.json().get('path')
        print(f"Using file path: {file_path}")
    else:
        print("❌ Upload failed. Cannot continue.")
        return
    
    # Test 3: Algorithm selection
    print("\n🔍 Testing algorithm selection")
    data = {"path": file_path, "algorithm": "kmeans"}
    response = requests.post(f"{API_URL}/clustering/select", json=data)
    print_response(response, "Algorithm Selection")
    
    # Test 4: Elbow method
    print("\n🔍 Testing elbow method")
    data = {"path": file_path, "algorithm": "kmeans", "max_k": 10}
    response = requests.post(f"{API_URL}/clustering/elbow", json=data)
    print_response(response, "Elbow Method")
    
    # Test 5: Dendrogram
    print("\n🔍 Testing dendrogram generation")
    data = {"path": file_path, "algorithm": "agnes", "method": "ward", "max_clusters": 5}
    response = requests.post(f"{API_URL}/clustering/dendrogram", json=data)
    print_response(response, "Dendrogram")
    
    # Test 6: Run KMeans
    print("\n🔍 Testing KMeans clustering")
    data = {"path": file_path, "algorithm": "kmeans", "k": 3}
    response = requests.post(f"{API_URL}/clustering/run", json=data)
    print_response(response, "KMeans Clustering")
    
    # Test 7: Run KMedoids
    print("\n🔍 Testing KMedoids clustering")
    data = {"path": file_path, "algorithm": "kmedoids", "k": 3}
    response = requests.post(f"{API_URL}/clustering/run", json=data)
    print_response(response, "KMedoids Clustering")
    
    # Test 8: Run Hierarchical Clustering
    print("\n🔍 Testing Hierarchical clustering")
    data = {"path": file_path, "algorithm": "agnes", "n_clusters": 3, "method": "ward"}
    response = requests.post(f"{API_URL}/clustering/run", json=data)
    print_response(response, "Hierarchical Clustering")
    
    # Test 9: Run DBSCAN
    print("\n🔍 Testing DBSCAN clustering")
    data = {"path": file_path, "eps": 0.5, "min_samples": 5}
    response = requests.post(f"{API_URL}/clustering/dbscan", json=data)
    print_response(response, "DBSCAN Clustering")
    
    # Test 10: Get Comparison
    print("\n🔍 Testing Comparison")
    response = requests.get(f"{API_URL}/clustering/comparison")
    print_response(response, "Clustering Comparison")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    try:
        # Create test data
        create_test_data()
        
        # Run tests
        test_api()
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection error: Could not connect to {API_URL}")
        print("Make sure the API server is running.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")