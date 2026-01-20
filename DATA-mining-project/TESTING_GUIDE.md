# Testing Guide for Data Mining Project

## Prerequisites

1. **Start the Flask Server**
   ```bash
   python run.py
   ```
   The server should be running at `http://localhost:5001`

2. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

## Running All Tests

### Option 1: Run All Tests at Once
```bash
python run_all_tests.py
```

### Option 2: Run Individual Test Suites

#### Test Data Processing API
```bash
python test_data_processing.py
```

#### Test Clustering Algorithms
```bash
python test_clustering_logic.py
```

#### Test Machine Learning Algorithms
```bash
python test_ml_logic.py
```

## What Each Test Suite Covers

### 1. Data Processing Tests (`test_data_processing.py`)
- ✅ File upload
- ✅ Data retrieval (head, columns)
- ✅ Column selection
- ✅ Statistics calculation
- ✅ Missing value handling (mean, median, mode)
- ✅ Data normalization (z-score, min-max)
- ✅ Categorical encoding (label, one-hot)
- ✅ Plotting (scatter, box plots)

### 2. Clustering Tests (`test_clustering_logic.py`)
- ✅ Data upload for clustering
- ✅ Algorithm selection
- ✅ Elbow method (K-Means, K-Medoids)
- ✅ Dendrogram generation
- ✅ K-Means clustering
- ✅ K-Medoids clustering
- ✅ AGNES (hierarchical)
- ✅ DIANA (hierarchical)
- ✅ DBSCAN (density-based)
- ✅ Algorithm comparison

### 3. Machine Learning Tests (`test_ml_logic.py`)
- ✅ Data preparation and encoding
- ✅ K-NN classification
- ✅ Naive Bayes classification
- ✅ Decision Tree (ID3, C4.5, CART)
- ✅ Linear Regression
- ✅ Neural Networks (MLP)
- ✅ Algorithm comparison
- ✅ Metrics validation

## Expected Output

Each test will show:
- ✅ Green checkmarks for successful tests
- ❌ Red X marks for failed tests
- 📊 Detailed metrics and results

## Troubleshooting

### Server Not Running
```
❌ Could not connect to http://localhost:5001
```
**Solution:** Start the server with `python run.py`

### Import Errors
```
ModuleNotFoundError: No module named 'X'
```
**Solution:** Install requirements: `pip install -r requirements.txt`

### Port Already in Use
```
Address already in use
```
**Solution:** Kill the existing process or change the port in `run.py`

## API Endpoints Reference

### Data Processing
- `POST /upload-data` - Upload CSV/ARFF file
- `GET /data/head` - Get first rows
- `GET /data/columns` - Get column names
- `POST /data/select-columns` - Select specific columns
- `GET /data/statistics` - Get statistics
- `POST /data/fill-missing` - Fill missing values
- `POST /data/normalize` - Normalize data
- `POST /data/categorical-to-numerical` - Encode categorical data

### Plotting
- `POST /plot/scatter` - Generate scatter plot
- `POST /plot/box` - Generate box plot

### Clustering
- `POST /upload-data-clustering` - Upload data for clustering
- `POST /clustering/select` - Select algorithm
- `POST /clustering/elbow` - Generate elbow plot
- `POST /clustering/dendrogram` - Generate dendrogram
- `POST /clustering/run` - Run clustering algorithm
- `POST /clustering/dbscan` - Run DBSCAN
- `GET /clustering/comparison` - Compare algorithms

### Machine Learning
- `POST /ml/knn` - K-Nearest Neighbors
- `POST /ml/naive-bayes` - Naive Bayes
- `POST /ml/decision-tree` - Decision Tree
- `POST /ml/linear-regression` - Linear Regression
- `POST /ml/neural-network` - Neural Network
- `GET /ml/comparison` - Compare ML algorithms

