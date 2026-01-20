# Frontend API Integration Guide

## Overview

This guide shows how to use the new unified API client (`lib/api.ts`) in your frontend components. The API client matches the test files' format and provides comprehensive error handling.

## Quick Start

### 1. Import the API Client

```typescript
import {
  dataApi,
  plottingApi,
  clusteringApi,
  mlApi,
  ApiError,
  getImageUrl,
} from "@/lib/api";
```

### 2. Basic Usage Pattern

```typescript
const handleAction = async () => {
  setLoading(true);
  setError(null);

  try {
    const result = await dataApi.uploadData(file);
    // Handle success
    console.log(result);
  } catch (error) {
    if (error instanceof ApiError) {
      setError(`Failed: ${error.message}`);
    } else {
      setError("An unknown error occurred");
    }
  } finally {
    setLoading(false);
  }
};
```

## API Modules

### Data Processing API (`dataApi`)

#### Upload Data

```typescript
const result = await dataApi.uploadData(file);
// Returns: { message, columns, head, path }
```

#### Create Sample Data

```typescript
const result = await dataApi.createSample();
// Returns: { message, columns, head, path }
```

#### Get Data Head

```typescript
const head = await dataApi.getDataHead(path);
// Returns: array of row objects
```

#### Get Columns

```typescript
const { columns } = await dataApi.getColumns(path);
// Returns: { columns: string[] }
```

#### Select Columns

```typescript
const result = await dataApi.selectColumns(path, ["col1", "col2"]);
// Returns: { message, selected }
```

#### Get Statistics

```typescript
const stats = await dataApi.getStatistics(path);
// Returns: { columnName: { missing, mean, median, mode } }
```

#### Fill Missing Values

```typescript
const result = await dataApi.fillMissing(path, {
  column1: "mean",
  column2: "median",
  column3: "mode",
});
// Returns: { message }
```

#### Normalize Data

```typescript
const result = await dataApi.normalize(path, "zscore", ["col1", "col2"]);
// method: 'zscore' | 'minmax'
// Returns: { message }
```

#### Categorical to Numerical

```typescript
const result = await dataApi.categoricalToNumerical(path, "label", [
  "category",
]);
// method: 'label' | 'onehot'
// Returns: { message, details }
```

### Plotting API (`plottingApi`)

#### Scatter Plot

```typescript
const result = await plottingApi.scatterPlot(path, "x_column", "y_column");
// Returns: { plot_url }
const imageUrl = getImageUrl(result.plot_url);
```

#### Box Plot

```typescript
const result = await plottingApi.boxPlot(path, ["col1", "col2"]);
// Returns: { plot_url }
const imageUrl = getImageUrl(result.plot_url);
```

### Clustering API (`clusteringApi`)

#### Upload Data for Clustering

```typescript
const result = await clusteringApi.uploadData(file);
// Returns: { message, columns, head, path }
```

#### Select Algorithm

```typescript
const result = await clusteringApi.selectAlgorithm(path, "kmeans");
// Returns: { message }
```

#### Generate Elbow Plot

```typescript
const result = await clusteringApi.generateElbow(path, "kmeans", 10);
// Returns: { image_url }
const imageUrl = getImageUrl(result.image_url);
```

#### Generate Dendrogram

```typescript
const result = await clusteringApi.generateDendrogram(path, "ward", 5);
// method: 'ward' | 'single' | 'complete' | 'average'
// Returns: { image_url }
```

#### Run Clustering

```typescript
const result = await clusteringApi.runClustering(path, "kmeans", { k: 3 });
// Returns: { message, performance, plot_url }
```

#### Run DBSCAN

```typescript
const result = await clusteringApi.runDBSCAN(path, 0.5, 5);
// Returns: { message, performance, plot_url }
```

#### Get Comparison

```typescript
const result = await clusteringApi.getComparison();
// Returns: { comparison: { algorithm: { silhouette, davies_bouldin, calinski_harabasz } } }
```

### Machine Learning API (`mlApi`)

#### Run K-NN

```typescript
const result = await mlApi.runKNN(path, "target_column", 10);
// Returns: { success, metrics: { best_k, accuracy, precision, recall, f1_score }, plot_url }
```

#### Run Naive Bayes

```typescript
const result = await mlApi.runNaiveBayes(path, "target_column");
// Returns: { success, metrics: { accuracy, precision, recall, f1_score } }
```

#### Run Decision Tree

```typescript
const result = await mlApi.runDecisionTree(path, "target_column", "cart");
// algorithmType: 'id3' | 'c4.5' | 'cart'
// Returns: { success, metrics: { accuracy, criterion_used, matrix_plot_url } }
```

#### Run Linear Regression

```typescript
const result = await mlApi.runLinearRegression(path, "target_column");
// Returns: { success, metrics: { mse, rmse, r2_score, coefficients, intercept }, plot_url }
```

#### Run Neural Network

```typescript
const result = await mlApi.runNeuralNetwork(path, "target_column", [100, 50]);
// Returns: { success, metrics: { accuracy, precision, recall, f1_score, iterations }, plot_url }
```

#### Get ML Comparison

```typescript
const result = await mlApi.getComparison();
// Returns: object with algorithm results
```

## Error Handling

### ApiError Class

The API client throws `ApiError` instances with:

- `statusCode`: HTTP status code (0 for network errors)
- `message`: Error message
- `response`: Original response data (if available)

### Example Error Handling

```typescript
try {
  const result = await dataApi.uploadData(file);
} catch (error) {
  if (error instanceof ApiError) {
    if (error.statusCode === 400) {
      console.error("Bad request:", error.message);
    } else if (error.statusCode === 500) {
      console.error("Server error:", error.message);
    } else if (error.statusCode === 0) {
      console.error("Network error:", error.message);
    }

    // Show user-friendly error
    setError(`Failed: ${error.message}`);
  } else {
    setError("An unknown error occurred");
  }
}
```

## Utility Functions

### Check Server Status

```typescript
const isRunning = await checkServer();
if (!isRunning) {
  console.error("Server is not running");
}
```

### Get Image URL

```typescript
// Converts relative paths to absolute URLs
const fullUrl = getImageUrl("/static/plot_123.png");
// Returns: http://localhost:5001/static/plot_123.png
```

## Migration Guide

### Before (Old Approach)

```typescript
const response = await fetch("/api/upload-data", {
  method: "POST",
  body: formData,
});
const data = await response.json();
```

### After (New API Client)

```typescript
const data = await dataApi.uploadData(file);
```

### Benefits

- ✅ Automatic error handling
- ✅ Type safety
- ✅ Consistent API across all components
- ✅ Handles NaN values in JSON
- ✅ Proper image URL construction
- ✅ Matches test files format

## Component Update Checklist

When updating a component:

1. [ ] Import API client: `import { dataApi, ApiError } from '@/lib/api'`
2. [ ] Replace `fetch()` calls with API client methods
3. [ ] Update error handling to use `ApiError`
4. [ ] Use `getImageUrl()` for image paths
5. [ ] Test the component functionality
6. [ ] Verify error states display correctly

## Examples

### Complete Component Example

```typescript
"use client";

import { useState } from "react";
import { dataApi, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";

export default function MyComponent() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const data = await dataApi.uploadData(file);
      setResult(data);
      console.log("Upload successful:", data);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(`Upload failed: ${err.message}`);
      } else {
        setError("An unknown error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <Button onClick={handleUpload} disabled={!file || loading}>
        {loading ? "Uploading..." : "Upload"}
      </Button>
      {error && <div className="text-red-500">{error}</div>}
      {result && <div>Success! Path: {result.path}</div>}
    </div>
  );
}
```

## Configuration

The API URL is configured via environment variable:

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:5001
```

Default: `http://localhost:5001`

## Troubleshooting

### Issue: Network Error

- Ensure Flask backend is running on port 5001
- Check `NEXT_PUBLIC_API_URL` environment variable

### Issue: CORS Error

- Verify CORS is enabled in Flask backend
- Check allowed origins in backend configuration

### Issue: NaN JSON Error

- Fixed automatically by the API client
- NaN values are replaced with null

### Issue: Image Not Loading

- Use `getImageUrl()` to construct full URLs
- Check network tab for actual URL being requested
- Verify image exists on backend

## Testing

The API client matches the test files format, so you can reference:

- `test_data_processing.py` for data API examples
- `test_clustering_logic.py` for clustering API examples
- `test_ml_logic.py` for ML API examples
