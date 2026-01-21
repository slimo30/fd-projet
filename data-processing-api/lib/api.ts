/**
 * API Client for Data Mining Project
 * Matches the test files' format with comprehensive error handling
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

class ApiError extends Error {
  constructor(public statusCode: number, message: string, public response?: any) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Base fetch wrapper with error handling
 */
async function apiFetch<T = any>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    let data;
    const contentType = response.headers.get('content-type');

    if (contentType && contentType.includes('application/json')) {
      const text = await response.text();
      // Replace NaN with null before parsing JSON
      const sanitizedText = text.replace(/:\s*NaN/g, ': null');
      data = JSON.parse(sanitizedText);
    } else {
      const text = await response.text();
      data = { message: text };
    }

    if (!response.ok) {
      throw new ApiError(
        response.status,
        data.error || data.message || `HTTP ${response.status}`,
        data
      );
    }

    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    // Network or other errors
    throw new ApiError(
      0,
      error instanceof Error ? error.message : 'Network error',
      null
    );
  }
}

// ============================================================================
// DATA PROCESSING API
// ============================================================================

export const dataApi = {
  /**
   * Upload data file
   */
  uploadData: async (file: File): Promise<{
    message: string;
    columns: string[];
    head: any[];
    path: string;
  }> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/upload-data`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new ApiError(response.status, error.error || 'Upload failed');
    }

    return response.json();
  },

  /**
   * Create sample data
   */
  createSample: async (clean: boolean = false): Promise<{
    message: string;
    columns: string[];
    head: any[];
    path: string;
    is_clean?: boolean;
  }> => {
    return apiFetch('/create-sample', {
      method: 'POST',
      body: JSON.stringify({ clean }),
    });
  },

  /**
   * Get data head (first rows or representative sample)
   */
  getDataHead: async (path: string, limit: number = 40, sample: boolean = true): Promise<any[]> => {
    const params = new URLSearchParams({ path });
    if (limit !== undefined) {
      params.append('limit', limit.toString());
    }
    if (sample !== undefined) {
      params.append('sample', sample.toString());
    }
    return apiFetch(`/data/head?${params.toString()}`);
  },

  /**
   * Get data columns
   */
  getColumns: async (path: string): Promise<{ columns: string[] }> => {
    return apiFetch(`/data/columns?path=${encodeURIComponent(path)}`);
  },

  /**
   * Select specific columns
   */
  selectColumns: async (path: string, columns: string[]): Promise<{
    message: string;
    selected: string[];
  }> => {
    return apiFetch('/data/select-columns', {
      method: 'POST',
      body: JSON.stringify({ path, columns }),
    });
  },

  /**
   * Get statistics for all columns
   */
  getStatistics: async (path: string): Promise<Record<string, {
    missing: number;
    mean?: number;
    median?: number;
    mode?: any;
  }>> => {
    return apiFetch(`/data/statistics?path=${encodeURIComponent(path)}`);
  },

  /**
   * Fill missing values
   */
  fillMissing: async (
    path: string,
    fillStrategies: Record<string, 'mean' | 'median' | 'mode'>
  ): Promise<{ message: string }> => {
    return apiFetch('/data/fill-missing', {
      method: 'POST',
      body: JSON.stringify({ path, fill: fillStrategies }),
    });
  },

  /**
   * Normalize data
   */
  normalize: async (
    path: string,
    method: 'zscore' | 'minmax',
    columns: string[]
  ): Promise<{ message: string }> => {
    return apiFetch('/data/normalize', {
      method: 'POST',
      body: JSON.stringify({ path, method, columns }),
    });
  },

  /**
   * Convert categorical to numerical
   */
  categoricalToNumerical: async (
    path: string,
    method: 'label' | 'onehot',
    columns: string[]
  ): Promise<{
    message: string;
    details: any;
  }> => {
    return apiFetch('/data/categorical-to-numerical', {
      method: 'POST',
      body: JSON.stringify({ path, method, columns }),
    });
  },

  /**
   * Binarize column
   */
  binarize: async (
    path: string,
    column: string,
    options: { zero_group?: string[], threshold?: number }
  ): Promise<{ message: string }> => {
    return apiFetch('/data/binarize', {
      method: 'POST',
      body: JSON.stringify({ path, column, ...options }),
    });
  },

  /**
   * Ordinal map column
   */
  ordinalMap: async (
    path: string,
    column: string,
    order: string[]
  ): Promise<{ message: string }> => {
    return apiFetch('/data/ordinal-map', {
      method: 'POST',
      body: JSON.stringify({ path, column, order }),
    });
  },

  /**
   * Save data
   */
  saveData: async (newPath: string): Promise<{
    message: string;
    path: string;
  }> => {
    return apiFetch('/data/save', {
      method: 'POST',
      body: JSON.stringify({ new_path: newPath }),
    });
  },

  /**
   * Generate KNN optimization plot
   */
  getKNNOptimization: async (
    path: string,
    column: string,
    maxK: number = 20
  ): Promise<{
    success: boolean;
    plot_url: string;
    optimal_k: number;
    min_error: number;
    k_values: number[];
    errors: number[];
    message: string;
  }> => {
    return apiFetch('/data/knn-optimization', {
      method: 'POST',
      body: JSON.stringify({ path, column, max_k: maxK }),
    });
  },
};

// ============================================================================
// PLOTTING API
// ============================================================================

export const plottingApi = {
  /**
   * Generate scatter plot
   */
  scatterPlot: async (
    path: string,
    x: string,
    y: string
  ): Promise<{ plot_url: string }> => {
    return apiFetch('/plot/scatter', {
      method: 'POST',
      body: JSON.stringify({ path, x, y }),
    });
  },

  /**
   * Generate box plot
   */
  boxPlot: async (
    path: string,
    columns: string[]
  ): Promise<{ plot_url: string }> => {
    return apiFetch('/plot/box', {
      method: 'POST',
      body: JSON.stringify({ path, columns }),
    });
  },
};

// ============================================================================
// CLUSTERING API
// ============================================================================

export const clusteringApi = {
  /**
   * Upload data for clustering
   */
  uploadData: async (file: File): Promise<{
    message: string;
    columns: string[];
    head: any[];
    path: string;
  }> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/upload-data-clustering`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new ApiError(response.status, error.error || 'Upload failed');
    }

    return response.json();
  },

  /**
   * Select clustering algorithm
   */
  selectAlgorithm: async (
    path: string,
    algorithm: string
  ): Promise<{ message: string }> => {
    return apiFetch('/clustering/select', {
      method: 'POST',
      body: JSON.stringify({ path, algorithm }),
    });
  },

  /**
   * Generate elbow plot
   */
  generateElbow: async (
    path: string,
    algorithm: 'kmeans' | 'kmedoids',
    maxK: number = 10
  ): Promise<{ image_url: string }> => {
    return apiFetch('/clustering/elbow', {
      method: 'POST',
      body: JSON.stringify({ path, algorithm, max_k: maxK }),
    });
  },

  /**
   * Generate dendrogram
   */
  generateDendrogram: async (
    path: string,
    method: 'ward' | 'single' | 'complete' | 'average' = 'ward',
    maxClusters: number = 5
  ): Promise<{ image_url: string }> => {
    return apiFetch('/clustering/dendrogram', {
      method: 'POST',
      body: JSON.stringify({ path, method, max_clusters: maxClusters }),
    });
  },

  /**
   * Run clustering algorithm
   */
  runClustering: async (
    path: string,
    algorithm: 'kmeans' | 'kmedoids' | 'agnes' | 'diana',
    params: any
  ): Promise<{
    message: string;
    performance: {
      silhouette: number;
      davies_bouldin: number;
      calinski_harabasz: number;
    };
    plot_url: string;
  }> => {
    return apiFetch('/clustering/run', {
      method: 'POST',
      body: JSON.stringify({ path, algorithm, ...params }),
    });
  },

  /**
   * Run DBSCAN clustering
   */
  runDBSCAN: async (
    path: string,
    eps: number,
    minSamples: number
  ): Promise<{
    message: string;
    performance: {
      silhouette: number;
      davies_bouldin: number;
      calinski_harabasz: number;
    };
    plot_url: string;
  }> => {
    return apiFetch('/clustering/dbscan', {
      method: 'POST',
      body: JSON.stringify({ path, eps, min_samples: minSamples }),
    });
  },

  /**
   * Get clustering comparison
   */
  getComparison: async (): Promise<{
    comparison: Record<string, {
      silhouette: number;
      davies_bouldin: number;
      calinski_harabasz: number;
    }>;
  }> => {
    return apiFetch('/clustering/comparison');
  },
};

// ============================================================================
// MACHINE LEARNING API
// ============================================================================

export const mlApi = {
  /**
   * Run K-NN algorithm
   */
  runKNN: async (
    path: string,
    target: string,
    maxK: number = 10
  ): Promise<{
    success: boolean;
    metrics: {
      best_k: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
    };
    plot_url: string;
  }> => {
    return apiFetch('/ml/knn', {
      method: 'POST',
      body: JSON.stringify({ path, target, max_k: maxK }),
    });
  },

  /**
   * Run Naive Bayes algorithm
   */
  runNaiveBayes: async (
    path: string,
    target: string
  ): Promise<{
    success: boolean;
    metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
    };
  }> => {
    return apiFetch('/ml/naive-bayes', {
      method: 'POST',
      body: JSON.stringify({ path, target }),
    });
  },

  /**
   * Run Decision Tree algorithm
   */
  runDecisionTree: async (
    path: string,
    target: string,
    algorithmType: 'id3' | 'c4.5' | 'cart' = 'cart'
  ): Promise<{
    success: boolean;
    metrics: {
      accuracy: number;
      criterion_used: string;
      matrix_plot_url: string;
    };
  }> => {
    return apiFetch('/ml/decision-tree', {
      method: 'POST',
      body: JSON.stringify({ path, target, algorithm_type: algorithmType }),
    });
  },

  /**
   * Run Linear Regression algorithm
   */
  runLinearRegression: async (
    path: string,
    target: string
  ): Promise<{
    success: boolean;
    metrics: {
      mse: number;
      rmse: number;
      r2_score: number;
      coefficients: number[];
      intercept: number;
    };
    plot_url: string;
  }> => {
    return apiFetch('/ml/linear-regression', {
      method: 'POST',
      body: JSON.stringify({ path, target }),
    });
  },

  /**
   * Run Neural Network algorithm
   */
  runNeuralNetwork: async (
    path: string,
    target: string,
    hiddenLayers: number[] = [100]
  ): Promise<{
    success: boolean;
    metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      iterations: number;
    };
    plot_url: string;
  }> => {
    return apiFetch('/ml/neural-network', {
      method: 'POST',
      body: JSON.stringify({ path, target, hidden_layers: hiddenLayers }),
    });
  },

  /**
   * Get ML comparison
   */
  getComparison: async (): Promise<Record<string, any>> => {
    return apiFetch('/ml/comparison');
  },

  /**
   * Make prediction with trained model
   */
  predict: async (
    algorithm: string,
    input: number[]
  ): Promise<{
    success: boolean;
    prediction: number | number[];
    features_used: string[];
    input_values: number[];
    probability?: number[];
  }> => {
    return apiFetch(`/ml/predict/${algorithm}`, {
      method: 'POST',
      body: JSON.stringify({ input }),
    });
  },

  /**
   * Get trained model information
   */
  getModelInfo: async (
    algorithm: string
  ): Promise<{
    algorithm: string;
    features: string[];
    has_scaler: boolean;
    is_trained: boolean;
  }> => {
    return apiFetch(`/ml/model-info/${algorithm}`);
  },
};

// ============================================================================
// CNN API
// ============================================================================

export const cnnApi = {
  getDatasets: async (): Promise<{
    success: boolean;
    datasets: Record<string, any>;
  }> => {
    return apiFetch('/cnn/datasets');
  },

  getDatasetInfo: async (dataset: string): Promise<{
    success: boolean;
    info: any;
  }> => {
    return apiFetch('/cnn/info', {
      method: 'POST',
      body: JSON.stringify({ dataset }),
    });
  },

  visualizeSamples: async (dataset: string, num_samples: number = 10): Promise<{
    success: boolean;
    plot_url: string;
    dataset: string;
    num_samples: number;
  }> => {
    return apiFetch('/cnn/visualize-samples', {
      method: 'POST',
      body: JSON.stringify({ dataset, num_samples }),
    });
  },

  train: async (
    dataset: string,
    epochs: number = 10,
    batch_size: number = 32,
    test_size: number = 0.2
  ): Promise<{
    success: boolean;
    metrics: any;
    model_summary: string;
    history_plot_url: string;
    confusion_matrix_plot_url: string;
    training_params: any;
  }> => {
    return apiFetch('/cnn/train', {
      method: 'POST',
      body: JSON.stringify({ dataset, epochs, batch_size, test_size }),
    });
  },

  getPredictions: async (): Promise<{
    success: boolean;
    plot_url: string;
    dataset: string;
    predictions: any[];
  }> => {
    return apiFetch('/cnn/predictions');
  },

  getErrorAnalysis: async (): Promise<{
    success: boolean;
    plot_url?: string;
    dataset?: string;
    num_errors: number;
    total_samples?: number;
    error_rate?: number;
    errors?: any[];
    message?: string;
  }> => {
    return apiFetch('/cnn/error-analysis');
  },

  getArchitecture: async (dataset: string): Promise<{
    success: boolean;
    dataset: string;
    model_summary: string;
    layers: any[];
    total_params: number;
    input_shape: number[];
    num_classes: number;
  }> => {
    return apiFetch('/cnn/architecture', {
      method: 'POST',
      body: JSON.stringify({ dataset }),
    });
  },

  compareDatasets: async (): Promise<{
    success: boolean;
    plot_url: string;
    comparison: any;
  }> => {
    return apiFetch('/cnn/compare-datasets');
  },
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if server is running
 */
export async function checkServer(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/`, { method: 'GET' });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get full image URL
 */
export function getImageUrl(path: string): string {
  if (path.startsWith('http')) {
    return path;
  }
  return `${API_URL}${path}`;
}

export { ApiError };
