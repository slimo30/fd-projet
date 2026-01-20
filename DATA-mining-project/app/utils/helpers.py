import os
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def is_safe_path(base, path):
    """Check if path is within the base directory to prevent traversal attacks."""
    base = os.path.abspath(base)
    path = os.path.abspath(path)
    return os.path.commonpath([base, path]) == base

def convert_to_native(val):
    """Convert NumPy types to native Python types"""
    if isinstance(val, (np.int64, np.float64)):
        return val.item()  # Convert NumPy scalar to native Python type
    return val

def save_plot(fig, static_folder, prefix):
    """Save matplotlib figure to static folder and return URL path"""
    filename = f"{prefix}_{uuid.uuid4()}.png"
    path = os.path.join(static_folder, filename)
    fig.savefig(path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return f'/static/{filename}'

def calculate_metrics(df, labels):
    """Calculate cluster quality metrics if there are at least 2 clusters"""
    if len(set(labels)) < 2:
        # Silhouette and others need at least 2 clusters
        return {
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None
        }
    try:
        return {
            'silhouette': float(silhouette_score(df, labels)),
            'davies_bouldin': float(davies_bouldin_score(df, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(df, labels))
        }
    except Exception as e:
        return {
            'error': str(e),
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None
        }
