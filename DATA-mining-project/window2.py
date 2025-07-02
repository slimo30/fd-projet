# import os
# import uuid
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn_extra.cluster import KMedoids
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from scipy.cluster.hierarchy import dendrogram, linkage
# from werkzeug.utils import secure_filename
# import os
# import uuid
# import matplotlib
# matplotlib.use('Agg')  # Set non-interactive backend
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import arff
# from flask import Flask, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import logging

# # Setup basic logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Define absolute paths for folders
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

# # Create directories if they don't exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['STATIC_FOLDER'] = STATIC_FOLDER
# CORS(app, origins=["http://localhost:3000"])

# # Global DataFrame to store the current data being processed
# global_df = None
# current_filepath = None
# app = Flask(__name__)
# CORS(app)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['STATIC_FOLDER'] = STATIC_FOLDER

# # In-memory storage for models and labels (for comparison)
# stored_results = {}

# # ========== Helper functions ==========
# def save_plot(fig, prefix):
#     filename = f"{prefix}_{uuid.uuid4()}.png"
#     path = os.path.join(app.config['STATIC_FOLDER'], filename)
#     fig.savefig(path)
#     plt.close(fig)
#     return f'/static/{filename}'

# def calculate_metrics(df, labels):
#     if len(set(labels)) < 2:
#         # Silhouette and others need at least 2 clusters
#         return {
#             'silhouette': None,
#             'davies_bouldin': None,
#             'calinski_harabasz': None
#         }
#     return {
#         'silhouette': silhouette_score(df, labels),
#         'davies_bouldin': davies_bouldin_score(df, labels),
#         'calinski_harabasz': calinski_harabasz_score(df, labels)
#     }

# # ========== Routes ==========
# # ========= Upload Route =========
# @app.route('/upload-data-cluster', methods=['POST'])
# def upload_data():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
#     try:
#         df = pd.read_csv(file) if filename.endswith('.csv') else pd.read_csv(file, comment='@')
#         if df.empty:
#             return jsonify({'error': 'Uploaded file is empty'}), 400
        
#         df.to_csv(filepath, index=False)
#         return jsonify({
#             'message': 'Data uploaded successfully',
#             'columns': df.columns.tolist(),
#             'head': df.head().to_dict(orient='records'),
#             'path': filepath
#         })
#     except Exception as e:
#         return jsonify({'error': f"Failed to upload file: {str(e)}"}), 500

# @app.route('/clustering/select', methods=['POST'])
# def select_algorithm():
#     data = request.json
#     required = {'path', 'algorithm'}
#     if not required.issubset(data.keys()):
#         return jsonify({'error': 'Missing required fields'}), 400
#     return jsonify({'message': 'Algorithm selected'})

# @app.route('/clustering/elbow', methods=['POST'])
# def generate_elbow():
#     data = request.json
#     df = pd.read_csv(data['path'])
#     algorithm = data['algorithm']
#     max_k = data.get('max_k', 10)

#     distortions = []
#     for k in range(1, max_k + 1):
#         model = KMeans(n_clusters=k) if algorithm == 'kmeans' else KMedoids(n_clusters=k)
#         model.fit(df)
#         distortions.append(model.inertia_)

#     fig = plt.figure()
#     plt.plot(range(1, max_k + 1), distortions, 'bx-')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Distortion')
#     plt.title(f'Elbow Method ({algorithm.upper()})')

#     image_url = save_plot(fig, 'elbow')
#     return jsonify({'image_url': image_url})

# @app.route('/clustering/dendrogram', methods=['POST'])
# def generate_dendrogram_route():
#     data = request.json
#     df = pd.read_csv(data['path'])
#     method = data.get('method', 'ward')
#     max_clusters = data.get('max_clusters', 5)

#     Z = linkage(df, method=method)

#     fig = plt.figure(figsize=(12, 6))
#     dendrogram(Z, truncate_mode='lastp', p=max_clusters)
#     plt.title(f'Dendrogram ({data["algorithm"].upper()})')
#     plt.xlabel('Sample index')
#     plt.ylabel('Distance')

#     image_url = save_plot(fig, 'dendrogram')
#     return jsonify({'image_url': image_url})

# @app.route('/clustering/run', methods=['POST'])
# def run_clustering():
#     data = request.json
#     df = pd.read_csv(data['path'])
#     algorithm = data['algorithm']

#     try:
#         if algorithm in ['kmeans', 'kmedoids']:
#             k = data['k']
#             model = KMeans(n_clusters=k) if algorithm == 'kmeans' else KMedoids(n_clusters=k)
#             labels = model.fit_predict(df)
        
#         elif algorithm in ['agnes', 'diana']:
#             model = AgglomerativeClustering(
#                 n_clusters=data['n_clusters'],
#                 linkage=data.get('method', 'ward')
#             )
#             labels = model.fit_predict(df)
        
#         else:
#             return jsonify({'error': 'Invalid algorithm in /clustering/run'}), 400

#         metrics = calculate_metrics(df, labels)

#         fig = plt.figure()
#         plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
#         plt.title(f'{algorithm.upper()} Clustering')
#         plot_url = save_plot(fig, 'cluster')

#         # Save for comparison
#         stored_results[algorithm] = metrics

#         return jsonify({
#             'message': f'{algorithm.upper()} clustering complete',
#             'performance': metrics,
#             'plot_url': plot_url
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/clustering/dbscan', methods=['POST'])
# def run_dbscan():
#     data = request.json
#     df = pd.read_csv(data['path'])

#     try:
#         model = DBSCAN(
#             eps=data['eps'],
#             min_samples=data['min_samples']
#         )
#         labels = model.fit_predict(df)

#         metrics = calculate_metrics(df, labels)

#         fig = plt.figure()
#         plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
#         plt.title('DBSCAN Clustering')
#         plot_url = save_plot(fig, 'cluster')

#         # Save for comparison
#         stored_results['dbscan'] = metrics

#         return jsonify({
#             'message': 'DBSCAN clustering complete',
#             'performance': metrics,
#             'plot_url': plot_url
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/clustering/comparison', methods=['GET'])
# def get_comparison():
#     if not stored_results:
#         return jsonify({'error': 'No clustering results found yet'}), 400
#     return jsonify({'comparison': stored_results})

# @app.route('/static/<filename>')
# def serve_static(filename):
#     return send_from_directory(app.config['STATIC_FOLDER'], filename)

# @app.route('/')
# def home():
#     return jsonify({
#         'message': 'API is running',
#         'endpoints': {
#             'upload': '/upload-data (POST)',
#             'clustering': [
#                 '/clustering/select (POST)',
#                 '/clustering/elbow (POST)',
#                 '/clustering/dendrogram (POST)',
#                 '/clustering/run (POST)',
#                 '/clustering/dbscan (POST)',
#                 '/clustering/comparison (GET)'
#             ]
#         }
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for generating figures without a display
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# In-memory storage for models and labels (for comparison)
stored_results = {}

# ========== Helper functions ==========
def save_plot(fig, prefix):
    """Save matplotlib figure to static folder and return URL path"""
    filename = f"{prefix}_{uuid.uuid4()}.png"
    path = os.path.join(app.config['STATIC_FOLDER'], filename)
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
            'silhouette': silhouette_score(df, labels),
            'davies_bouldin': davies_bouldin_score(df, labels),
            'calinski_harabasz': calinski_harabasz_score(df, labels)
        }
    except Exception as e:
        return {
            'error': str(e),
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None
        }

# ========== Routes ==========
@app.route('/upload-data-clustering', methods=['POST'])
def upload_data():
    """Handle file upload and parse CSV data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Handle both standard CSV and ARFF files
        df = pd.read_csv(file) if filename.endswith('.csv') else pd.read_csv(file, comment='@')
        if df.empty:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        # Save the processed file
        df.to_csv(filepath, index=False)
        return jsonify({
            'message': 'Data uploaded successfully',
            'columns': df.columns.tolist(),
            'head': df.head().to_dict(orient='records'),
            'path': filepath
        })
    except Exception as e:
        return jsonify({'error': f"Failed to upload file: {str(e)}"}), 500

@app.route('/clustering/select', methods=['POST'])
def select_algorithm():
    """Validate algorithm selection parameters"""
    data = request.json
    required = {'path', 'algorithm'}
    if not required.issubset(data.keys()):
        return jsonify({'error': 'Missing required fields'}), 400
    return jsonify({'message': 'Algorithm selected'})

@app.route('/clustering/elbow', methods=['POST'])
def generate_elbow():
    """Generate elbow plot for KMeans/KMedoids to determine optimal K"""
    data = request.json
    df = pd.read_csv(data['path'])
    algorithm = data['algorithm']
    max_k = data.get('max_k', 10)

    distortions = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k) if algorithm == 'kmeans' else KMedoids(n_clusters=k)
        model.fit(df)
        distortions.append(model.inertia_)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title(f'Elbow Method ({algorithm.upper()})')
    plt.grid(True, linestyle='--', alpha=0.7)

    image_url = save_plot(fig, 'elbow')
    return jsonify({'image_url': image_url})

@app.route('/clustering/dendrogram', methods=['POST'])
def generate_dendrogram_route():
    """Generate dendrogram visualization for hierarchical clustering"""
    data = request.json
    df = pd.read_csv(data['path'])
    method = data.get('method', 'ward')
    max_clusters = data.get('max_clusters', 5)

    Z = linkage(df, method=method)

    fig = plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=max_clusters)
    plt.title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} linkage)')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.grid(True, linestyle='--', alpha=0.7)

    image_url = save_plot(fig, 'dendrogram')
    return jsonify({'image_url': image_url})

@app.route('/clustering/run', methods=['POST'])
def run_clustering():
    """Run K-means, K-medoids, or Hierarchical clustering"""
    data = request.json
    df = pd.read_csv(data['path'])
    algorithm = data['algorithm']

    try:
        if algorithm in ['kmeans', 'kmedoids']:
            k = data['k']
            model = KMeans(n_clusters=k) if algorithm == 'kmeans' else KMedoids(n_clusters=k)
            labels = model.fit_predict(df)
        
        elif algorithm in ['agnes', 'diana']:
            model = AgglomerativeClustering(
                n_clusters=data['n_clusters'],
                linkage=data.get('method', 'ward')
            )
            labels = model.fit_predict(df)
        
        else:
            return jsonify({'error': 'Invalid algorithm in /clustering/run'}), 400

        metrics = calculate_metrics(df, labels)

        # Create scatter plot visualization
        fig = plt.figure(figsize=(10, 6))
        if df.shape[1] >= 2:  # If we have at least 2 dimensions
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
            plt.title(f'{algorithm.upper()} Clustering Results')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            plt.colorbar(label='Cluster')
        else:
            plt.text(0.5, 0.5, "Not enough dimensions to plot", 
                    horizontalalignment='center', verticalalignment='center')
        
        plot_url = save_plot(fig, 'cluster')

        # Save for comparison
        stored_results[algorithm] = metrics

        return jsonify({
            'message': f'{algorithm.upper()} clustering complete',
            'performance': metrics,
            'plot_url': plot_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clustering/dbscan', methods=['POST'])
def run_dbscan():
    """Run DBSCAN density-based clustering"""
    data = request.json
    df = pd.read_csv(data['path'])

    try:
        model = DBSCAN(
            eps=data['eps'],
            min_samples=data['min_samples']
        )
        labels = model.fit_predict(df)

        metrics = calculate_metrics(df, labels)

        # Create scatter plot visualization
        fig = plt.figure(figsize=(10, 6))
        if df.shape[1] >= 2:  # If we have at least 2 dimensions
            scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
            plt.title('DBSCAN Clustering Results')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            
            # Add legend to distinguish noise points (-1)
            unique_labels = set(labels)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=scatter.cmap(scatter.norm(label)), 
                              markersize=10, label=f'Cluster {label}' if label >= 0 else 'Noise') 
                              for label in unique_labels]
            plt.legend(handles=legend_elements)
        else:
            plt.text(0.5, 0.5, "Not enough dimensions to plot", 
                    horizontalalignment='center', verticalalignment='center')
        
        plot_url = save_plot(fig, 'cluster')

        # Save for comparison
        stored_results['dbscan'] = metrics

        return jsonify({
            'message': 'DBSCAN clustering complete',
            'performance': metrics,
            'plot_url': plot_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clustering/comparison', methods=['GET'])
def get_comparison():
    """Compare metrics across different clustering algorithms"""
    if not stored_results:
        return jsonify({'error': 'No clustering results found yet'}), 400
    
    # Create comparison visualization if multiple algorithms are available
    if len(stored_results) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        algorithms = list(stored_results.keys())
        
        x = np.arange(len(metrics))
        width = 0.2
        multiplier = 0
        
        for algorithm, results in stored_results.items():
            values = [results.get(metric) for metric in metrics]
            offset = width * multiplier
            rects = ax.bar(x + offset, values, width, label=algorithm.upper())
            multiplier += 1
        
        ax.set_ylabel('Score')
        ax.set_title('Clustering Algorithm Metrics Comparison')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'])
        ax.legend(loc='best')
        
        comparison_url = save_plot(fig, 'comparison')
        return jsonify({
            'comparison': stored_results,
            'plot_url': comparison_url
        })
    
    return jsonify({'comparison': stored_results})

# @app.route('/static/<filename>')
# def serve_static(filename):
#     """Serve static files (images)"""
#     return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/')
def home():
    """API root endpoint with documentation"""
    return jsonify({
        'message': 'Clustering API is running',
        'endpoints': {
            'upload': '/upload-data (POST)',
            'clustering': [
                '/clustering/select (POST)',
                '/clustering/elbow (POST)',
                '/clustering/dendrogram (POST)',
                '/clustering/run (POST)',
                '/clustering/dbscan (POST)',
                '/clustering/comparison (GET)'
            ]
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)