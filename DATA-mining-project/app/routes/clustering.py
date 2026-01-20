from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from app.state import state
from app.utils.helpers import is_safe_path, save_plot, calculate_metrics

clustering_bp = Blueprint('clustering', __name__)

@clustering_bp.route('/upload-data-clustering', methods=['POST'])
def upload_data_cluster():
    # Note: This is similar to the main data upload but specific to clustering window context
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    try:
        df = pd.read_csv(file) if filename.endswith('.csv') else pd.read_csv(file, comment='@')
        if df.empty:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        df.to_csv(filepath, index=False)
        # Update global state as well, though clustering might be independent in some flows
        state.clustering_df = df # Using a separate attribute if needed or shared state
        
        return jsonify({
            'message': 'Data uploaded successfully',
            'columns': df.columns.tolist(),
            'head': df.head().to_dict(orient='records'),
            'path': filepath
        })
    except Exception as e:
        return jsonify({'error': f"Failed to upload file: {str(e)}"}), 500

@clustering_bp.route('/clustering/select', methods=['POST'])
def select_algorithm():
    data = request.json
    required = {'path', 'algorithm'}
    if not required.issubset(data.keys()):
        return jsonify({'error': 'Missing required fields'}), 400
    return jsonify({'message': 'Algorithm selected'})

@clustering_bp.route('/clustering/elbow', methods=['POST'])
def generate_elbow():
    data = request.json
    try:
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

        image_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'elbow')
        return jsonify({'image_url': image_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@clustering_bp.route('/clustering/dendrogram', methods=['POST'])
def generate_dendrogram_route():
    data = request.json
    try:
        df = pd.read_csv(data['path'])
        method = data.get('method', 'ward')
        max_clusters = data.get('max_clusters', 5)

        Z = linkage(df, method=method)

        fig = plt.figure(figsize=(12, 6))
        dendrogram(Z, truncate_mode='lastp', p=max_clusters)
        plt.title(f'Dendrogram ({data.get("algorithm", "Hierarchical").upper()})')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')

        image_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'dendrogram')
        return jsonify({'image_url': image_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@clustering_bp.route('/clustering/run', methods=['POST'])
def run_clustering():
    data = request.json
    try:
        df = pd.read_csv(data['path'])
        algorithm = data['algorithm']

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

        fig = plt.figure()
        if df.shape[1] >= 2:
             plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
             plt.title(f'{algorithm.upper()} Clustering')
        else:
             plt.text(0.5, 0.5, "Not enough dimensions to plot")
             
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cluster')

        # Save for comparison
        state.clustering_results[algorithm] = metrics

        return jsonify({
            'message': f'{algorithm.upper()} clustering complete',
            'performance': metrics,
            'plot_url': plot_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@clustering_bp.route('/clustering/dbscan', methods=['POST'])
def run_dbscan():
    data = request.json
    try:
        df = pd.read_csv(data['path']) # Always read from path to ensure fresh data

        model = DBSCAN(
            eps=data['eps'],
            min_samples=data['min_samples']
        )
        labels = model.fit_predict(df)

        metrics = calculate_metrics(df, labels)

        fig = plt.figure()
        if df.shape[1] >= 2:
             plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
             plt.title('DBSCAN Clustering')
        else:
             plt.text(0.5, 0.5, "Not enough dimensions to plot")

        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cluster')

        # Save for comparison
        state.clustering_results['dbscan'] = metrics

        return jsonify({
            'message': 'DBSCAN clustering complete',
            'performance': metrics,
            'plot_url': plot_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@clustering_bp.route('/clustering/comparison', methods=['GET'])
def get_comparison():
    if not state.clustering_results:
        return jsonify({'error': 'No clustering results found yet'}), 400
    return jsonify({'comparison': state.clustering_results})
