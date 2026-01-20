from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, mean_squared_error, r2_score,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.utils.helpers import save_plot
from app.state import state


ml_bp = Blueprint('ml', __name__)


def prepare_data(df, target_col, test_size=0.2, is_classification=True):
    """Helper to split data and encode target if necessary.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Ratio for test split (default 0.2)
        is_classification: Whether this is a classification task (default True)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    df_clean = df.dropna()
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    le = LabelEncoder()
    
    # If classification, ensure target is discrete
    if is_classification:
        # fit_transform handles converting strings, mixed types, and even floats (1.0, 2.0) 
        # into discrete integer classes (0, 1, ...). This prevents "Unknown label type: continuous".
        y = le.fit_transform(y)
    elif y.dtype == 'object':
        # For regression, only encode if it's an object (string)
        y = le.fit_transform(y)
    
    # Encode features if they are object type
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])
        
    return train_test_split(X, y, test_size=test_size, random_state=42)


@ml_bp.route('/ml/knn', methods=['POST'])
def run_knn():
    """K-Nearest Neighbors classification endpoint.
    
    Expected JSON:
        - path: CSV file path
        - target: Target column name
        - max_k: Maximum k value to test (default 20)
    """
    data = request.json
    path = data.get('path')
    target = data.get('target')
    max_k = int(data.get('max_k', 20))
    
    if not path or not target:
        return jsonify({'error': 'Path and target column required'}), 400
        
    try:
        df = pd.read_csv(path)
        X_train, X_test, y_train, y_test = prepare_data(df, target, is_classification=True)
        
        # Ensure max_k doesn't exceed training samples
        max_k = min(max_k, len(X_train))
        
        # 1. Run optimization (Accuracy vs k)
        accuracies = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            accuracies.append(knn.score(X_test, y_test))
            
        # Plot Accuracy vs K
        fig = plt.figure(figsize=(10, 6))
        plt.plot(k_range, accuracies, marker='o')
        plt.title('K-NN: Accuracy vs K')
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'knn_accuracy')
        plt.close(fig)
        
        # 2. Train final model with best K
        best_k = k_range[np.argmax(accuracies)]
        final_model = KNeighborsClassifier(n_neighbors=best_k)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        
        # Metrics
        metrics = {
            'best_k': int(best_k),
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }

        # Save to global state for comparison
        state.ml_results['knn'] = metrics
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'plot_url': plot_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/ml/naive-bayes', methods=['POST'])
def run_naive_bayes():
    """Naive Bayes classification endpoint.
    
    Expected JSON:
        - path: CSV file path
        - target: Target column name
    """
    data = request.json
    path = data.get('path')
    target = data.get('target')
    
    if not path or not target:
        return jsonify({'error': 'Path and target column required'}), 400
    
    try:
        df = pd.read_csv(path)
        X_train, X_test, y_train, y_test = prepare_data(df, target, is_classification=True)
        
        # Gaussian Naive Bayes (Standard choice for continuous features)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        
        # Save to global state for comparison
        state.ml_results['naive_bayes'] = metrics
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/ml/decision-tree', methods=['POST'])
def run_decision_tree():
    """Decision Tree classification endpoint.
    
    Expected JSON:
        - path: CSV file path
        - target: Target column name
        - algorithm_type: 'id3', 'c4.5', or 'cart' (default 'cart')
    """
    data = request.json
    path = data.get('path')
    target = data.get('target')
    algo_type = data.get('algorithm_type', 'cart').lower()
    
    if not path or not target:
        return jsonify({'error': 'Path and target column required'}), 400
    
    try:
        df = pd.read_csv(path)
        X_train, X_test, y_train, y_test = prepare_data(df, target, is_classification=True)
        
        # Map user selection to sklearn parameters
        # ID3/C4.5 use Information Gain -> criterion='entropy'
        # CART uses Gini Index -> criterion='gini'
        criterion = 'entropy' if algo_type in ['id3', 'c4.5'] else 'gini'
        
        dt = DecisionTreeClassifier(criterion=criterion, random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot Confusion Matrix
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({algo_type.upper()})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'dt_confusion_matrix')
        plt.close(fig)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'criterion_used': criterion,
            'matrix_plot_url': plot_url
        }
        
        state.ml_results['decision_tree'] = metrics
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/ml/linear-regression', methods=['POST'])
def run_linear_regression():
    """Linear Regression endpoint.
    
    Expected JSON:
        - path: CSV file path
        - target: Target column name (must be numeric)
    """
    data = request.json
    path = data.get('path')
    target = data.get('target')
    
    if not path or not target:
        return jsonify({'error': 'Path and target column required'}), 400
    
    try:
        df = pd.read_csv(path)
        
        # Linear regression requires numerical target
        if not pd.api.types.is_numeric_dtype(df[target]):
            return jsonify({'error': 'Target column must be numerical for Linear Regression'}), 400
            
        X_train, X_test, y_train, y_test = prepare_data(df, target, is_classification=False)
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2_score': float(r2_score(y_test, y_pred)),
            'coefficients': lr.coef_.tolist(),
            'intercept': float(lr.intercept_)
        }
        
        state.ml_results['linear_regression'] = metrics
        
        # Plot Actual vs Predicted
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Linear Regression: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'linreg_fit')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'plot_url': plot_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/ml/neural-network', methods=['POST'])
def run_neural_network():
    """Neural Network classification endpoint.
    
    Expected JSON:
        - path: CSV file path
        - target: Target column name
        - hidden_layers: Hidden layer configuration as int, string, or list (default '(100,)')
    """
    data = request.json
    path = data.get('path')
    target = data.get('target')
    hidden_layer_sizes = data.get('hidden_layers', '(100,)')
    
    if not path or not target:
        return jsonify({'error': 'Path and target column required'}), 400
    
    try:
        # Convert simple integer input to tuple for MLP
        if isinstance(hidden_layer_sizes, int):
            hidden_layers = (hidden_layer_sizes,)
        elif isinstance(hidden_layer_sizes, str):
            # Basic parsing for tuple string like "(5, 2)" or "100, 50"
            try:
                sanitized = hidden_layer_sizes.strip('()[] ')
                if ',' in sanitized:
                    hidden_layers = tuple(int(x.strip()) for x in sanitized.split(',') if x.strip())
                else:
                    hidden_layers = (int(sanitized),)
            except:
                hidden_layers = (100,)  # default fallback
        else:
            # Assume it is a list from JSON
            hidden_layers = tuple(hidden_layer_sizes)

        df = pd.read_csv(path)
        X_train, X_test, y_train, y_test = prepare_data(df, target, is_classification=True)
        
        # Scale data for Neural Network (important for convergence)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)
        
        # Plot Loss Curve
        fig = plt.figure(figsize=(10, 6))
        plt.plot(mlp.loss_curve_)
        plt.title('Neural Network Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        loss_plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'nn_loss')
        plt.close(fig)
        
        # ROC Curve (only for binary classification)
        roc_plot_url = None
        auc_score = None
        
        if len(np.unique(y_test)) == 2:
            y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = float(auc(fpr, tpr))
            
            fig = plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            roc_plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'nn_roc')
            plt.close(fig)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'iterations': int(mlp.n_iter_),
            'auc': auc_score
        }
        
        state.ml_results['neural_network'] = metrics
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'loss_plot_url': loss_plot_url,
            'roc_plot_url': roc_plot_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/ml/comparison', methods=['GET'])
def get_ml_comparison():
    """Get comparison of all ML algorithms run so far."""
    if not state.ml_results:
        return jsonify({'error': 'No ML results available yet'}), 400
    
    try:
        # Generate comparison plot for Accuracy (common metric)
        algorithms = []
        accuracies = []
        
        for algo, metrics in state.ml_results.items():
            if 'accuracy' in metrics:
                algorithms.append(algo.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
            elif 'r2_score' in metrics:  # For regression
                algorithms.append(algo.replace('_', ' ').title())
                accuracies.append(metrics['r2_score'])  # Use R2 as "accuracy" analog
        
        plot_url = None
        if algorithms:
            fig = plt.figure(figsize=(10, 6))
            bars = plt.bar(algorithms, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.xlabel('Algorithm')
            plt.ylabel('Score (Accuracy / R²)')
            plt.title('ML Algorithm Comparison')
            plt.ylim(0, 1.05)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            for i, (bar, v) in enumerate(zip(bars, accuracies)):
                plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, 
                        f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'ml_comparison')
            plt.close(fig)
        
        return jsonify({
            'success': True,
            'comparison': state.ml_results,
            'plot_url': plot_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
