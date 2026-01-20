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

def prepare_data(df, target_col, test_size=0.2, is_classification=False):
    """Helper to split data and encode target if necessary."""
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
    data = request.json
    path = data.get('path')
    target = data.get('target')
    max_k = int(data.get('max_k', 2))
    
    if not path or not target:
        return jsonify({'error': 'Path and target column required'}), 400
        
    try:
        df = pd.read_csv(path)
        X_train, X_test, y_train, y_test = prepare_data(df, target, is_classification=True)
        
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
    data = request.json
    path = data.get('path')
    target = data.get('target')
    
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
    data = request.json
    path = data.get('path')
    target = data.get('target')
    # Algorithm choice: 'id3', 'c4.5', 'cart' (managed via criterion)
    algo_type = data.get('algorithm_type', 'cart').lower()
    
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
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
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
    data = request.json
    path = data.get('path')
    target = data.get('target')
    
    try:
        df = pd.read_csv(path)
        
        # Linear regression requires numerical target
        if not pd.api.types.is_numeric_dtype(df[target]):
            return jsonify({'error': 'Target column must be numerical for Linear Regression'}), 400
            
        X_train, X_test, y_train, y_test = prepare_data(df, target)
        
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
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'linreg_fit')
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'plot_url': plot_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ml_bp.route('/ml/neural-network', methods=['POST'])
def run_neural_network():
    data = request.json
    path = data.get('path')
    target = data.get('target')
    hidden_layer_sizes = data.get('hidden_layers', '(100,)') # tuple as string or list
    
    try:
        # Convert simple integer input to tuple for MLP
        if isinstance(hidden_layer_sizes, int):
             hidden_layers = (hidden_layer_sizes,)
        elif isinstance(hidden_layer_sizes, str):
             # safe eval or parsing logic
             try:
                # Basic parsing for tuple string like "(5, 2)" or "100, 50"
                sanitized = hidden_layer_sizes.strip('()[] ')
                if ',' in sanitized:
                    hidden_layers = tuple(int(x.strip()) for x in sanitized.split(',') if x.strip())
                else:
                    hidden_layers = (int(sanitized),)
             except:
                hidden_layers = (100,) # default fallback
        else:
             # Assume it is a list from JSON
             hidden_layers = tuple(hidden_layer_sizes)

        df = pd.read_csv(path)
        X_train, X_test, y_train, y_test = prepare_data(df, target, is_classification=True)
        
        # Scale data for Neural Network (important for convergence)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        
        # Plot Loss Curve
        fig = plt.figure(figsize=(10, 6))
        plt.plot(mlp.loss_curve_)
        plt.title('Neural Network Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'nn_loss')
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'iterations': int(mlp.n_iter_)
        }
        
        state.ml_results['neural_network'] = metrics
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'plot_url': plot_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ml_bp.route('/ml/comparison', methods=['GET'])
def get_ml_comparison():
    if not state.ml_results:
        return jsonify({'error': 'No ML results available yet'}), 400
    return jsonify(state.ml_results)
