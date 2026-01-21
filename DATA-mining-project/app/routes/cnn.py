from flask import Blueprint, request, jsonify, current_app
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from app.utils.helpers import save_plot
from app.state import state
import io


cnn_bp = Blueprint('cnn', __name__)


# Dataset configurations
DATASETS = {
    'digits': {
        'name': 'Digits (Scikit-Learn)',
        'shape': (8, 8, 1),
        'num_classes': 10,
        'classes': list(range(10)),
        'description': 'Handwritten digits 8×8 grayscale images (1797 samples)',
        'load_function': 'load_digits'
    },
    'mnist': {
        'name': 'MNIST',
        'shape': (28, 28, 1),
        'num_classes': 10,
        'classes': list(range(10)),
        'description': 'Handwritten digits 28×28 grayscale images (70,000 samples)',
        'load_function': 'mnist'
    },
    'fashion_mnist': {
        'name': 'Fashion-MNIST',
        'shape': (28, 28, 1),
        'num_classes': 10,
        'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'description': 'Fashion products 28×28 grayscale images (70,000 samples)',
        'load_function': 'fashion_mnist'
    },
    'cifar10': {
        'name': 'CIFAR-10',
        'shape': (32, 32, 3),
        'num_classes': 10,
        'classes': ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],
        'description': 'Color images 32×32 RGB (60,000 samples)',
        'load_function': 'cifar10'
    }
}


def load_dataset(dataset_name='digits', test_size=0.2):
    """
    Load and preprocess the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('digits', 'mnist', 'fashion_mnist', 'cifar10')
        test_size: Test split ratio (only for digits dataset)
    
    Returns:
        X_train, X_test, y_train, y_test, dataset_info
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")
    
    dataset_info = DATASETS[dataset_name]
    
    if dataset_name == 'digits':
        # Load Scikit-Learn Digits
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        # Normalize
        X = X / 16.0
        
        # Reshape
        X = X.reshape(-1, 8, 8, 1)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
    elif dataset_name == 'mnist':
        # Load MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape to add channel dimension
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
    elif dataset_name == 'fashion_mnist':
        # Load Fashion-MNIST
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        
        # Normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape to add channel dimension
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
    elif dataset_name == 'cifar10':
        # Load CIFAR-10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    
    return X_train, X_test, y_train, y_test, dataset_info


def build_cnn_model(input_shape, num_classes, dataset_name='digits'):
    """
    Build a CNN model adapted to the dataset.
    
    Architecture varies based on input size:
    - Small (8×8): Simple CNN
    - Medium (28×28): Standard CNN
    - Large (32×32 RGB): Deeper CNN
    """
    model = Sequential()
    
    if dataset_name == 'digits':
        # Simple CNN for 8×8 images
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif dataset_name in ['mnist', 'fashion_mnist']:
        # Standard CNN for 28×28 images
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif dataset_name == 'cifar10':
        # Deeper CNN for 32×32 RGB images
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
    
    return model


@cnn_bp.route('/cnn/datasets', methods=['GET'])
def get_available_datasets():
    """Get list of available datasets."""
    return jsonify({
        'success': True,
        'datasets': DATASETS
    })


@cnn_bp.route('/cnn/info', methods=['POST'])
def get_dataset_info():
    """Get detailed information about a specific dataset."""
    try:
        data = request.json or {}
        dataset_name = data.get('dataset', 'digits').lower()
        
        if dataset_name not in DATASETS:
            return jsonify({'error': f'Unknown dataset: {dataset_name}'}), 400
        
        dataset_info = DATASETS[dataset_name].copy()
        
        # Optimize for large datasets
        if dataset_name in ['cifar10', 'fashion_mnist', 'mnist']:
             # Use predefined info to avoid loading full dataset just for metadata
             dataset_info.update({
                'train_samples': 60000 if dataset_name == 'mnist' or dataset_name == 'fashion_mnist' else 50000,
                'test_samples': 10000,
                'total_samples': 70000 if dataset_name == 'mnist' or dataset_name == 'fashion_mnist' else 60000,
                'image_shape': list(dataset_info['shape'])
             })
             # Placeholder distribution for performance
             dataset_info['train_distribution'] = {c: "Unknown" for c in dataset_info['classes']}
             dataset_info['test_distribution'] = {c: "Unknown" for c in dataset_info['classes']}
        else:
            # Load dataset to get actual counts
            X_train, X_test, y_train, y_test, _ = load_dataset(dataset_name)
            
            # Class distribution
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            
            class_names = dataset_info['classes']
            train_distribution = {
                str(class_names[int(k)]): int(v) 
                for k, v in zip(unique_train, counts_train)
            }
            test_distribution = {
                str(class_names[int(k)]): int(v) 
                for k, v in zip(unique_test, counts_test)
            }
            
            dataset_info.update({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X_train) + len(X_test),
                'train_distribution': train_distribution,
                'test_distribution': test_distribution,
                'image_shape': list(X_train.shape[1:])
            })
        
        return jsonify({
            'success': True,
            'info': dataset_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cnn_bp.route('/cnn/visualize-samples', methods=['POST'])
def visualize_samples():
    """Visualize sample images from the selected dataset."""
    try:
        data = request.json or {}
        dataset_name = data.get('dataset', 'digits').lower()
        num_samples = int(data.get('num_samples', 10))
        
        # Load dataset
        X_train, X_test, y_train, y_test, dataset_info = load_dataset(dataset_name)
        
        # Combine train and test for sampling
        X_all = np.concatenate([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        # Select random samples
        indices = np.random.choice(len(X_all), num_samples, replace=False)
        
        class_names = dataset_info['classes']
        
        for idx, ax in enumerate(axes):
            if idx < num_samples:
                image = X_all[indices[idx]]
                label = y_all[indices[idx]]
                label_name = class_names[int(label)]
                
                # Handle different image formats
                if image.shape[-1] == 1:  # Grayscale
                    ax.imshow(image.squeeze(), cmap='gray')
                else:  # RGB
                    ax.imshow(image)
                
                ax.set_title(f'{label_name}')
                ax.axis('off')
        
        plt.suptitle(f'{dataset_info["name"]} - Sample Images', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cnn_samples')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'dataset': dataset_name,
            'num_samples': num_samples
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cnn_bp.route('/cnn/train', methods=['POST'])
def train_cnn():
    """
    Train a CNN model on the selected dataset.
    
    Expected JSON:
        - dataset: Dataset name ('digits', 'mnist', 'fashion_mnist', 'cifar10')
        - epochs: Number of training epochs (default: 10)
        - batch_size: Batch size for training (default: 32)
        - test_size: Test split ratio for digits dataset (default: 0.2)
    """
    try:
        data = request.json or {}
        dataset_name = data.get('dataset', 'digits').lower()
        epochs = int(data.get('epochs', 10))
        batch_size = int(data.get('batch_size', 32))
        test_size = float(data.get('test_size', 0.2))
        
        if dataset_name not in DATASETS:
            return jsonify({'error': f'Unknown dataset: {dataset_name}'}), 400
        
        print(f"Training CNN on {dataset_name} with epochs={epochs}, batch_size={batch_size}")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, dataset_info = load_dataset(dataset_name, test_size)
        
        print(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Build model
        model = build_cnn_model(
            input_shape=dataset_info['shape'],
            num_classes=dataset_info['num_classes'],
            dataset_name=dataset_name
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Make predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
        ax1.plot(history.history['val_accuracy'], label='Test Accuracy', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'CNN Training History - {dataset_info["name"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss', marker='o')
        ax2.plot(history.history['val_loss'], label='Test Loss', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss over Epochs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cnn_history')
        plt.close(fig)
        
        # Plot confusion matrix
        fig = plt.figure(figsize=(12, 10))
        class_names = dataset_info['classes']
        
        # For better visualization with many classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {dataset_info["name"]} (Accuracy: {accuracy:.2%})')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cnn_confusion_matrix')
        plt.close(fig)
        
        # Get model summary
        summary_io = io.StringIO()
        model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        model_summary = summary_io.getvalue()
        
        # Per-class accuracy
        per_class_accuracy = {}
        for i in range(dataset_info['num_classes']):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = accuracy_score(y_test[mask], y_pred[mask])
                per_class_accuracy[str(class_names[i])] = float(class_acc)
        
        # Prepare metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1]),
            'train_loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'epochs_trained': len(history.history['accuracy']),
            'total_params': int(model.count_params()),
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': per_class_accuracy
        }
        
        # Save to state
        state.ml_results[f'cnn_{dataset_name}'] = metrics
        
        # Store model in state
        state.cnn_model = {
            'model': model,
            'dataset': dataset_name,
            'dataset_info': dataset_info,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'history': history.history
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'model_summary': model_summary,
            'history_plot_url': history_plot_url,
            'confusion_matrix_plot_url': cm_plot_url,
            'training_params': {
                'dataset': dataset_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'test_size': test_size if dataset_name == 'digits' else 'preset',
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@cnn_bp.route('/cnn/predictions', methods=['GET'])
def get_predictions():
    """Get sample predictions from the trained model."""
    try:
        if not hasattr(state, 'cnn_model') or state.cnn_model is None:
            return jsonify({'error': 'No trained CNN model found. Train a model first.'}), 400
        
        model_data = state.cnn_model
        model = model_data['model']
        dataset_info = model_data['dataset_info']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        # Get 10 random samples
        num_samples = min(10, len(X_test))
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes = axes.ravel()
        
        predictions_info = []
        class_names = dataset_info['classes']
        
        for idx, ax in enumerate(axes):
            if idx < num_samples:
                i = indices[idx]
                image = X_test[i]
                true_label = int(y_test[i])
                pred_label = int(y_pred[i])
                
                # Get prediction probabilities
                probs = model.predict(X_test[i:i+1], verbose=0)[0]
                confidence = float(probs[pred_label])
                
                # Plot
                if image.shape[-1] == 1:  # Grayscale
                    ax.imshow(image.squeeze(), cmap='gray')
                else:  # RGB
                    ax.imshow(image)
                
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f'True: {class_names[true_label]}\n'
                           f'Pred: {class_names[pred_label]}\n'
                           f'Conf: {confidence:.2%}', 
                           color=color, fontsize=9)
                ax.axis('off')
                
                predictions_info.append({
                    'index': int(i),
                    'true_label': str(class_names[true_label]),
                    'predicted_label': str(class_names[pred_label]),
                    'confidence': confidence,
                    'correct': bool(true_label == pred_label)
                })
        
        plt.suptitle(f'{dataset_info["name"]} - Sample Predictions', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cnn_predictions')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'dataset': model_data['dataset'],
            'predictions': predictions_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cnn_bp.route('/cnn/error-analysis', methods=['GET'])
def error_analysis():
    """Analyze misclassified samples."""
    try:
        if not hasattr(state, 'cnn_model') or state.cnn_model is None:
            return jsonify({'error': 'No trained CNN model found. Train a model first.'}), 400
        
        model_data = state.cnn_model
        model = model_data['model']
        dataset_info = model_data['dataset_info']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        # Find misclassified samples
        errors = np.where(y_test != y_pred)[0]
        
        if len(errors) == 0:
            return jsonify({
                'success': True,
                'message': 'Perfect accuracy! No errors to analyze.',
                'num_errors': 0
            })
        
        # Select up to 10 errors
        num_errors = min(10, len(errors))
        error_indices = errors[:num_errors]
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes = axes.ravel()
        
        error_info = []
        class_names = dataset_info['classes']
        
        for idx, ax in enumerate(axes):
            if idx < num_errors:
                i = error_indices[idx]
                image = X_test[i]
                true_label = int(y_test[i])
                pred_label = int(y_pred[i])
                
                # Get prediction probabilities
                probs = model.predict(X_test[i:i+1], verbose=0)[0]
                confidence = float(probs[pred_label])
                
                # Plot
                if image.shape[-1] == 1:  # Grayscale
                    ax.imshow(image.squeeze(), cmap='gray')
                else:  # RGB
                    ax.imshow(image)
                
                ax.set_title(f'True: {class_names[true_label]}\n'
                           f'Pred: {class_names[pred_label]}\n'
                           f'Conf: {confidence:.2%}', 
                           color='red', fontsize=9)
                ax.axis('off')
                
                # Top 3 predictions
                top3_indices = np.argsort(probs)[-3:][::-1]
                error_info.append({
                    'index': int(i),
                    'true_label': str(class_names[true_label]),
                    'predicted_label': str(class_names[pred_label]),
                    'confidence': confidence,
                    'top3_predictions': [
                        {
                            'class': str(class_names[j]),
                            'probability': float(probs[j])
                        }
                        for j in top3_indices
                    ]
                })
            else:
                ax.axis('off')
        
        plt.suptitle(f'Misclassified Samples - {dataset_info["name"]} '
                     f'(Total Errors: {len(errors)}/{len(y_test)})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cnn_errors')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'dataset': model_data['dataset'],
            'num_errors': len(errors),
            'total_samples': len(y_test),
            'error_rate': float(len(errors) / len(y_test)),
            'errors': error_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cnn_bp.route('/cnn/architecture', methods=['POST'])
def get_architecture():
    """Get visual representation of CNN architecture for a dataset."""
    try:
        data = request.json or {}
        dataset_name = data.get('dataset', 'digits').lower()
        
        if dataset_name not in DATASETS:
            return jsonify({'error': f'Unknown dataset: {dataset_name}'}), 400
        
        dataset_info = DATASETS[dataset_name]
        
        # Build model
        model = build_cnn_model(
            input_shape=dataset_info['shape'],
            num_classes=dataset_info['num_classes'],
            dataset_name=dataset_name
        )
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Get text summary
        summary_io = io.StringIO()
        model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        model_summary = summary_io.getvalue()
        
        # Layer information
        layers_info = []
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output.shape) if hasattr(layer, 'output') else str(getattr(layer, 'output_shape', 'Unknown')),
                'params': int(layer.count_params())
            }
            
            if hasattr(layer, 'filters'):
                layer_info['filters'] = int(layer.filters)
            if hasattr(layer, 'kernel_size'):
                layer_info['kernel_size'] = layer.kernel_size
            if hasattr(layer, 'pool_size'):
                layer_info['pool_size'] = layer.pool_size
            if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
                layer_info['activation'] = layer.activation.__name__
            if hasattr(layer, 'rate'):
                layer_info['dropout_rate'] = float(layer.rate)
                
            layers_info.append(layer_info)
        
        return jsonify({
            'success': True,
            'dataset': dataset_name,
            'model_summary': model_summary,
            'layers': layers_info,
            'total_params': int(model.count_params()),
            'input_shape': list(dataset_info['shape']),
            'num_classes': dataset_info['num_classes']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cnn_bp.route('/cnn/compare-datasets', methods=['GET'])
def compare_datasets():
    """Compare results across different datasets."""
    try:
        # Get all CNN results from state
        cnn_results = {k: v for k, v in state.ml_results.items() if k.startswith('cnn_')}
        
        if not cnn_results:
            return jsonify({'error': 'No CNN results available. Train models first.'}), 400
        
        # Extract dataset names and accuracies
        datasets = []
        accuracies = []
        params = []
        
        for key, metrics in cnn_results.items():
            dataset_name = key.replace('cnn_', '')
            datasets.append(DATASETS[dataset_name]['name'])
            accuracies.append(metrics['accuracy'])
            params.append(metrics['total_params'])
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        bars1 = ax1.bar(datasets, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('CNN Accuracy Comparison')
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Model complexity comparison
        bars2 = ax2.bar(datasets, params, color='coral', edgecolor='darkred', alpha=0.7)
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Model Complexity Comparison')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, param in zip(bars2, params):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{param:,}', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'cnn_dataset_comparison')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'comparison': cnn_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
