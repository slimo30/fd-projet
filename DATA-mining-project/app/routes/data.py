from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
import os
import uuid
import arff
from werkzeug.utils import secure_filename
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.state import state
from app.utils.helpers import is_safe_path, convert_to_native, save_plot

data_bp = Blueprint('data', __name__)

@data_bp.route('/create-sample', methods=['POST'])
def create_sample():
    try:
        # Get the clean option from request (default is False for backward compatibility)
        request_data = request.get_json() or {}
        create_clean = request_data.get('clean', False)
        
        if create_clean:
            # Create clean, numeric, normalized dataset
            np.random.seed(42)  # For reproducibility
            
            # Generate normalized numeric data
            data = {
                'feature_1': np.random.uniform(0, 1, 100),  # Already normalized [0, 1]
                'feature_2': np.random.uniform(0, 1, 100),
                'feature_3': np.random.uniform(0, 1, 100),
                'feature_4': np.random.uniform(0, 1, 100),
                'feature_5': np.random.uniform(0, 1, 100),
                'target': np.random.randint(0, 2, 100)  # Binary target (0 or 1)
            }
            
            df = pd.DataFrame(data)
            # Round to 4 decimal places for cleaner display
            for col in df.columns:
                if col != 'target':
                    df[col] = df[col].round(4)
        else:
            # Create raw data with categorical values and missing data
            data = {
                'id': range(1, 101),
                'category': ['A', 'B', 'C', 'D', 'E'] * 20,
                'value': [float(i) for i in range(1, 101)],
                'status': ['active', 'inactive', 'pending', 'unknown'] * 25,
                'rating': [i % 5 + 1 for i in range(100)]
            }
            
            df = pd.DataFrame(data)
            
            # Add missing values randomly (~10%)
            for col in df.columns:
                df.loc[df.sample(frac=0.1).index, col] = np.nan
        
        # Save to file
        dataset_type = "clean" if create_clean else "raw"
        filename = f"sample_{dataset_type}_{uuid.uuid4()}.csv"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        df.to_csv(filepath, index=False)
        
        state.df = df
        state.current_filepath = filepath
        
        return jsonify({
            'message': f'{"Clean" if create_clean else "Raw"} sample data created',
            'columns': df.columns.tolist(),
            'head': df.head().to_dict(orient='records'),
            'path': filepath,
            'is_clean': create_clean
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/upload-data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    # create a unique copy
    new_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")

    try:
        if file.filename.lower().endswith('.arff'):
            content = file.read().decode('utf-8')
            data = arff.loads(content)
            df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
        else:
            file.seek(0)
            df = pd.read_csv(file)
        
        df.to_csv(filepath, index=False)
        state.df = df
        state.current_filepath = filepath
        
        return jsonify({
            'message': 'Data uploaded',
            'columns': df.columns.tolist(),
            'head': df.head().to_dict(orient='records'),
            'path': filepath
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/data/head', methods=['GET'])
def get_data_head():
    path = request.args.get('path')
    limit = request.args.get('limit', type=int, default=40)
    
    if not path or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid path'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Get representative sample using stratified approach
    total_rows = len(state.df)
    
    if total_rows <= limit:
        # If dataset is smaller than limit, return all rows with index
        df_with_index = state.df.copy()
        df_with_index.insert(0, '__row_index__', range(len(state.df)))
        return jsonify(df_with_index.to_dict(orient='records'))
    
    # Smart sampling strategy: evenly distributed rows across the dataset
    # This ensures we capture patterns throughout the entire dataset
    indices = np.linspace(0, total_rows - 1, limit, dtype=int)
    df_sample = state.df.iloc[indices].copy()
    
    # Add the original row index as the first column
    df_sample.insert(0, '__row_index__', indices)
    
    return jsonify(df_sample.to_dict(orient='records'))

@data_bp.route('/data/columns', methods=['GET'])
def get_data_columns():
    path = request.args.get('path')
    if not path or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid path'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'columns': state.df.columns.tolist()})

@data_bp.route('/data/select-columns', methods=['POST'])
def select_columns():
    data = request.get_json()
    path = data.get('path')
    columns = data.get('columns')
    
    if not path or not columns or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid request'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        missing = [col for col in columns if col not in state.df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400
        
        state.df = state.df[columns]
        state.df.to_csv(state.current_filepath, index=False)
        
        return jsonify({'message': 'Columns selected', 'selected': columns})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/data/statistics', methods=['GET'])
def get_statistics():
    path = request.args.get('path')
    if not path or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid path'}), 400

    full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], os.path.basename(path))

    if full_path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(full_path)
            state.current_filepath = full_path
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    try:
        stats = {}
        for col in state.df.columns:
            col_stats = {}
            col_stats['missing'] = int(state.df[col].isna().sum())

            if pd.api.types.is_numeric_dtype(state.df[col]):
                col_stats['mean'] = convert_to_native(state.df[col].mean())
                col_stats['median'] = convert_to_native(state.df[col].median())
            else:
                col_stats['mean'] = None
                col_stats['median'] = None

            mode = state.df[col].mode()
            col_stats['mode'] = convert_to_native(mode.iloc[0] if not mode.empty else None)

            stats[col] = col_stats

        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/data/fill-missing', methods=['POST'])
def fill_missing():
    data = request.get_json()
    path = data.get('path')
    fill_strategies = data.get('fill')
    
    if not path or not fill_strategies or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid request'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        for col, strategy in fill_strategies.items():
            if col not in state.df.columns:
                return jsonify({'error': f'Column {col} not found'}), 400
            
            # Parse strategy - it might be "knn:5" format
            if isinstance(strategy, str) and strategy.startswith('knn:'):
                # Extract K value from "knn:5" format
                try:
                    k_value = int(strategy.split(':')[1])
                except (IndexError, ValueError):
                    k_value = 5  # Default fallback
                
                # Try to convert to numeric if it's not already (handles encoded categorical)
                if not pd.api.types.is_numeric_dtype(state.df[col]):
                    try:
                        state.df[col] = pd.to_numeric(state.df[col], errors='coerce')
                    except:
                        return jsonify({'error': f'KNN imputation only applicable for numeric columns: {col}'}), 400
                
                # Check if we have valid numeric data after conversion
                if state.df[col].notna().sum() == 0:
                    return jsonify({'error': f'Column {col} has no valid numeric values for KNN imputation'}), 400
                
                # Use KNN imputer with specified K
                imputer = KNNImputer(n_neighbors=k_value)
                state.df[col] = imputer.fit_transform(state.df[[col]])
                
            elif strategy == 'mean':
                # Try to convert to numeric if needed
                if not pd.api.types.is_numeric_dtype(state.df[col]):
                    try:
                        state.df[col] = pd.to_numeric(state.df[col], errors='coerce')
                    except:
                        return jsonify({'error': f'Mean not applicable for {col}'}), 400
                state.df[col].fillna(state.df[col].mean(), inplace=True)
            elif strategy == 'median':
                # Try to convert to numeric if needed
                if not pd.api.types.is_numeric_dtype(state.df[col]):
                    try:
                        state.df[col] = pd.to_numeric(state.df[col], errors='coerce')
                    except:
                        return jsonify({'error': f'Median not applicable for {col}'}), 400
                state.df[col].fillna(state.df[col].median(), inplace=True)
            elif strategy == 'mode':
                mode_val = state.df[col].mode()[0] if not state.df[col].mode().empty else None
                state.df[col].fillna(mode_val, inplace=True)
            else:
                return jsonify({'error': f'Invalid strategy: {strategy}'}), 400
        
        state.df.to_csv(state.current_filepath, index=False)
        return jsonify({'message': 'Missing values filled'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/data/normalize', methods=['POST'])
def normalize_data():
    data = request.get_json()
    path = data.get('path')
    method = data.get('method')
    columns = data.get('columns')
    
    if not path or not method or not columns or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid request'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        for col in columns:
            if col not in state.df.columns:
                return jsonify({'error': f'Column {col} not found'}), 400
            if not pd.api.types.is_numeric_dtype(state.df[col]):
                return jsonify({'error': f'Non-numeric column: {col}'}), 400
            
            if method == 'zscore':
                mean, std = state.df[col].mean(), state.df[col].std()
                if std == 0:
                    state.df[col] = 0.0
                else:
                    state.df[col] = (state.df[col] - mean) / std
            elif method == 'minmax':
                min_val, max_val = state.df[col].min(), state.df[col].max()
                if max_val == min_val:
                    state.df[col] = 0.0
                else:
                    state.df[col] = (state.df[col] - min_val) / (max_val - min_val)
            else:
                return jsonify({'error': 'Invalid method'}), 400
        
        state.df.to_csv(state.current_filepath, index=False)
        return jsonify({'message': 'Data normalized'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/data/categorical-to-numerical', methods=['POST'])
def categorical_to_numerical():
    data = request.get_json()
    path = data.get('path')
    columns = data.get('columns')
    method = data.get('method', 'label')
    
    if not path or not columns or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid request'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        missing_columns = [col for col in columns if col not in state.df.columns]
        if missing_columns:
            return jsonify({'error': f'Columns not found: {missing_columns}'}), 400
        
        result = {'transformed_columns': {}}
        
        if method == 'label':
            for col in columns:
                if not pd.api.types.is_numeric_dtype(state.df[col]):
                    # Get unique categories excluding NaN
                    categories = state.df[col].dropna().unique()
                    mapping = {category: idx for idx, category in enumerate(categories)}
                    # Map categories to numbers, keep NaN as NaN (don't fill with -1)
                    state.df[col] = state.df[col].map(mapping)
                    result['transformed_columns'][col] = {'method': 'label'}
        
        elif method == 'onehot':
             # One-hot encoding with dummy_na=False to exclude NaN from encoding
             # NaN values will remain as NaN in all dummy columns
             state.df = pd.get_dummies(state.df, columns=columns, dummy_na=False)
             
        state.df.to_csv(state.current_filepath, index=False)
        return jsonify({'message': 'Categorical columns converted', 'details': result})
            
    except Exception as e:
         return jsonify({'error': str(e)}), 500

@data_bp.route('/data/save', methods=['POST'])
def save_data():
    data = request.get_json()
    new_filepath = data.get('new_path')
    
    if not new_filepath or not is_safe_path(current_app.config['UPLOAD_FOLDER'], new_filepath):
        return jsonify({'error': 'Invalid path'}), 400
    
    try:
        if state.df is not None:
            state.df.to_csv(new_filepath, index=False)
            return jsonify({'message': 'Data saved successfully', 'path': new_filepath})
        return jsonify({'error': 'No data to save'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/data/knn-optimization', methods=['POST'])
def knn_optimization():
    """
    Generate a plot showing optimal K for KNN imputer by testing different K values
    """
    data = request.get_json()
    path = data.get('path')
    column = data.get('column')
    max_k = data.get('max_k', 20)
    
    if not path or not column or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid request'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        if column not in state.df.columns:
            return jsonify({'error': f'Column {column} not found'}), 400
        
        # Try to convert to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(state.df[column]):
            try:
                # Attempt to convert to numeric (handles strings that are numbers)
                state.df[column] = pd.to_numeric(state.df[column], errors='coerce')
            except:
                return jsonify({'error': f'Column "{column}" cannot be converted to numeric. KNN optimization only works for numeric columns.'}), 400
        
        # Get the column data
        col_data = state.df[[column]].copy()
        
        # Check if there are missing values or all values are NaN after conversion
        if col_data[column].isna().all():
            return jsonify({'error': f'Column "{column}" has no valid numeric values'}), 400
            
        if col_data[column].isna().sum() == 0:
            return jsonify({'error': 'No missing values found in the selected column'}), 400
        
        # Create a copy with complete cases only for validation
        complete_data = col_data.dropna()
        
        if len(complete_data) < max_k:
            max_k = len(complete_data) - 1
            if max_k < 1:
                return jsonify({'error': 'Not enough complete data for KNN optimization'}), 400
        
        # Simulate missing values in complete data to measure error
        # We'll randomly mask 20% of complete values and try to predict them
        np.random.seed(42)
        test_size = max(1, int(len(complete_data) * 0.2))
        
        # Reset index to ensure continuous indexing
        complete_data_reset = complete_data.reset_index(drop=True)
        
        # Select test indices from the reset data
        test_indices = np.random.choice(len(complete_data_reset), size=test_size, replace=False)
        
        # Create test data with artificial missing values
        test_data = complete_data_reset.copy()
        true_values = test_data.loc[test_indices, column].copy().values
        test_data.loc[test_indices, column] = np.nan
        
        # Test different K values
        k_values = range(1, min(max_k + 1, len(complete_data_reset)))
        errors = []
        
        for k in k_values:
            imputer = KNNImputer(n_neighbors=k)
            imputed_data = imputer.fit_transform(test_data)
            predicted_values = imputed_data[test_indices, 0]
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
            errors.append(rmse)
        
        # Find optimal K (minimum error)
        optimal_k = k_values[np.argmin(errors)]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, errors, marker='o', linewidth=2, markersize=6, color='#3b82f6')
        ax.axvline(x=optimal_k, color='#10b981', linestyle='--', linewidth=2, 
                   label=f'Optimal K = {optimal_k}')
        ax.scatter([optimal_k], [errors[optimal_k - 1]], color='#10b981', s=200, 
                   zorder=5, edgecolors='white', linewidth=2)
        
        ax.set_xlabel('Number of Neighbors (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (Root Mean Square Error)', fontsize=12, fontweight='bold')
        ax.set_title(f'KNN Imputer Optimization for "{column}"', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best')
        
        # Add annotation for optimal point
        ax.annotate(f'Best K: {optimal_k}\nRMSE: {errors[optimal_k - 1]:.4f}',
                   xy=(optimal_k, errors[optimal_k - 1]),
                   xytext=(10, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#10b981', alpha=0.7, edgecolor='white'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#10b981', lw=2),
                   fontsize=10, color='white', fontweight='bold')
        
        plt.tight_layout()
        
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'knn_optimization')
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'optimal_k': int(optimal_k),
            'min_error': float(errors[optimal_k - 1]),
            'k_values': list(k_values),
            'errors': [float(e) for e in errors],
            'message': f'Optimal K value is {optimal_k} with RMSE of {errors[optimal_k - 1]:.4f}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
