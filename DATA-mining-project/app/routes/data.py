from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
import os
import uuid
import arff
from werkzeug.utils import secure_filename
from app.state import state
from app.utils.helpers import is_safe_path, convert_to_native

data_bp = Blueprint('data', __name__)

@data_bp.route('/create-sample', methods=['POST'])
def create_sample():
    try:
        # Create sample data
        data = {
            'id': range(1, 101),
            'category': ['A', 'B', 'C', 'D', 'E'] * 20,
            'value': [float(i) for i in range(1, 101)],
            'status': ['active', 'inactive', 'pending', 'unknown'] * 25,
            'rating': [i % 5 + 1 for i in range(100)]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add missing values randomly
        for col in df.columns:
            df.loc[df.sample(frac=0.1).index, col] = np.nan
        
        # Save to file
        filename = f"sample_{uuid.uuid4()}.csv"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        df.to_csv(filepath, index=False)
        
        state.df = df
        state.current_filepath = filepath
        
        return jsonify({
            'message': 'Sample data created',
            'columns': df.columns.tolist(),
            'head': df.head().to_dict(orient='records'),
            'path': filepath
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
    if not path or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid path'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify(state.df.head().to_dict(orient='records'))

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
            
            if strategy == 'mean':
                if not pd.api.types.is_numeric_dtype(state.df[col]):
                    return jsonify({'error': f'Mean not applicable for {col}'}), 400
                state.df[col].fillna(state.df[col].mean(), inplace=True)
            elif strategy == 'median':
                if not pd.api.types.is_numeric_dtype(state.df[col]):
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
                    categories = state.df[col].dropna().unique()
                    mapping = {category: idx for idx, category in enumerate(categories)}
                    state.df[col] = state.df[col].map(mapping).fillna(-1)
                    result['transformed_columns'][col] = {'method': 'label'}
        
        elif method == 'onehot':
             state.df = pd.get_dummies(state.df, columns=columns, dummy_na=True)
             
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

@data_bp.route('/data/get-selected-columns', methods=['GET'])
def get_selected_columns():
    if state.df is not None:
        return jsonify({
            'message': 'Selected columns retrieved successfully',
            'selected_columns': state.df.columns.tolist()
        }), 200
    else:
        return jsonify({
            'message': 'No data loaded or columns selected yet'
        }), 400
