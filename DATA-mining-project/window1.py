# # # import os
# # # import uuid
# # # import matplotlib
# # # matplotlib.use('Agg')  # Set non-interactive backend
# # # import matplotlib.pyplot as plt
# # # import pandas as pd
# # # import numpy as np
# # # import arff
# # # from flask import Flask, request, jsonify
# # # from werkzeug.utils import secure_filename
# # # from flask_cors import CORS
# # # from flask import Flask, send_from_directory, jsonify


# # # app = Flask(__name__)
# # # CORS(app, origins=["http://localhost:3000"])
# # # # Configure upload and static folders
# # # UPLOAD_FOLDER = 'uploads'
# # # STATIC_FOLDER = 'static'
# # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # # app.config['STATIC_FOLDER'] = STATIC_FOLDER
# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # os.makedirs(STATIC_FOLDER, exist_ok=True)

# # # # Global DataFrame to store the current data being processed
# # # global_df = None
# # # current_filepath = None

# # # def is_safe_path(base, path):
# # #     """Check if path is within the base directory to prevent traversal attacks."""
# # #     base = os.path.abspath(base)
# # #     path = os.path.abspath(path)
# # #     return os.path.commonpath([base, path]) == base

# # # @app.route('/create-sample', methods=['POST'])
# # # def create_sample():
# # #     global global_df, current_filepath
    
# # #     try:
# # #         # Create sample data
# # #         data = {
# # #             'id': range(1, 101),
# # #             'category': ['A', 'B', 'C', 'D', 'E'] * 20,
# # #             'value': [float(i) for i in range(1, 101)],
# # #             'status': ['active', 'inactive', 'pending', 'unknown'] * 25,
# # #             'rating': [i % 5 + 1 for i in range(100)]
# # #         }
        
# # #         # Create DataFrame
# # #         global_df = pd.DataFrame(data)
        
# # #         # Add missing values randomly
# # #         for col in global_df.columns:
# # #             global_df.loc[global_df.sample(frac=0.1).index, col] = np.nan
        
# # #         # Save to file
# # #         filename = f"sample_{uuid.uuid4()}.csv"
# # #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# # #         global_df.to_csv(filepath, index=False)
# # #         current_filepath = filepath
        
# # #         return jsonify({
# # #             'message': 'Sample data created',
# # #             'columns': global_df.columns.tolist(),
# # #             'head': global_df.head().to_dict(orient='records'),
# # #             'path': filepath
# # #         })
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500
# # # @app.route('/upload-data', methods=['POST'])
# # # def upload_data():
# # #     global global_df, current_filepath
    
# # #     if 'file' not in request.files:
# # #         return jsonify({'error': 'No file uploaded'}), 400
# # #     file = request.files['file']
# # #     if file.filename == '':
# # #         return jsonify({'error': 'No selected file'}), 400

# # #     # Generate unique filename and save as CSV
# # #     filename = secure_filename(file.filename)
# # #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# # #     new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")

# # #     try:
# # #         if file.filename.lower().endswith('.arff'):
# # #             content = file.read().decode('utf-8')
# # #             data = arff.loads(content)
# # #             global_df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
# # #         else:
# # #             file.seek(0)
# # #             global_df = pd.read_csv(file)
        
# # #         # Save the DataFrame and update the current filepath
# # #         global_df.to_csv(filepath, index=False)
# # #         current_filepath = filepath
        
# # #         return jsonify({
# # #             'message': 'Data uploaded',
# # #             'columns': global_df.columns.tolist(),
# # #             'head': global_df.head().to_dict(orient='records'),
# # #             'path': filepath
# # #         })
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # @app.route('/data/head', methods=['GET'])
# # # def get_data_head():
# # #     global global_df, current_filepath
    
# # #     path = request.args.get('path')
# # #     if not path or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid path'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     return jsonify(global_df.head().to_dict(orient='records'))

# # # @app.route('/data/columns', methods=['GET'])
# # # def get_data_columns():
# # #     global global_df, current_filepath
    
# # #     path = request.args.get('path')
# # #     if not path or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid path'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     return jsonify({'columns': global_df.columns.tolist()})

# # # @app.route('/data/select-columns', methods=['POST'])
# # # def select_columns():
# # #     global global_df, current_filepath
    
# # #     data = request.get_json()
# # #     path = data.get('path')
# # #     columns = data.get('columns')
    
# # #     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid request'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     try:
# # #         missing = [col for col in columns if col not in global_df.columns]
# # #         if missing:
# # #             return jsonify({'error': f'Missing columns: {missing}'}), 400
        
# # #         global_df = global_df[columns]
# # #         global_df.to_csv(current_filepath, index=False)
        
# # #         return jsonify({'message': 'Columns selected', 'selected': columns})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # # import os
# # # # @app.route('/data/statistics', methods=['GET'])
# # # # def get_statistics():
# # # #     global global_df, current_filepath

# # # #     path = request.args.get('path')
# # # #     if not path or not is_safe_path(UPLOAD_FOLDER, path):
# # # #         return jsonify({'error': 'Invalid path'}), 400

# # # #     # Construct full file path safely
# # # #     full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(path))

# # # #     # If the path is different from current_filepath, load the data
# # # #     if full_path != current_filepath or global_df is None:
# # # #         try:
# # # #             global_df = pd.read_csv(full_path)
# # # #             current_filepath = full_path
# # # #         except Exception as e:
# # # #             return jsonify({'error': str(e)}), 500

# # # #     try:
# # # #         stats = {}
# # # #         for col in global_df.columns:
# # # #             col_stats = {}
# # # #             if pd.api.types.is_numeric_dtype(global_df[col]):
# # # #                 col_stats['mean'] = float(global_df[col].mean())
# # # #                 col_stats['median'] = float(global_df[col].median())
# # # #             else:
# # # #                 col_stats['mean'] = None
# # # #                 col_stats['median'] = None
# # # #             mode = global_df[col].mode()
# # # #             col_stats['mode'] = mode[0].item() if not mode.empty else None  # `.item()` to ensure native Python type
# # # #             stats[col] = col_stats
# # # #         return jsonify(stats)
# # # #     except Exception as e:
# # # #         return jsonify({'error': str(e)}), 500

# # # def convert_to_native(val):
# # #     """Convert NumPy types to native Python types"""
# # #     if isinstance(val, (np.int64, np.float64)):
# # #         return val.item()  # Convert NumPy scalar to native Python type
# # #     return val

# # # @app.route('/data/statistics', methods=['GET'])
# # # def get_statistics():
# # #     global global_df, current_filepath

# # #     path = request.args.get('path')
# # #     if not path or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid path'}), 400

# # #     full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(path))

# # #     if full_path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(full_path)
# # #             current_filepath = full_path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500

# # #     try:
# # #         stats = {}
# # #         for col in global_df.columns:
# # #             col_stats = {}
# # #             # Number of missing values
# # #             col_stats['missing'] = global_df[col].isna().sum()

# # #             # Handle numeric columns
# # #             if pd.api.types.is_numeric_dtype(global_df[col]):
# # #                 col_stats['mean'] = convert_to_native(global_df[col].mean())
# # #                 col_stats['median'] = convert_to_native(global_df[col].median())
# # #             else:
# # #                 col_stats['mean'] = None
# # #                 col_stats['median'] = None

# # #             # Mode calculation and conversion to native type
# # #             mode = global_df[col].mode()
# # #             col_stats['mode'] = convert_to_native(mode.iloc[0] if not mode.empty else None)

# # #             stats[col] = col_stats

# # #         # Convert the whole stats dict to ensure all numeric values are native Python types
# # #         stats = {col: {key: convert_to_native(value) for key, value in col_stats.items()} for col, col_stats in stats.items()}

# # #         return jsonify(stats)

# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500
# # # @app.route('/data/get-selected-columns', methods=['GET'])
# # # def get_selected_columns():
# # #     # Ensure global_df exists and is not None
# # #     if global_df is not None:
# # #         # Get the selected columns (columns in the global_df)
# # #         selected_columns = global_df.columns.tolist()
# # #         return jsonify({
# # #             'message': 'Selected columns retrieved successfully',
# # #             'selected_columns': selected_columns
# # #         }), 200
# # #     else:
# # #         return jsonify({
# # #             'message': 'No data loaded or columns selected yet'
# # #         }), 400

# # # # Fix: Properly serve static files
# # # @app.route('/static/<path:filename>')
# # # def serve_static(filename):
# # #     return send_from_directory(app.config['STATIC_FOLDER'], filename)

# # # @app.route('/data/fill-missing', methods=['POST'])
# # # def fill_missing():
# # #     global global_df, current_filepath
    
# # #     data = request.get_json()
# # #     path = data.get('path')
# # #     fill_strategies = data.get('fill')
    
# # #     if not path or not fill_strategies or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid request'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     try:
# # #         for col, strategy in fill_strategies.items():
# # #             if col not in global_df.columns:
# # #                 return jsonify({'error': f'Column {col} not found'}), 400
# # #             if strategy == 'mean':
# # #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# # #                     return jsonify({'error': f'Mean not applicable for {col}'}), 400
# # #                 global_df[col].fillna(global_df[col].mean(), inplace=True)
# # #             elif strategy == 'median':
# # #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# # #                     return jsonify({'error': f'Median not applicable for {col}'}), 400
# # #                 global_df[col].fillna(global_df[col].median(), inplace=True)
# # #             elif strategy == 'mode':
# # #                 mode_val = global_df[col].mode()[0] if not global_df[col].mode().empty else None
# # #                 global_df[col].fillna(mode_val, inplace=True)
# # #             else:
# # #                 return jsonify({'error': f'Invalid strategy: {strategy}'}), 400
        
# # #         global_df.to_csv(current_filepath, index=False)
# # #         return jsonify({'message': 'Missing values filled'})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # @app.route('/data/normalize', methods=['POST'])
# # # def normalize_data():
# # #     global global_df, current_filepath
    
# # #     data = request.get_json()
# # #     path = data.get('path')
# # #     method = data.get('method')
# # #     columns = data.get('columns')
    
# # #     if not path or not method or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid request'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     try:
# # #         for col in columns:
# # #             if col not in global_df.columns:
# # #                 return jsonify({'error': f'Column {col} not found'}), 400
# # #             if not pd.api.types.is_numeric_dtype(global_df[col]):
# # #                 return jsonify({'error': f'Non-numeric column: {col}'}), 400
# # #             if method == 'zscore':
# # #                 mean, std = global_df[col].mean(), global_df[col].std()
# # #                 if std == 0:
# # #                     global_df[col] = 0.0
# # #                 else:
# # #                     global_df[col] = (global_df[col] - mean) / std
# # #             elif method == 'minmax':
# # #                 min_val, max_val = global_df[col].min(), global_df[col].max()
# # #                 if max_val == min_val:
# # #                     global_df[col] = 0.0
# # #                 else:
# # #                     global_df[col] = (global_df[col] - min_val) / (max_val - min_val)
# # #             else:
# # #                 return jsonify({'error': 'Invalid method'}), 400
        
# # #         global_df.to_csv(current_filepath, index=False)
# # #         return jsonify({'message': 'Data normalized'})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # @app.route('/plot/scatter', methods=['POST'])
# # # def generate_scatter_plot():
# # #     global global_df, current_filepath
    
# # #     data = request.get_json()
# # #     path = data.get('path')
# # #     x_col = data.get('x')
# # #     y_col = data.get('y')
    
# # #     if not path or not x_col or not y_col or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid request'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     try:
# # #         plt.figure()
# # #         plt.scatter(global_df[x_col], global_df[y_col])
# # #         plt.xlabel(x_col)
# # #         plt.ylabel(y_col)
# # #         plt.title(f'Scatter Plot: {x_col} vs {y_col}')
# # #         plot_name = f'scatter_{uuid.uuid4()}.png'
# # #         plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_name)
# # #         plt.savefig(plot_path)
# # #         plt.close()
# # #         return jsonify({'image_url': f'/static/{plot_name}'})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # @app.route('/plot/box', methods=['POST'])
# # # def generate_box_plot():
# # #     global global_df, current_filepath
    
# # #     data = request.get_json()
# # #     path = data.get('path')
# # #     columns = data.get('columns')
    
# # #     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid request'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     try:
# # #         plt.figure()
# # #         global_df[columns].boxplot()
# # #         plt.title('Box Plot')
# # #         plot_name = f'box_{uuid.uuid4()}.png'
# # #         plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_name)
# # #         plt.savefig(plot_path)
# # #         plt.close()
# # #         return jsonify({'image_url': f'/static/{plot_name}'})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # @app.route('/data/save', methods=['POST'])
# # # def save_data():
# # #     global global_df, current_filepath
    
# # #     # Get the new file path from the request
# # #     data = request.get_json()
# # #     new_filepath = data.get('new_path')
    
# # #     if not new_filepath or not is_safe_path(UPLOAD_FOLDER, new_filepath):
# # #         return jsonify({'error': 'Invalid path'}), 400
    
# # #     try:
# # #         # Save the global_df to the new file path
# # #         global_df.to_csv(new_filepath, index=False)
        
# # #         return jsonify({'message': 'Data saved successfully', 'path': new_filepath})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500


# # # @app.route('/data/categorical-to-numerical', methods=['POST'])
# # # def categorical_to_numerical():
# # #     global global_df, current_filepath
    
# # #     data = request.get_json()
# # #     path = data.get('path')
# # #     columns = data.get('columns')
# # #     method = data.get('method', 'label')  # Default to label encoding
    
# # #     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# # #         return jsonify({'error': 'Invalid request'}), 400
    
# # #     # If the path is different from current_filepath, load the data
# # #     if path != current_filepath or global_df is None:
# # #         try:
# # #             global_df = pd.read_csv(path)
# # #             current_filepath = path
# # #         except Exception as e:
# # #             return jsonify({'error': str(e)}), 500
    
# # #     try:
# # #         missing_columns = [col for col in columns if col not in global_df.columns]
# # #         if missing_columns:
# # #             return jsonify({'error': f'Columns not found: {missing_columns}'}), 400
        
# # #         result = {'transformed_columns': {}}
        
# # #         if method == 'label':
# # #             # Label encoding (each category gets an integer value)
# # #             for col in columns:
# # #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# # #                     # Create a mapping of categories to numbers
# # #                     categories = global_df[col].dropna().unique()
# # #                     mapping = {category: idx for idx, category in enumerate(categories)}
                    
# # #                     # Store the original column
# # #                     original_col = global_df[col].copy()
                    
# # #                     # Apply the mapping
# # #                     global_df[col] = global_df[col].map(mapping)
                    
# # #                     # Handle NaN values if any
# # #                     global_df[col] = global_df[col].fillna(-1)
                    
# # #                     # Store the mapping for reference
# # #                     result['transformed_columns'][col] = {
# # #                         'method': 'label',
# # #                         'mapping': mapping
# # #                     }
# # #                 else:
# # #                     result['transformed_columns'][col] = {
# # #                         'method': 'none',
# # #                         'reason': 'Column is already numeric'
# # #                     }
        
# # #         elif method == 'onehot':
# # #             # One-hot encoding (each category becomes a binary column)
# # #             for col in columns:
# # #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# # #                     # Get dummies (one-hot encoding)
# # #                     one_hot = pd.get_dummies(global_df[col], prefix=col, drop_first=False)
                    
# # #                     # Drop the original column and join the one-hot encoded columns
# # #                     global_df = global_df.drop(col, axis=1)
# # #                     global_df = pd.concat([global_df, one_hot], axis=1)
                    
# # #                     # Store the created columns
# # #                     result['transformed_columns'][col] = {
# # #                         'method': 'onehot',
# # #                         'created_columns': one_hot.columns.tolist()
# # #                     }
# # #                 else:
# # #                     result['transformed_columns'][col] = {
# # #                         'method': 'none',
# # #                         'reason': 'Column is already numeric'
# # #                     }
        
# # #         else:
# # #             return jsonify({'error': f'Invalid method: {method}. Use "label" or "onehot"'}), 400
        
# # #         # Save the transformed dataframe
# # #         global_df.to_csv(current_filepath, index=False)
        
# # #         result['message'] = 'Categorical data transformed to numerical'
# # #         result['columns'] = global_df.columns.tolist()
        
# # #         return jsonify(result)
    
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # @app.route('/')
# # # def home():
# # #     return "Flask API is running! Use /upload-data to start."

# # # if __name__ == '__main__':
# # #     app.run(debug=True)
# # import os
# # import uuid
# # import matplotlib
# # matplotlib.use('Agg')  # Set non-interactive backend
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import numpy as np
# # import arff
# # from flask import Flask, request, jsonify, send_from_directory
# # from werkzeug.utils import secure_filename
# # from flask_cors import CORS

# # app = Flask(__name__, static_folder='static', static_url_path='/static')
# # CORS(app, origins=["http://localhost:3000"])
# # # Configure upload and static folders
# # UPLOAD_FOLDER = 'uploads'
# # STATIC_FOLDER = 'static'
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['STATIC_FOLDER'] = STATIC_FOLDER
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(STATIC_FOLDER, exist_ok=True)

# # # Global DataFrame to store the current data being processed
# # global_df = None
# # current_filepath = None

# # def is_safe_path(base, path):
# #     """Check if path is within the base directory to prevent traversal attacks."""
# #     base = os.path.abspath(base)
# #     path = os.path.abspath(path)
# #     return os.path.commonpath([base, path]) == base

# # @app.route('/create-sample', methods=['POST'])
# # def create_sample():
# #     global global_df, current_filepath
    
# #     try:
# #         # Create sample data
# #         data = {
# #             'id': range(1, 101),
# #             'category': ['A', 'B', 'C', 'D', 'E'] * 20,
# #             'value': [float(i) for i in range(1, 101)],
# #             'status': ['active', 'inactive', 'pending', 'unknown'] * 25,
# #             'rating': [i % 5 + 1 for i in range(100)]
# #         }
        
# #         # Create DataFrame
# #         global_df = pd.DataFrame(data)
        
# #         # Add missing values randomly
# #         for col in global_df.columns:
# #             global_df.loc[global_df.sample(frac=0.1).index, col] = np.nan
        
# #         # Save to file
# #         filename = f"sample_{uuid.uuid4()}.csv"
# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         global_df.to_csv(filepath, index=False)
# #         current_filepath = filepath
        
# #         return jsonify({
# #             'message': 'Sample data created',
# #             'columns': global_df.columns.tolist(),
# #             'head': global_df.head().to_dict(orient='records'),
# #             'path': filepath
# #         })
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/upload-data', methods=['POST'])
# # def upload_data():
# #     global global_df, current_filepath
    
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file uploaded'}), 400
# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400

# #     # Generate unique filename and save as CSV
# #     filename = secure_filename(file.filename)
# #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #     new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")

# #     try:
# #         if file.filename.lower().endswith('.arff'):
# #             content = file.read().decode('utf-8')
# #             data = arff.loads(content)
# #             global_df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
# #         else:
# #             file.seek(0)
# #             global_df = pd.read_csv(file)
        
# #         # Save the DataFrame and update the current filepath
# #         global_df.to_csv(filepath, index=False)
# #         current_filepath = filepath
        
# #         return jsonify({
# #             'message': 'Data uploaded',
# #             'columns': global_df.columns.tolist(),
# #             'head': global_df.head().to_dict(orient='records'),
# #             'path': filepath
# #         })
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/data/head', methods=['GET'])
# # def get_data_head():
# #     global global_df, current_filepath
    
# #     path = request.args.get('path')
# #     if not path or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid path'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     return jsonify(global_df.head().to_dict(orient='records'))

# # @app.route('/data/columns', methods=['GET'])
# # def get_data_columns():
# #     global global_df, current_filepath
    
# #     path = request.args.get('path')
# #     if not path or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid path'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     return jsonify({'columns': global_df.columns.tolist()})

# # @app.route('/data/select-columns', methods=['POST'])
# # def select_columns():
# #     global global_df, current_filepath
    
# #     data = request.get_json()
# #     path = data.get('path')
# #     columns = data.get('columns')
    
# #     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid request'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     try:
# #         missing = [col for col in columns if col not in global_df.columns]
# #         if missing:
# #             return jsonify({'error': f'Missing columns: {missing}'}), 400
        
# #         global_df = global_df[columns]
# #         global_df.to_csv(current_filepath, index=False)
        
# #         return jsonify({'message': 'Columns selected', 'selected': columns})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # def convert_to_native(val):
# #     """Convert NumPy types to native Python types"""
# #     if isinstance(val, (np.int64, np.float64)):
# #         return val.item()  # Convert NumPy scalar to native Python type
# #     return val

# # @app.route('/data/statistics', methods=['GET'])
# # def get_statistics():
# #     global global_df, current_filepath

# #     path = request.args.get('path')
# #     if not path or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid path'}), 400

# #     full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(path))

# #     if full_path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(full_path)
# #             current_filepath = full_path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500

# #     try:
# #         stats = {}
# #         for col in global_df.columns:
# #             col_stats = {}
# #             # Number of missing values
# #             col_stats['missing'] = global_df[col].isna().sum()

# #             # Handle numeric columns
# #             if pd.api.types.is_numeric_dtype(global_df[col]):
# #                 col_stats['mean'] = convert_to_native(global_df[col].mean())
# #                 col_stats['median'] = convert_to_native(global_df[col].median())
# #             else:
# #                 col_stats['mean'] = None
# #                 col_stats['median'] = None

# #             # Mode calculation and conversion to native type
# #             mode = global_df[col].mode()
# #             col_stats['mode'] = convert_to_native(mode.iloc[0] if not mode.empty else None)

# #             stats[col] = col_stats

# #         # Convert the whole stats dict to ensure all numeric values are native Python types
# #         stats = {col: {key: convert_to_native(value) for key, value in col_stats.items()} for col, col_stats in stats.items()}

# #         return jsonify(stats)

# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/data/get-selected-columns', methods=['GET'])
# # def get_selected_columns():
# #     # Ensure global_df exists and is not None
# #     if global_df is not None:
# #         # Get the selected columns (columns in the global_df)
# #         selected_columns = global_df.columns.tolist()
# #         return jsonify({
# #             'message': 'Selected columns retrieved successfully',
# #             'selected_columns': selected_columns
# #         }), 200
# #     else:
# #         return jsonify({
# #             'message': 'No data loaded or columns selected yet'
# #         }), 400

# # @app.route('/data/fill-missing', methods=['POST'])
# # def fill_missing():
# #     global global_df, current_filepath
    
# #     data = request.get_json()
# #     path = data.get('path')
# #     fill_strategies = data.get('fill')
    
# #     if not path or not fill_strategies or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid request'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     try:
# #         for col, strategy in fill_strategies.items():
# #             if col not in global_df.columns:
# #                 return jsonify({'error': f'Column {col} not found'}), 400
# #             if strategy == 'mean':
# #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# #                     return jsonify({'error': f'Mean not applicable for {col}'}), 400
# #                 global_df[col].fillna(global_df[col].mean(), inplace=True)
# #             elif strategy == 'median':
# #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# #                     return jsonify({'error': f'Median not applicable for {col}'}), 400
# #                 global_df[col].fillna(global_df[col].median(), inplace=True)
# #             elif strategy == 'mode':
# #                 mode_val = global_df[col].mode()[0] if not global_df[col].mode().empty else None
# #                 global_df[col].fillna(mode_val, inplace=True)
# #             else:
# #                 return jsonify({'error': f'Invalid strategy: {strategy}'}), 400
        
# #         global_df.to_csv(current_filepath, index=False)
# #         return jsonify({'message': 'Missing values filled'})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/data/normalize', methods=['POST'])
# # def normalize_data():
# #     global global_df, current_filepath
    
# #     data = request.get_json()
# #     path = data.get('path')
# #     method = data.get('method')
# #     columns = data.get('columns')
    
# #     if not path or not method or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid request'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     try:
# #         for col in columns:
# #             if col not in global_df.columns:
# #                 return jsonify({'error': f'Column {col} not found'}), 400
# #             if not pd.api.types.is_numeric_dtype(global_df[col]):
# #                 return jsonify({'error': f'Non-numeric column: {col}'}), 400
# #             if method == 'zscore':
# #                 mean, std = global_df[col].mean(), global_df[col].std()
# #                 if std == 0:
# #                     global_df[col] = 0.0
# #                 else:
# #                     global_df[col] = (global_df[col] - mean) / std
# #             elif method == 'minmax':
# #                 min_val, max_val = global_df[col].min(), global_df[col].max()
# #                 if max_val == min_val:
# #                     global_df[col] = 0.0
# #                 else:
# #                     global_df[col] = (global_df[col] - min_val) / (max_val - min_val)
# #             else:
# #                 return jsonify({'error': 'Invalid method'}), 400
        
# #         global_df.to_csv(current_filepath, index=False)
# #         return jsonify({'message': 'Data normalized'})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/plot/scatter', methods=['POST'])
# # def generate_scatter_plot():
# #     global global_df, current_filepath
    
# #     data = request.get_json()
# #     path = data.get('path')
# #     x_col = data.get('x')
# #     y_col = data.get('y')
    
# #     if not path or not x_col or not y_col or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid request'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     try:
# #         plt.figure(figsize=(10, 6))
# #         plt.scatter(global_df[x_col], global_df[y_col])
# #         plt.xlabel(x_col)
# #         plt.ylabel(y_col)
# #         plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        
# #         # Save plot with full absolute path
# #         plot_name = f'scatter_{uuid.uuid4()}.png'
# #         plot_path = os.path.join(os.path.abspath(app.config['STATIC_FOLDER']), plot_name)
# #         plt.savefig(plot_path)
# #         plt.close()
        
# #         # Return URL that client can use to access the image
# #         return jsonify({'image_url': f'/static/{plot_name}'})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/plot/box', methods=['POST'])
# # def generate_box_plot():
# #     global global_df, current_filepath
    
# #     data = request.get_json()
# #     path = data.get('path')
# #     columns = data.get('columns')
    
# #     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid request'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     try:
# #         plt.figure(figsize=(10, 6))
# #         global_df[columns].boxplot()
# #         plt.title('Box Plot')
        
# #         # Save plot with full absolute path
# #         plot_name = f'box_{uuid.uuid4()}.png'
# #         plot_path = os.path.join(os.path.abspath(app.config['STATIC_FOLDER']), plot_name)
# #         plt.savefig(plot_path, bbox_inches='tight')
# #         plt.close()
        
# #         # Return URL that client can use to access the image
# #         return jsonify({'image_url': f'/static/{plot_name}'})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/data/save', methods=['POST'])
# # def save_data():
# #     global global_df, current_filepath
    
# #     # Get the new file path from the request
# #     data = request.get_json()
# #     new_filepath = data.get('new_path')
    
# #     if not new_filepath or not is_safe_path(UPLOAD_FOLDER, new_filepath):
# #         return jsonify({'error': 'Invalid path'}), 400
    
# #     try:
# #         # Save the global_df to the new file path
# #         global_df.to_csv(new_filepath, index=False)
        
# #         return jsonify({'message': 'Data saved successfully', 'path': new_filepath})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/data/categorical-to-numerical', methods=['POST'])
# # def categorical_to_numerical():
# #     global global_df, current_filepath
    
# #     data = request.get_json()
# #     path = data.get('path')
# #     columns = data.get('columns')
# #     method = data.get('method', 'label')  # Default to label encoding
    
# #     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
# #         return jsonify({'error': 'Invalid request'}), 400
    
# #     # If the path is different from current_filepath, load the data
# #     if path != current_filepath or global_df is None:
# #         try:
# #             global_df = pd.read_csv(path)
# #             current_filepath = path
# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500
    
# #     try:
# #         missing_columns = [col for col in columns if col not in global_df.columns]
# #         if missing_columns:
# #             return jsonify({'error': f'Columns not found: {missing_columns}'}), 400
        
# #         result = {'transformed_columns': {}}
        
# #         if method == 'label':
# #             # Label encoding (each category gets an integer value)
# #             for col in columns:
# #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# #                     # Create a mapping of categories to numbers
# #                     categories = global_df[col].dropna().unique()
# #                     mapping = {category: idx for idx, category in enumerate(categories)}
                    
# #                     # Store the original column
# #                     original_col = global_df[col].copy()
                    
# #                     # Apply the mapping
# #                     global_df[col] = global_df[col].map(mapping)
                    
# #                     # Handle NaN values if any
# #                     global_df[col] = global_df[col].fillna(-1)
                    
# #                     # Store the mapping for reference
# #                     result['transformed_columns'][col] = {
# #                         'method': 'label',
# #                         'mapping': mapping
# #                     }
# #                 else:
# #                     result['transformed_columns'][col] = {
# #                         'method': 'none',
# #                         'reason': 'Column is already numeric'
# #                     }
        
# #         elif method == 'onehot':
# #             # One-hot encoding (each category becomes a binary column)
# #             for col in columns:
# #                 if not pd.api.types.is_numeric_dtype(global_df[col]):
# #                     # Get dummies (one-hot encoding)
# #                     one_hot = pd.get_dummies(global_df[col], prefix=col, drop_first=False)
                    
# #                     # Drop the original column and join the one-hot encoded columns
# #                     global_df = global_df.drop(col, axis=1)
# #                     global_df = pd.concat([global_df, one_hot], axis=1)
                    
# #                     # Store the created columns
# #                     result['transformed_columns'][col] = {
# #                         'method': 'onehot',
# #                         'created_columns': one_hot.columns.tolist()
# #                     }
# #                 else:
# #                     result['transformed_columns'][col] = {
# #                         'method': 'none',
# #                         'reason': 'Column is already numeric'
# #                     }
        
# #         else:
# #             return jsonify({'error': f'Invalid method: {method}. Use "label" or "onehot"'}), 400
        
# #         # Save the transformed dataframe
# #         global_df.to_csv(current_filepath, index=False)
        
# #         result['message'] = 'Categorical data transformed to numerical'
# #         result['columns'] = global_df.columns.tolist()
        
# #         return jsonify(result)
    
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/')
# # def home():
# #     return "Flask API is running! Use /upload-data to start."

# # if __name__ == '__main__':
# #     app.run(debug=True)
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

# def is_safe_path(base, path):
#     """Check if path is within the base directory to prevent traversal attacks."""
#     base = os.path.abspath(base)
#     path = os.path.abspath(path)
#     return os.path.commonpath([base, path]) == base

# @app.route('/create-sample', methods=['POST'])
# def create_sample():
#     global global_df, current_filepath
    
#     try:
#         # Create sample data
#         data = {
#             'id': range(1, 101),
#             'category': ['A', 'B', 'C', 'D', 'E'] * 20,
#             'value': [float(i) for i in range(1, 101)],
#             'status': ['active', 'inactive', 'pending', 'unknown'] * 25,
#             'rating': [i % 5 + 1 for i in range(100)]
#         }
        
#         # Create DataFrame
#         global_df = pd.DataFrame(data)
        
#         # Add missing values randomly
#         for col in global_df.columns:
#             global_df.loc[global_df.sample(frac=0.1).index, col] = np.nan
        
#         # Save to file
#         filename = f"sample_{uuid.uuid4()}.csv"
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         global_df.to_csv(filepath, index=False)
#         current_filepath = filepath
        
#         return jsonify({
#             'message': 'Sample data created',
#             'columns': global_df.columns.tolist(),
#             'head': global_df.head().to_dict(orient='records'),
#             'path': filepath
#         })
#     except Exception as e:
#         logger.error(f"Error in create_sample: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/upload-data', methods=['POST'])
# def upload_data():
#     global global_df, current_filepath
    
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Generate unique filename and save as CSV
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")

#     try:
#         if file.filename.lower().endswith('.arff'):
#             content = file.read().decode('utf-8')
#             data = arff.loads(content)
#             global_df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
#         else:
#             file.seek(0)
#             global_df = pd.read_csv(file)
        
#         # Save the DataFrame and update the current filepath
#         global_df.to_csv(filepath, index=False)
#         current_filepath = filepath
        
#         return jsonify({
#             'message': 'Data uploaded',
#             'columns': global_df.columns.tolist(),
#             'head': global_df.head().to_dict(orient='records'),
#             'path': filepath
#         })
#     except Exception as e:
#         logger.error(f"Error in upload_data: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/data/head', methods=['GET'])
# def get_data_head():
#     global global_df, current_filepath
    
#     path = request.args.get('path')
#     if not path or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid path'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in get_data_head: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     return jsonify(global_df.head().to_dict(orient='records'))

# @app.route('/data/columns', methods=['GET'])
# def get_data_columns():
#     global global_df, current_filepath
    
#     path = request.args.get('path')
#     if not path or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid path'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in get_data_columns: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     return jsonify({'columns': global_df.columns.tolist()})

# @app.route('/data/select-columns', methods=['POST'])
# def select_columns():
#     global global_df, current_filepath
    
#     data = request.get_json()
#     path = data.get('path')
#     columns = data.get('columns')
    
#     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid request'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in select_columns: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     try:
#         missing = [col for col in columns if col not in global_df.columns]
#         if missing:
#             return jsonify({'error': f'Missing columns: {missing}'}), 400
        
#         global_df = global_df[columns]
#         global_df.to_csv(current_filepath, index=False)
        
#         return jsonify({'message': 'Columns selected', 'selected': columns})
#     except Exception as e:
#         logger.error(f"Error in select_columns: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# def convert_to_native(val):
#     """Convert NumPy types to native Python types"""
#     if isinstance(val, (np.int64, np.float64)):
#         return val.item()  # Convert NumPy scalar to native Python type
#     return val

# @app.route('/data/statistics', methods=['GET'])
# def get_statistics():
#     global global_df, current_filepath

#     path = request.args.get('path')
#     if not path or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid path'}), 400

#     full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(path))

#     if full_path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(full_path)
#             current_filepath = full_path
#         except Exception as e:
#             logger.error(f"Error in get_statistics: {str(e)}")
#             return jsonify({'error': str(e)}), 500

#     try:
#         stats = {}
#         for col in global_df.columns:
#             col_stats = {}
#             # Number of missing values
#             col_stats['missing'] = global_df[col].isna().sum()

#             # Handle numeric columns
#             if pd.api.types.is_numeric_dtype(global_df[col]):
#                 col_stats['mean'] = convert_to_native(global_df[col].mean())
#                 col_stats['median'] = convert_to_native(global_df[col].median())
#             else:
#                 col_stats['mean'] = None
#                 col_stats['median'] = None

#             # Mode calculation and conversion to native type
#             mode = global_df[col].mode()
#             col_stats['mode'] = convert_to_native(mode.iloc[0] if not mode.empty else None)

#             stats[col] = col_stats

#         # Convert the whole stats dict to ensure all numeric values are native Python types
#         stats = {col: {key: convert_to_native(value) for key, value in col_stats.items()} for col, col_stats in stats.items()}

#         return jsonify(stats)

#     except Exception as e:
#         logger.error(f"Error in get_statistics: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/data/get-selected-columns', methods=['GET'])
# def get_selected_columns():
#     # Ensure global_df exists and is not None
#     if global_df is not None:
#         # Get the selected columns (columns in the global_df)
#         selected_columns = global_df.columns.tolist()
#         return jsonify({
#             'message': 'Selected columns retrieved successfully',
#             'selected_columns': selected_columns
#         }), 200
#     else:
#         return jsonify({
#             'message': 'No data loaded or columns selected yet'
#         }), 400

# @app.route('/data/fill-missing', methods=['POST'])
# def fill_missing():
#     global global_df, current_filepath
    
#     data = request.get_json()
#     path = data.get('path')
#     fill_strategies = data.get('fill')
    
#     if not path or not fill_strategies or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid request'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in fill_missing: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     try:
#         for col, strategy in fill_strategies.items():
#             if col not in global_df.columns:
#                 return jsonify({'error': f'Column {col} not found'}), 400
#             if strategy == 'mean':
#                 if not pd.api.types.is_numeric_dtype(global_df[col]):
#                     return jsonify({'error': f'Mean not applicable for {col}'}), 400
#                 global_df[col].fillna(global_df[col].mean(), inplace=True)
#             elif strategy == 'median':
#                 if not pd.api.types.is_numeric_dtype(global_df[col]):
#                     return jsonify({'error': f'Median not applicable for {col}'}), 400
#                 global_df[col].fillna(global_df[col].median(), inplace=True)
#             elif strategy == 'mode':
#                 mode_val = global_df[col].mode()[0] if not global_df[col].mode().empty else None
#                 global_df[col].fillna(mode_val, inplace=True)
#             else:
#                 return jsonify({'error': f'Invalid strategy: {strategy}'}), 400
        
#         global_df.to_csv(current_filepath, index=False)
#         return jsonify({'message': 'Missing values filled'})
#     except Exception as e:
#         logger.error(f"Error in fill_missing: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/data/normalize', methods=['POST'])
# def normalize_data():
#     global global_df, current_filepath
    
#     data = request.get_json()
#     path = data.get('path')
#     method = data.get('method')
#     columns = data.get('columns')
    
#     if not path or not method or not columns or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid request'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in normalize_data: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     try:
#         for col in columns:
#             if col not in global_df.columns:
#                 return jsonify({'error': f'Column {col} not found'}), 400
#             if not pd.api.types.is_numeric_dtype(global_df[col]):
#                 return jsonify({'error': f'Non-numeric column: {col}'}), 400
#             if method == 'zscore':
#                 mean, std = global_df[col].mean(), global_df[col].std()
#                 if std == 0:
#                     global_df[col] = 0.0
#                 else:
#                     global_df[col] = (global_df[col] - mean) / std
#             elif method == 'minmax':
#                 min_val, max_val = global_df[col].min(), global_df[col].max()
#                 if max_val == min_val:
#                     global_df[col] = 0.0
#                 else:
#                     global_df[col] = (global_df[col] - min_val) / (max_val - min_val)
#             else:
#                 return jsonify({'error': 'Invalid method'}), 400
        
#         global_df.to_csv(current_filepath, index=False)
#         return jsonify({'message': 'Data normalized'})
#     except Exception as e:
#         logger.error(f"Error in normalize_data: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/plot/scatter', methods=['POST'])
# def generate_scatter_plot():
#     global global_df, current_filepath
    
#     data = request.get_json()
#     path = data.get('path')
#     x_col = data.get('x')
#     y_col = data.get('y')
    
#     if not path or not x_col or not y_col or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid request'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in generate_scatter_plot: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     try:
#         plt.figure(figsize=(10, 6))
#         plt.scatter(global_df[x_col], global_df[y_col])
#         plt.xlabel(x_col)
#         plt.ylabel(y_col)
#         plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        
#         # Create a unique filename for the plot
#         plot_name = f'scatter_{uuid.uuid4()}.png'
#         plot_path = os.path.join(STATIC_FOLDER, plot_name)
        
#         # Log the path for debugging
#         logger.debug(f"Saving scatter plot to: {plot_path}")
        
#         # Save the plot
#         plt.savefig(plot_path, bbox_inches='tight', dpi=100)
#         plt.close()
        
#         # Return URL that client can use to access the image
#         return jsonify({'image_url': f'/static/{plot_name}'})
#     except Exception as e:
#         logger.error(f"Error in generate_scatter_plot: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/plot/box', methods=['POST'])
# def generate_box_plot():
#     global global_df, current_filepath
    
#     data = request.get_json()
#     path = data.get('path')
#     columns = data.get('columns')
    
#     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid request'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in generate_box_plot: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     try:
#         plt.figure(figsize=(10, 6))
#         global_df[columns].boxplot()
#         plt.title('Box Plot')
        
#         # Create a unique filename for the plot
#         plot_name = f'box_{uuid.uuid4()}.png'
#         plot_path = os.path.join(STATIC_FOLDER, plot_name)
        
#         # Log the path for debugging
#         logger.debug(f"Saving box plot to: {plot_path}")
        
#         # Save the plot
#         plt.savefig(plot_path, bbox_inches='tight', dpi=100)
#         plt.close()
        
#         # Return URL that client can use to access the image
#         return jsonify({'image_url': f'/static/{plot_name}'})
#     except Exception as e:
#         logger.error(f"Error in generate_box_plot: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/data/save', methods=['POST'])
# def save_data():
#     global global_df, current_filepath
    
#     # Get the new file path from the request
#     data = request.get_json()
#     new_filepath = data.get('new_path')
    
#     if not new_filepath or not is_safe_path(UPLOAD_FOLDER, new_filepath):
#         return jsonify({'error': 'Invalid path'}), 400
    
#     try:
#         # Save the global_df to the new file path
#         global_df.to_csv(new_filepath, index=False)
        
#         return jsonify({'message': 'Data saved successfully', 'path': new_filepath})
#     except Exception as e:
#         logger.error(f"Error in save_data: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/data/categorical-to-numerical', methods=['POST'])
# def categorical_to_numerical():
#     global global_df, current_filepath
    
#     data = request.get_json()
#     path = data.get('path')
#     columns = data.get('columns')
#     method = data.get('method', 'label')  # Default to label encoding
    
#     if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid request'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in categorical_to_numerical: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     try:
#         missing_columns = [col for col in columns if col not in global_df.columns]
#         if missing_columns:
#             return jsonify({'error': f'Columns not found: {missing_columns}'}), 400
        
#         result = {'transformed_columns': {}}
        
#         if method == 'label':
#             # Label encoding (each category gets an integer value)
#             for col in columns:
#                 if not pd.api.types.is_numeric_dtype(global_df[col]):
#                     # Create a mapping of categories to numbers
#                     categories = global_df[col].dropna().unique()
#                     mapping = {category: idx for idx, category in enumerate(categories)}
                    
#                     # Store the original column
#                     original_col = global_df[col].copy()
                    
#                     # Apply the mapping
#                     global_df[col] = global_df[col].map(mapping)
                    
#                     # Handle NaN values if any
#                     global_df[col] = global_df[col].fillna(-1)
                    
#                     # Store the mapping for reference
#                     result['transformed_columns'][col] = {
#                         'method': 'label',
#                         'mapping': mapping
#                     }
#                 else:
#                     result['transformed_columns'][col] = {
#                         'method': 'none',
#                         'reason': 'Column is already numeric'
#                     }
        
#         elif method == 'onehot':
#             # One-hot encoding (each category becomes a binary column)
#             for col in columns:
#                 if not pd.api.types.is_numeric_dtype(global_df[col]):
#                     # Get dummies (one-hot encoding)
#                     one_hot = pd.get_dummies(global_df[col], prefix=col, drop_first=False)
                    
#                     # Drop the original column and join the one-hot encoded columns
#                     global_df = global_df.drop(col, axis=1)
#                     global_df = pd.concat([global_df, one_hot], axis=1)
                    
#                     # Store the created columns
#                     result['transformed_columns'][col] = {
#                         'method': 'onehot',
#                         'created_columns': one_hot.columns.tolist()
#                     }
#                 else:
#                     result['transformed_columns'][col] = {
#                         'method': 'none',
#                         'reason': 'Column is already numeric'
#                     }
        
#         else:
#             return jsonify({'error': f'Invalid method: {method}. Use "label" or "onehot"'}), 400
        
#         # Save the transformed dataframe
#         global_df.to_csv(current_filepath, index=False)
        
#         result['message'] = 'Categorical data transformed to numerical'
#         result['columns'] = global_df.columns.tolist()
        
#         return jsonify(result)
    
#     except Exception as e:
#         logger.error(f"Error in categorical_to_numerical: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# # Explicit route to serve static files
# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     logger.debug(f"Serving static file: {filename}")
#     return send_from_directory(STATIC_FOLDER, filename)

# @app.route('/')
# def home():
#     return "Flask API is running! Use /upload-data to start."

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import uuid
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arff
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from werkzeug.utils import secure_filename
import json
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import os
import uuid
import io
import base64
from scipy.interpolate import UnivariateSpline  # Use scipy for spline functionality

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Setup basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define absolute paths for folders
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
CORS(app, origins=["http://localhost:3000"])

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size



# Global DataFrame to store the current data being processed
global_df = None
current_filepath = None

# In-memory storage for clustering models and labels (for comparison)
stored_results = {}

# ========== Helper functions ==========

def is_safe_path(base, path):
    """Check if path is within the base directory to prevent traversal attacks."""
    base = os.path.abspath(base)
    path = os.path.abspath(path)
    return os.path.commonpath([base, path]) == base
def get_best_features_for_visualization(df, labels=None):
    """Select best features for 2D visualization using PCA if needed"""
    # If the dataframe has 2 columns, use them directly
    if df.shape[1] == 2:
        return df, df.columns.tolist()
    
    # If more than 2 columns, use PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)
    
    # Calculate feature importance for explanation
    feature_importance = [(abs(pca.components_[0][i]) + abs(pca.components_[1][i]), df.columns[i]) 
                          for i in range(len(df.columns))]
    feature_importance.sort(reverse=True)
    
    top_features = [f[1] for f in feature_importance[:2]]
    
    # Return DataFrame with PCA components
    pca_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    
    return pca_df, top_features
def convert_to_native(val):
    """Convert NumPy types to native Python types"""
    if isinstance(val, (np.int64, np.float64)):
        return val.item()  # Convert NumPy scalar to native Python type
    return val

def save_plot(fig, prefix):
    """Save matplotlib figure to static folder and return URL path"""
    filename = f"{prefix}_{uuid.uuid4()}.png"
    path = os.path.join(app.config['STATIC_FOLDER'], filename)
    fig.savefig(path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return f'/static/{filename}'

# def calculate_metrics(df, labels):
#     """Calculate cluster quality metrics if there are at least 2 clusters"""
#     if len(set(labels)) < 2:
#         # Silhouette and others need at least 2 clusters
#         return {
#             'silhouette': None,
#             'davies_bouldin': None,
#             'calinski_harabasz': None
#         }
#     try:
#         return {
#             'silhouette': silhouette_score(df, labels),
#             'davies_bouldin': davies_bouldin_score(df, labels),
#             'calinski_harabasz': calinski_harabasz_score(df, labels)
#         }
#     except Exception as e:
#         return {
#             'error': str(e),
#             'silhouette': None,
#             'davies_bouldin': None,
#             'calinski_harabasz': None
#         }
def calculate_metrics(data, labels):
    """Calculate clustering performance metrics"""
    metrics = {}
    
    # Handle the case when there's only one cluster label or all points are noise (-1)
    unique_labels = set(labels)
    valid_labels = [l for l in unique_labels if l != -1]
    
    if len(valid_labels) > 1:
        # Filter out noise points for metric calculation
        if -1 in unique_labels:
            valid_indices = labels != -1
            if sum(valid_indices) > 1:  # Ensure we have at least 2 valid points
                metrics['silhouette'] = float(silhouette_score(data[valid_indices], labels[valid_indices]))
                metrics['davies_bouldin'] = float(davies_bouldin_score(data[valid_indices], labels[valid_indices]))
                metrics['calinski_harabasz'] = float(calinski_harabasz_score(data[valid_indices], labels[valid_indices]))
        else:
            metrics['silhouette'] = float(silhouette_score(data, labels))
            metrics['davies_bouldin'] = float(davies_bouldin_score(data, labels))
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(data, labels))
    else:
        metrics['silhouette'] = 0.0
        metrics['davies_bouldin'] = 0.0
        metrics['calinski_harabasz'] = 0.0
    
    metrics['num_clusters'] = len(valid_labels)
    
    return metrics
# ========== Data Preprocessing Routes ==========

@app.route('/create-sample', methods=['POST'])
def create_sample():
    global global_df, current_filepath
    
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
        global_df = pd.DataFrame(data)
        
        # Add missing values randomly
        for col in global_df.columns:
            global_df.loc[global_df.sample(frac=0.1).index, col] = np.nan
        
        # Save to file
        filename = f"sample_{uuid.uuid4()}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        global_df.to_csv(filepath, index=False)
        current_filepath = filepath
        
        return jsonify({
            'message': 'Sample data created',
            'columns': global_df.columns.tolist(),
            'head': global_df.head().to_dict(orient='records'),
            'path': filepath
        })
    except Exception as e:
        logger.error(f"Error in create_sample: {str(e)}")
        return jsonify({'error': str(e)}), 500

# @app.route('/upload-data', methods=['POST'])
# def upload_data():
#     global global_df, current_filepath
    
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Generate unique filename and save as CSV
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")

#     try:
#         if file.filename.lower().endswith('.arff'):
#             content = file.read().decode('utf-8')
#             data = arff.loads(content)
#             global_df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
#         else:
#             file.seek(0)
#             global_df = pd.read_csv(file)
        
#         # Save the DataFrame and update the current filepath
#         global_df.to_csv(filepath, index=False)
#         current_filepath = filepath
        
#         return jsonify({
#             'message': 'Data uploaded',
#             'columns': global_df.columns.tolist(),
#             'head': global_df.head().to_dict(orient='records'),
#             'path': filepath
#         })
#     except Exception as e:
#         logger.error(f"Error in upload_data: {str(e)}")
#         return jsonify({'error': str(e)}), 500

@app.route('/upload-data', methods=['POST'])
def upload_data():
    global global_df, current_filepath
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Generate unique filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")

    try:
        # Save the uploaded file directly first
        file.save(filepath)
        
        # Process based on file type
        if file.filename.lower().endswith('.arff'):
            try:
                import arff  # Make sure to install 'liac-arff' package
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = arff.load(f)
                global_df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
            except Exception as e:
                logger.error(f"Error parsing ARFF: {str(e)}")
                raise
        else:
            # Try different encodings if there are issues
            try:
                global_df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                try:
                    global_df = pd.read_csv(filepath, encoding='latin1')
                except:
                    global_df = pd.read_csv(filepath, encoding='ISO-8859-1')
            except Exception as e:
                logger.error(f"Error reading CSV: {str(e)}")
                raise
        
        current_filepath = filepath
        
        # Convert NaN values to None for JSON serialization
        preview_data = global_df.head().replace({np.nan: None}).to_dict(orient='records')
        
        return jsonify({
            'message': 'Data uploaded successfully',
            'columns': global_df.columns.tolist(),
            'head': preview_data,
            'rows': len(global_df),
            'path': filepath
        })
    except Exception as e:
        logger.error(f"Error in upload_data: {str(e)}")
        
        # Clean up the file if it was saved but processing failed
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
                
        return jsonify({
            'error': 'Failed to process file',
            'details': str(e)
        }), 500

# @app.route('/data/head', methods=['GET'])
# def get_data_head():
#     global global_df, current_filepath
    
#     path = request.args.get('path')
#     if not path or not is_safe_path(UPLOAD_FOLDER, path):
#         return jsonify({'error': 'Invalid path'}), 400
    
#     # If the path is different from current_filepath, load the data
#     if path != current_filepath or global_df is None:
#         try:
#             global_df = pd.read_csv(path)
#             current_filepath = path
#         except Exception as e:
#             logger.error(f"Error in get_data_head: {str(e)}")
#             return jsonify({'error': str(e)}), 500
    
#     return jsonify(global_df.head().to_dict(orient='records'))

@app.route('/data/head', methods=['GET'])
def get_data_head():
    global global_df, current_filepath
    
    path = request.args.get('path')
    if not path or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid path'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in get_data_head: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # Convert NaN values to None for proper JSON serialization
    head_data = global_df.head().replace({np.nan: None}).to_dict(orient='records')
    
    return jsonify(head_data)

# Helper function to check if the path is safe (prevent directory traversal)

@app.route('/data/columns', methods=['GET'])
def get_data_columns():
    global global_df, current_filepath
    
    path = request.args.get('path')
    if not path or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid path'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in get_data_columns: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'columns': global_df.columns.tolist()})

@app.route('/data/select-columns', methods=['POST'])
def select_columns():
    global global_df, current_filepath
    
    data = request.get_json()
    path = data.get('path')
    columns = data.get('columns')
    
    if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid request'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in select_columns: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    try:
        missing = [col for col in columns if col not in global_df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400
        
        global_df = global_df[columns]
        global_df.to_csv(current_filepath, index=False)
        
        return jsonify({'message': 'Columns selected', 'selected': columns})
    except Exception as e:
        logger.error(f"Error in select_columns: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/statistics', methods=['GET'])
def get_statistics():
    global global_df, current_filepath

    path = request.args.get('path')
    if not path or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid path'}), 400

    full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(path))

    if full_path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(full_path)
            current_filepath = full_path
        except Exception as e:
            logger.error(f"Error in get_statistics: {str(e)}")
            return jsonify({'error': str(e)}), 500

    try:
        stats = {}
        for col in global_df.columns:
            col_stats = {}
            # Number of missing values
            col_stats['missing'] = global_df[col].isna().sum()

            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(global_df[col]):
                col_stats['mean'] = convert_to_native(global_df[col].mean())
                col_stats['median'] = convert_to_native(global_df[col].median())
            else:
                col_stats['mean'] = None
                col_stats['median'] = None

            # Mode calculation and conversion to native type
            mode = global_df[col].mode()
            col_stats['mode'] = convert_to_native(mode.iloc[0] if not mode.empty else None)

            stats[col] = col_stats

        # Convert the whole stats dict to ensure all numeric values are native Python types
        stats = {col: {key: convert_to_native(value) for key, value in col_stats.items()} for col, col_stats in stats.items()}

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/get-selected-columns', methods=['GET'])
def get_selected_columns():
    # Ensure global_df exists and is not None
    if global_df is not None:
        # Get the selected columns (columns in the global_df)
        selected_columns = global_df.columns.tolist()
        return jsonify({
            'message': 'Selected columns retrieved successfully',
            'selected_columns': selected_columns
        }), 200
    else:
        return jsonify({
            'message': 'No data loaded or columns selected yet'
        }), 400

@app.route('/data/fill-missing', methods=['POST'])
def fill_missing():
    global global_df, current_filepath
    
    data = request.get_json()
    path = data.get('path')
    fill_strategies = data.get('fill')
    
    if not path or not fill_strategies or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid request'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in fill_missing: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    try:
        for col, strategy in fill_strategies.items():
            if col not in global_df.columns:
                return jsonify({'error': f'Column {col} not found'}), 400
            if strategy == 'mean':
                if not pd.api.types.is_numeric_dtype(global_df[col]):
                    return jsonify({'error': f'Mean not applicable for {col}'}), 400
                global_df[col].fillna(global_df[col].mean(), inplace=True)
            elif strategy == 'median':
                if not pd.api.types.is_numeric_dtype(global_df[col]):
                    return jsonify({'error': f'Median not applicable for {col}'}), 400
                global_df[col].fillna(global_df[col].median(), inplace=True)
            elif strategy == 'mode':
                mode_val = global_df[col].mode()[0] if not global_df[col].mode().empty else None
                global_df[col].fillna(mode_val, inplace=True)
            else:
                return jsonify({'error': f'Invalid strategy: {strategy}'}), 400
        
        global_df.to_csv(current_filepath, index=False)
        return jsonify({'message': 'Missing values filled'})
    except Exception as e:
        logger.error(f"Error in fill_missing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/normalize', methods=['POST'])
def normalize_data():
    global global_df, current_filepath
    
    data = request.get_json()
    path = data.get('path')
    method = data.get('method')
    columns = data.get('columns')
    
    if not path or not method or not columns or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid request'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in normalize_data: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    try:
        for col in columns:
            if col not in global_df.columns:
                return jsonify({'error': f'Column {col} not found'}), 400
            if not pd.api.types.is_numeric_dtype(global_df[col]):
                return jsonify({'error': f'Non-numeric column: {col}'}), 400
            if method == 'zscore':
                mean, std = global_df[col].mean(), global_df[col].std()
                if std == 0:
                    global_df[col] = 0.0
                else:
                    global_df[col] = (global_df[col] - mean) / std
            elif method == 'minmax':
                min_val, max_val = global_df[col].min(), global_df[col].max()
                if max_val == min_val:
                    global_df[col] = 0.0
                else:
                    global_df[col] = (global_df[col] - min_val) / (max_val - min_val)
            else:
                return jsonify({'error': 'Invalid method'}), 400
        
        global_df.to_csv(current_filepath, index=False)
        return jsonify({'message': 'Data normalized'})
    except Exception as e:
        logger.error(f"Error in normalize_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/plot/scatter', methods=['POST'])
def generate_scatter_plot():
    global global_df, current_filepath
    
    data = request.get_json()
    path = data.get('path')
    x_col = data.get('x')
    y_col = data.get('y')
    
    if not path or not x_col or not y_col or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid request'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in generate_scatter_plot: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(global_df[x_col], global_df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        
        # Create a unique filename for the plot
        plot_name = f'scatter_{uuid.uuid4()}.png'
        plot_path = os.path.join(STATIC_FOLDER, plot_name)
        
        # Log the path for debugging
        logger.debug(f"Saving scatter plot to: {plot_path}")
        
        # Save the plot
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Return URL that client can use to access the image
        return jsonify({'image_url': f'/static/{plot_name}'})
    except Exception as e:
        logger.error(f"Error in generate_scatter_plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/plot/box', methods=['POST'])
def generate_box_plot():
    global global_df, current_filepath
    
    data = request.get_json()
    path = data.get('path')
    columns = data.get('columns')
    
    if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid request'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in generate_box_plot: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    try:
        plt.figure(figsize=(10, 6))
        global_df[columns].boxplot()
        plt.title('Box Plot')
        
        # Create a unique filename for the plot
        plot_name = f'box_{uuid.uuid4()}.png'
        plot_path = os.path.join(STATIC_FOLDER, plot_name)
        
        # Log the path for debugging
        logger.debug(f"Saving box plot to: {plot_path}")
        
        # Save the plot
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Return URL that client can use to access the image
        return jsonify({'image_url': f'/static/{plot_name}'})
    except Exception as e:
        logger.error(f"Error in generate_box_plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/save', methods=['POST'])
def save_data():
    global global_df, current_filepath
    
    # Get the new file path from the request
    data = request.get_json()
    new_filepath = data.get('new_path')
    
    if not new_filepath or not is_safe_path(UPLOAD_FOLDER, new_filepath):
        return jsonify({'error': 'Invalid path'}), 400
    
    try:
        # Save the global_df to the new file path
        global_df.to_csv(new_filepath, index=False)
        
        return jsonify({'message': 'Data saved successfully', 'path': new_filepath})
    except Exception as e:
        logger.error(f"Error in save_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/categorical-to-numerical', methods=['POST'])
def categorical_to_numerical():
    global global_df, current_filepath
    
    data = request.get_json()
    path = data.get('path')
    columns = data.get('columns')
    method = data.get('method', 'label')  # Default to label encoding
    
    if not path or not columns or not is_safe_path(UPLOAD_FOLDER, path):
        return jsonify({'error': 'Invalid request'}), 400
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in categorical_to_numerical: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    try:
        missing_columns = [col for col in columns if col not in global_df.columns]
        if missing_columns:
            return jsonify({'error': f'Columns not found: {missing_columns}'}), 400
        
        result = {'transformed_columns': {}}
        
        if method == 'label':
            # Label encoding (each category gets an integer value)
            for col in columns:
                if not pd.api.types.is_numeric_dtype(global_df[col]):
                    # Create a mapping of categories to numbers
                    categories = global_df[col].dropna().unique()
                    mapping = {category: idx for idx, category in enumerate(categories)}
                    
                    # Store the original column
                    original_col = global_df[col].copy()
                    
                    # Apply the mapping
                    global_df[col] = global_df[col].map(mapping)
                    
                    # Handle NaN values if any
                    global_df[col] = global_df[col].fillna(-1)
                    
                    # Store the mapping for reference
                    result['transformed_columns'][col] = {
                        'method': 'label',
                        'mapping': mapping
                    }
                else:
                    result['transformed_columns'][col] = {
                        'method': 'none',
                        'reason': 'Column is already numeric'
                    }
        
        elif method == 'onehot':
            # One-hot encoding (each category becomes a binary column)
            for col in columns:
                if not pd.api.types.is_numeric_dtype(global_df[col]):
                    # Get dummies (one-hot encoding)
                    one_hot = pd.get_dummies(global_df[col], prefix=col, drop_first=False)
                    
                    # Drop the original column and join the one-hot encoded columns
                    global_df = global_df.drop(col, axis=1)
                    global_df = pd.concat([global_df, one_hot], axis=1)
                    
                    # Store the created columns
                    result['transformed_columns'][col] = {
                        'method': 'onehot',
                        'created_columns': one_hot.columns.tolist()
                    }
                else:
                    result['transformed_columns'][col] = {
                        'method': 'none',
                        'reason': 'Column is already numeric'
                    }
        
        else:
            return jsonify({'error': f'Invalid method: {method}. Use "label" or "onehot"'}), 400
        
        # Save the transformed dataframe
        global_df.to_csv(current_filepath, index=False)
        
        result['message'] = 'Categorical data transformed to numerical'
        result['columns'] = global_df.columns.tolist()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in categorical_to_numerical: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ========== Clustering Routes ==========

@app.route('/upload-data-clustering', methods=['POST'])
def upload_data_clustering():
    """Handle file upload and parse CSV data"""
    global global_df, current_filepath
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Handle both standard CSV and ARFF files
        if filename.endswith('.csv'):
            global_df = pd.read_csv(file)
        else:
            global_df = pd.read_csv(file, comment='@')
            
        if global_df.empty:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        # Save the processed file
        global_df.to_csv(filepath, index=False)
        current_filepath = filepath
        
        return jsonify({
            'message': 'Data uploaded successfully',
            'columns': global_df.columns.tolist(),
            'head': global_df.head().to_dict(orient='records'),
            'path': filepath
        })
    except Exception as e:
        logger.error(f"Error in upload_data_clustering: {str(e)}")
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
    global global_df, current_filepath
    
    data = request.json
    path = data.get('path')
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in generate_elbow: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    algorithm = data['algorithm']
    max_k = data.get('max_k', 10)

    try:
        distortions = []
        for k in range(1, max_k + 1):
            model = KMeans(n_clusters=k) if algorithm == 'kmeans' else KMedoids(n_clusters=k)
            model.fit(global_df)
            distortions.append(model.inertia_)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), distortions, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title(f'Elbow Method ({algorithm.upper()})')
        plt.grid(True, linestyle='--', alpha=0.7)

        image_url = save_plot(fig, 'elbow')
        return jsonify({'image_url': image_url})
    except Exception as e:
        logger.error(f"Error in generate_elbow: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clustering/dendrogram', methods=['POST'])
def generate_dendrogram_route():
    """Generate dendrogram visualization for hierarchical clustering"""
    global global_df, current_filepath
    
    data = request.json
    path = data.get('path')
    
    # If the path is different from current_filepath, load the data
    if path != current_filepath or global_df is None:
        try:
            global_df = pd.read_csv(path)
            current_filepath = path
        except Exception as e:
            logger.error(f"Error in generate_dendrogram_route: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    method = data.get('method', 'ward')
    max_clusters = data.get('max_clusters', 5)

    try:
        Z = linkage(global_df, method=method)

        fig = plt.figure(figsize=(12, 6))
        dendrogram(Z, truncate_mode='lastp', p=max_clusters)
        plt.title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} linkage)')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.grid(True, linestyle='--', alpha=0.7)

        image_url = save_plot(fig, 'dendrogram')
        return jsonify({'image_url': image_url})
    except Exception as e:
        logger.error(f"Error in generate_dendrogram_route: {str(e)}")
        return jsonify({'error': str(e)}), 500


# @app.route('/clustering/run', methods=['POST'])
# def run_clustering():
#     """Run K-means, K-medoids, or Hierarchical clustering"""
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

#         # Create scatter plot visualization
#         fig = plt.figure(figsize=(10, 6))
#         if df.shape[1] >= 2:  # If we have at least 2 dimensions
#             plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
#             plt.title(f'{algorithm.upper()} Clustering Results')
#             plt.xlabel(df.columns[0])
#             plt.ylabel(df.columns[1])
#             plt.colorbar(label='Cluster')
#         else:
#             plt.text(0.5, 0.5, "Not enough dimensions to plot", 
#                     horizontalalignment='center', verticalalignment='center')
        
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
#     """Run DBSCAN density-based clustering"""
#     data = request.json
#     df = pd.read_csv(data['path'])

#     try:
#         model = DBSCAN(
#             eps=data['eps'],
#             min_samples=data['min_samples']
#         )
#         labels = model.fit_predict(df)

#         metrics = calculate_metrics(df, labels)

#         # Create scatter plot visualization
#         fig = plt.figure(figsize=(10, 6))
#         if df.shape[1] >= 2:  # If we have at least 2 dimensions
#             scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10')
#             plt.title('DBSCAN Clustering Results')
#             plt.xlabel(df.columns[0])
#             plt.ylabel(df.columns[1])
            
#             # Add legend to distinguish noise points (-1)
#             unique_labels = set(labels)
#             legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
#                               markerfacecolor=scatter.cmap(scatter.norm(label)), 
#                               markersize=10, label=f'Cluster {label}' if label >= 0 else 'Noise') 
#                               for label in unique_labels]
#             plt.legend(handles=legend_elements)
#         else:
#             plt.text(0.5, 0.5, "Not enough dimensions to plot", 
#                     horizontalalignment='center', verticalalignment='center')
        
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
#     """Compare metrics across different clustering algorithms"""
#     if not stored_results:
#         return jsonify({'error': 'No clustering results found yet'}), 400
    
#     # Create comparison visualization if multiple algorithms are available
#     if len(stored_results) > 1:
#         fig, ax = plt.subplots(figsize=(12, 6))
        
#         metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
#         algorithms = list(stored_results.keys())
        
#         x = np.arange(len(metrics))
#         width = 0.2
#         multiplier = 0
        
#         for algorithm, results in stored_results.items():
#             values = [results.get(metric) for metric in metrics]
#             offset = width * multiplier
#             rects = ax.bar(x + offset, values, width, label=algorithm.upper())
#             multiplier += 1
        
#         ax.set_ylabel('Score')
#         ax.set_title('Clustering Algorithm Metrics Comparison')
#         ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
#         ax.set_xticklabels(['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'])
#         ax.legend(loc='best')
        
#         comparison_url = save_plot(fig, 'comparison')
#         return jsonify({
#             'comparison': stored_results,
#             'plot_url': comparison_url
#         })
    
#     return jsonify({'comparison': stored_results})

# # @app.route('/static/<filename>')
# # def serve_static(filename):
# #     """Serve static files (images)"""
# #     return send_from_directory(app.config['STATIC_FOLDER'], filename)

# @app.route('/')
# def home():
#     """API root endpoint with documentation"""
#     return jsonify({
#         'message': 'Clustering API is running',
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

@app.route('/clustering/run', methods=['POST'])
def run_clustering():
    """Run K-means, K-medoids, or Hierarchical clustering"""
    data = request.json
    df = pd.read_csv(data['path'])
    algorithm = data['algorithm']

    try:
        # Get selected features or use all
        selected_features = data.get('features', df.columns.tolist())
        df_selected = df[selected_features] if selected_features else df
        
        if algorithm in ['kmeans', 'kmedoids']:
            k = data['k']
            
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42)
            else:  # kmedoids
                model = KMedoids(n_clusters=k, random_state=42)
                
            labels = model.fit_predict(df_selected)
            
            # For KMeans, we can get the centers
            if algorithm == 'kmeans':
                centers = model.cluster_centers_
                # Find the closest points to centers for KMedoids-like interpretation
                closest_points = []
                for i in range(k):
                    cluster_points = df_selected[labels == i]
                    if len(cluster_points) > 0:
                        distances = np.linalg.norm(cluster_points.values - centers[i], axis=1)
                        closest_idx = distances.argmin()
                        closest_points.append(cluster_points.iloc[closest_idx].to_dict())
            else:  # KMedoids has medoids
                centers = model.cluster_centers_
                closest_points = [df_selected.iloc[idx].to_dict() for idx in model.medoid_indices_]
        
        elif algorithm in ['agnes', 'diana']:
            model = AgglomerativeClustering(
                n_clusters=data['n_clusters'],
                linkage=data.get('method', 'ward')
            )
            labels = model.fit_predict(df_selected)
            
            # For hierarchical clustering, find the centroids of each cluster
            centers = []
            closest_points = []
            for i in range(data['n_clusters']):
                cluster_points = df_selected[labels == i]
                if len(cluster_points) > 0:
                    center = cluster_points.mean().to_numpy()
                    centers.append(center)
                    
                    # Find the closest point to the centroid
                    distances = np.linalg.norm(cluster_points.values - center, axis=1)
                    closest_idx = distances.argmin()
                    closest_points.append(cluster_points.iloc[closest_idx].to_dict())
        
        else:
            return jsonify({'error': 'Invalid algorithm in /clustering/run'}), 400

        # Calculate metrics on the selected features
        metrics = calculate_metrics(df_selected.values, labels)
        
        # Visualization using the best features or PCA
        viz_data, top_features = get_best_features_for_visualization(df_selected, labels)
        
        # Create scatter plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # First plot: Original feature visualization (or PCA if more than 2D)
        scatter = ax1.scatter(viz_data.iloc[:, 0], viz_data.iloc[:, 1], c=labels, cmap='tab10', alpha=0.7)
        
        # Plot the cluster centers
        if 'centers' in locals():
            if viz_data.columns[0] == 'PC1':  # If we used PCA
                # Transform centers to PCA space for plotting
                pca = PCA(n_components=2)
                pca.fit(df_selected)
                if algorithm in ['kmeans', 'kmedoids']:
                    centers_2d = pca.transform(centers)
                else:
                    centers_2d = pca.transform(np.array(centers))
            else:
                centers_2d = np.array(centers)[:, :2] if len(centers) > 0 and len(centers[0]) >= 2 else []
            
            if len(centers_2d) > 0:
                ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], s=200, c='black', marker='X', alpha=0.7)
        
        ax1.set_title(f'{algorithm.upper()} Clustering Results')
        ax1.set_xlabel(viz_data.columns[0])
        ax1.set_ylabel(viz_data.columns[1])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Cluster')
        
        # Second plot: Feature distribution by cluster
        for i in set(labels):
            if i != -1:  # Skip noise points
                cluster_points = viz_data[labels == i]
                ax2.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f'Cluster {i}', alpha=0.7)
        
        ax2.set_title('Cluster Distribution')
        ax2.set_xlabel(viz_data.columns[0])
        ax2.set_ylabel(viz_data.columns[1])
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        plot_url = save_plot(fig, 'cluster')
        
        # Create cluster distribution plot
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_names = [f'Cluster {i}' for i in cluster_counts.index]
        if -1 in cluster_counts.index:
            cluster_names[cluster_counts.index.get_loc(-1)] = 'Noise'
            
        ax.bar(cluster_names, cluster_counts.values, color='skyblue')
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels on top of bars
        for i, v in enumerate(cluster_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        distribution_plot = save_plot(fig2, 'distribution')
        
        # Save for comparison
        stored_results[algorithm] = metrics
        
        # Add cluster assignments to original dataframe for download
        df_result = df.copy()
        df_result['cluster'] = labels
        result_path = os.path.join(app.config['STATIC_FOLDER'], f"{algorithm}_result_{uuid.uuid4().hex}.csv")
        df_result.to_csv(result_path, index=False)

        return jsonify({
            'message': f'{algorithm.upper()} clustering complete',
            'performance': metrics,
            'plot_url': plot_url,
            'distribution_plot': distribution_plot,
            'result_file': f"/static/{os.path.basename(result_path)}",
            'cluster_centers': closest_points if 'closest_points' in locals() else [],
            'top_features': top_features
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/clustering/dbscan', methods=['POST'])
def run_dbscan():
    """Run DBSCAN density-based clustering"""
    data = request.json
    df = pd.read_csv(data['path'])

    try:
        # Get selected features or use all
        selected_features = data.get('features', df.columns.tolist())
        df_selected = df[selected_features] if selected_features else df
        
        model = DBSCAN(
            eps=data['eps'],
            min_samples=data['min_samples']
        )
        labels = model.fit_predict(df_selected)

        metrics = calculate_metrics(df_selected.values, labels)
        
        # Visualization using the best features or PCA
        viz_data, top_features = get_best_features_for_visualization(df_selected, labels)
        
        # Create scatter plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Create a custom colormap that uses red for noise points (-1)
        import matplotlib.colors as mcolors
        cmap = plt.cm.tab10.copy()
        colors = [cmap(i) for i in range(10)]
        colors = [(0.8, 0, 0, 1)] + colors  # Red for noise, then regular colors
        custom_cmap = mcolors.ListedColormap(colors)
        
        # Adjust labels for colormap (shift by 1 to make noise -1 appear as 0)
        plot_labels = labels + 1
        
        # First plot
        scatter = ax1.scatter(viz_data.iloc[:, 0], viz_data.iloc[:, 1], c=plot_labels, cmap=custom_cmap, alpha=0.7)
        
        ax1.set_title('DBSCAN Clustering Results')
        ax1.set_xlabel(viz_data.columns[0])
        ax1.set_ylabel(viz_data.columns[1])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        unique_labels = set(labels)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=colors[label+1] if label >= 0 else colors[0], 
                            markersize=10, label=f'Cluster {label}' if label >= 0 else 'Noise') 
                            for label in sorted(unique_labels)]
        ax1.legend(handles=legend_elements, loc='best')
        
        # Second plot: Feature distribution by cluster
        for i in sorted(unique_labels):
            cluster_points = viz_data[labels == i]
            label = 'Noise' if i == -1 else f'Cluster {i}'
            color = 'red' if i == -1 else None
            ax2.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], 
                        label=label, alpha=0.7, color=color)
        
        ax2.set_title('Cluster Distribution')
        ax2.set_xlabel(viz_data.columns[0])
        ax2.set_ylabel(viz_data.columns[1])
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        plot_url = save_plot(fig, 'dbscan')
        
        # Create cluster distribution plot
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_names = [f'Cluster {i}' for i in cluster_counts.index]
        bar_colors = ['red' if i == -1 else 'skyblue' for i in cluster_counts.index]
        
        if -1 in cluster_counts.index:
            cluster_names[cluster_counts.index.get_loc(-1)] = 'Noise'
            
        ax.bar(cluster_names, cluster_counts.values, color=bar_colors)
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels on top of bars
        for i, v in enumerate(cluster_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        distribution_plot = save_plot(fig2, 'distribution')
        
        # Save for comparison
        stored_results['dbscan'] = metrics
        
        # Add cluster assignments to original dataframe for download
        df_result = df.copy()
        df_result['cluster'] = labels
        result_path = os.path.join(app.config['STATIC_FOLDER'], f"dbscan_result_{uuid.uuid4().hex}.csv")
        df_result.to_csv(result_path, index=False)
        
        # Calculate cluster centers for non-noise points
        closest_points = []
        for i in sorted(unique_labels):
            if i != -1:  # Skip noise points
                cluster_points = df_selected[labels == i]
                if len(cluster_points) > 0:
                    # Find the core points
                    core_sample_indices = model.core_sample_indices_
                    core_points = [idx for idx in core_sample_indices if labels[idx] == i]
                    
                    if core_points:
                        # Select a representative point (first core point)
                        rep_point = df_selected.iloc[core_points[0]].to_dict()
                        closest_points.append({
                            'cluster': int(i),
                            'representative_point': rep_point,
                            'is_core': True
                        })

        return jsonify({
            'message': 'DBSCAN clustering complete',
            'performance': metrics,
            'plot_url': plot_url,
            'distribution_plot': distribution_plot,
            'result_file': f"/static/{os.path.basename(result_path)}",
            'cluster_centers': closest_points,
            'top_features': top_features,
            'parameters': {
                'eps': data['eps'],
                'min_samples': data['min_samples'],
                'num_clusters': len([l for l in unique_labels if l != -1]),
                'noise_points': int(sum(labels == -1))
            }
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/clustering/comparison', methods=['GET'])
def get_comparison():
    """Compare metrics across different clustering algorithms"""
    if not stored_results:
        return jsonify({'error': 'No clustering results found yet'}), 400
    
    # Create comparison visualization if multiple algorithms are available
    if len(stored_results) > 1:
        # Create comparison table
        comparison_data = {}
        for algorithm, metrics in stored_results.items():
            comparison_data[algorithm] = {
                'silhouette': metrics.get('silhouette', 0),
                'davies_bouldin': metrics.get('davies_bouldin', 0),
                'calinski_harabasz': metrics.get('calinski_harabasz', 0),
                'num_clusters': metrics.get('num_clusters', 0)
            }
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        titles = ['Silhouette Score (higher is better)', 
                 'Davies-Bouldin Score (lower is better)', 
                 'Calinski-Harabasz Score (higher is better)']
        
        algorithms = list(stored_results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            values = [comparison_data[alg][metric] for alg in algorithms]
            
            # For Davies-Bouldin, lower is better, so invert the bars
            if metric == 'davies_bouldin':
                max_val = max(values) * 1.2
                values = [max_val - v for v in values]
                axes[i].set_ylabel('Inverted Score (higher is better)')
            else:
                axes[i].set_ylabel('Score')
                
            axes[i].bar(algorithms, values, color=colors)
            axes[i].set_title(title)
            axes[i].set_xticklabels([a.upper() for a in algorithms], rotation=45)
            
            # Add values on top of bars
            for j, v in enumerate(values):
                original_value = comparison_data[algorithms[j]][metric]
                axes[i].text(j, v * 1.05, f'{original_value:.3f}', ha='center')
        
        plt.tight_layout()
        comparison_url = save_plot(fig, 'comparison')
        
        # Create radar chart for overall comparison
        # Using UnivariateSpline from scipy as requested
        from matplotlib.path import Path
        from scipy.interpolate import UnivariateSpline
        from matplotlib.patches import PathPatch
        
        fig2 = plt.figure(figsize=(10, 10))
        ax = fig2.add_subplot(111, polar=True)
        
        # Define the categories and normalized values
        categories = ['Silhouette\n(higher is better)', 
                      'Davies-Bouldin\n(lower is better)', 
                      'Calinski-Harabasz\n(higher is better)']
        
        # Normalize scores between 0 and 1
        silhouette_values = [d['silhouette'] for d in comparison_data.values()]
        db_values = [d['davies_bouldin'] for d in comparison_data.values()]
        ch_values = [d['calinski_harabasz'] for d in comparison_data.values()]
        
        # Normalize
        def normalize(values):
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return [0.5 for _ in values]
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        norm_silhouette = normalize(silhouette_values)
        # For Davies-Bouldin, lower is better, so invert
        norm_db = [1 - v for v in normalize(db_values)]
        norm_ch = normalize(ch_values)
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw the radar chart for each algorithm
        for i, algorithm in enumerate(algorithms):
            values = [norm_silhouette[i], norm_db[i], norm_ch[i]]
            values += values[:1]  # Close the loop
            
            # Draw the shape
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=algorithm.upper(), color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Clustering Algorithm Performance Comparison')
        radar_plot = save_plot(fig2, 'radar')
        
        return jsonify({
            'comparison': comparison_data,
            'plot_url': comparison_url,
            'radar_plot': radar_plot
        })
    
    return jsonify({'comparison': stored_results})

@app.route('/static/<filename>')
def serve_static(filename):
    """Serve static files (images)"""
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/')
def home():
    """API root endpoint with documentation"""
    return jsonify({
        'message': 'Enhanced Clustering API is running',
        'endpoints': {
            'upload': '/upload-data (POST) - Upload dataset CSV',
            'clustering': [
                '/clustering/select (POST) - Select best features for clustering',
                '/clustering/elbow (POST) - Run elbow method to find optimal K',
                '/clustering/dendrogram (POST) - Create hierarchical clustering dendrogram',
                '/clustering/run (POST) - Run K-means, K-medoids, or Hierarchical clustering',
                '/clustering/dbscan (POST) - Run DBSCAN density-based clustering',
                '/clustering/comparison (GET) - Compare metrics across algorithms'
            ]
        },
        'version': '2.0'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
