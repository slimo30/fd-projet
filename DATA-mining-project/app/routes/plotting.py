from flask import Blueprint, request, jsonify, current_app
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import uuid
import os
from app.state import state
from app.utils.helpers import is_safe_path, save_plot

plotting_bp = Blueprint('plotting', __name__)

@plotting_bp.route('/plot/scatter', methods=['POST'])
def generate_scatter_plot():
    data = request.get_json()
    path = data.get('path')
    x_col = data.get('x')
    y_col = data.get('y')
    
    if not path or not x_col or not y_col or not is_safe_path(current_app.config['UPLOAD_FOLDER'], path):
        return jsonify({'error': 'Invalid request'}), 400
    
    if path != state.current_filepath or state.df is None:
        try:
            state.df = pd.read_csv(path)
            state.current_filepath = path
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        fig = plt.figure()
        plt.scatter(state.df[x_col], state.df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'scatter')
        return jsonify({'plot_url': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@plotting_bp.route('/plot/box', methods=['POST'])
def generate_box_plot():
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
        fig = plt.figure()
        state.df[columns].boxplot()
        plt.title('Box Plot')
        
        plot_url = save_plot(fig, current_app.config['STATIC_FOLDER'], 'box')
        return jsonify({'plot_url': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
