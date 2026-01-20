from flask import Flask, send_from_directory
from flask_cors import CORS
from app.config import Config
from app.routes.data import data_bp
from app.routes.plotting import plotting_bp
from app.routes.clustering import clustering_bp
from app.routes import main_bp
import os

def create_app(config_class=Config):
    # Disable default static handler to avoid conflict with our custom route
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_class)
    
    # Initialize CORS
    CORS(app)
    
    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(plotting_bp)
    app.register_blueprint(clustering_bp)
    
    # Explicitly serve static files from the configured STATIC_FOLDER
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        static_folder = app.config.get('STATIC_FOLDER')
        full_path = os.path.join(static_folder, filename)
        
        if not os.path.exists(full_path):
            app.logger.error(f"File not found: {full_path}")
            return "File not found", 404
            
        return send_from_directory(static_folder, filename)
    
    return app
