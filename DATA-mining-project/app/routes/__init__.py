from flask import Blueprint, jsonify, current_app

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return jsonify({
        'message': 'API is running',
        'endpoints': {
            'data': '/upload-data (POST)',
            'clustering': [
                '/clustering/select (POST)',
                '/clustering/elbow (POST)',
                '/clustering/dendrogram (POST)',
                '/clustering/run (POST)',
                '/clustering/dbscan (POST)',
                '/clustering/comparison (GET)'
            ],
            'plotting': [
                '/plot/scatter (POST)',
                '/plot/box (POST)'
            ]
        }
    })
