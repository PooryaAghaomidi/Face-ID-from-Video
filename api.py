import os
import yaml
import logging
import tempfile
from typing import Optional
from face_sim import FaceIdentification
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global variable for the FaceIdentification instance
face_identifier = None


@app.route('/initialize', methods=['POST'])
def initialize():
    """
    Initializes the FaceIdentification system.
    Optionally accepts JSON config overrides.
    """
    global face_identifier

    if request.is_json:
        params = request.get_json()
    else:
        params = {}

    try:
        face_identifier = FaceIdentification(**params)
        return jsonify({'message': 'FaceIdentification initialized successfully.'})
    except Exception as e:
        return jsonify({'error': f'Failed to initialize: {str(e)}'}), 500


@app.route('/update_db', methods=['POST'])
def update_db():
    global face_identifier
    if face_identifier is None:
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    image_paths = []

    if 'images' in request.files:
        files = request.files.getlist('images')
        for f in files:
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)
            image_paths.append(file_path)

    if request.is_json:
        data = request.get_json()
        if 'image_paths' in data:
            image_paths.extend(data['image_paths'])

    if not image_paths:
        return jsonify({'error': 'No images provided.'}), 400

    face_identifier.update_db(image_paths)
    return jsonify({'message': f'{len(image_paths)} images processed and added.'})


@app.route('/get_identities', methods=['GET'])
def get_identities():
    global face_identifier
    if face_identifier is None:
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    identities = face_identifier.get_identities()
    return jsonify({'identities': identities})


@app.route('/remove_identity', methods=['POST'])
def remove_identity():
    global face_identifier
    if face_identifier is None:
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    data = request.get_json()
    identity = data.get('identity') if data else None

    face_identifier.remove_identity(identity)
    return jsonify({'message': f'Identity "{identity}" removed.' if identity else 'All identities removed.'})


@app.route('/face_id', methods=['POST'])
def face_id():
    global face_identifier

    if face_identifier is None:
        logging.error("Face identifier not initialized. Call /initialize first.")
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    video_path: Optional[str] = None

    if 'video' in request.files:
        video_file = request.files['video']

        if video_file.filename == '':
            logging.warning("Empty video file uploaded.")
            return jsonify({'error': 'No video file selected.'}), 400
        
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

    elif request.is_json:
        data = request.get_json()
        video_path = data.get('video_path')

        if not video_path:
            logging.warning("No video_path provided in JSON payload.")
            return jsonify({'error': 'No video_path provided in JSON.'}), 400

    if not video_path or not os.path.exists(video_path):
        logging.error(f"Invalid video path: {video_path}")
        return jsonify({'error': 'No valid video provided.'}), 400

    try:
        result = face_identifier.face_id(video_path)
        logging.info(
            f"Face ID processed successfully. "
            f"Time: {result['time']}, Matches: {len(result['image_path'])}"
        )
        return jsonify({
            "matches": result["image_path"],
            "time": result["time"]
            })
    except Exception as e:
        logging.error(f"Face ID processing failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during face identification.'}), 500


@app.route('/')
def index():
    return "Face Identification API is running. Initialize it with /initialize."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config["port"])
