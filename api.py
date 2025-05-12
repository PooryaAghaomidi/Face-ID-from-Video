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


@app.route('/get_users', methods=['GET'])
def get_users():
    global face_identifier
    if face_identifier is None:
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    users = face_identifier.get_users()
    return jsonify({'users': users})


@app.route('/remove_identity', methods=['POST'])
def remove_identity():
    global face_identifier
    if face_identifier is None:
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    data = request.get_json()
    identity = data.get('identity') if data else None

    face_identifier.remove_identity(identity)
    return jsonify({'message': f'Identity "{identity}" removed.' if identity else 'All identities removed.'})


@app.route('/store_faces', methods=['POST'])
def store_faces():
    global face_identifier

    if face_identifier is None:
        logging.error("Face identifier not initialized. Call /initialize first.")
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    if 'user_id' not in request.form:
        return jsonify({'error': 'user_id is required in form data.'}), 400

    user_id = request.form['user_id']

    if 'video' not in request.files or request.files['video'].filename == '':
        logging.warning("No video file provided.")
        return jsonify({'error': 'No video file provided.'}), 400

    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    success = face_identifier.store_video(video_path, user_id)
    if success:
        return jsonify({'message': f'Frames stored for user {user_id}.'}), 200
    else:
        return jsonify({'error': 'Failed to store frames.'}), 500


@app.route('/match_faces/<user_id>', methods=['GET'])
def match_faces(user_id):
    global face_identifier

    if face_identifier is None:
        logging.error("Face identifier not initialized. Call /initialize first.")
        return jsonify({'error': 'System not initialized. Call /initialize first.'}), 400

    try:
        result = face_identifier.face_id(user_id)
        logging.info(
            f"Face match completed for user '{user_id}'. "
            f"Time: {result['time']}, Matches: {len(result['image_path'])}"
        )
        return jsonify({
            "matches": result["image_path"],
            "time": result["time"]
        })
    except Exception as e:
        logging.error(f"Error during face match: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error during face matching.'}), 500


@app.route('/')
def index():
    return "Face Identification API is running. Initialize it with /initialize."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config["port"])
