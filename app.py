import os
import uuid
import json
import shutil
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google.oauth2 import service_account
import ocr_logic  # Your OCR logic module

# Load environment variables
load_dotenv(override=True)

app = Flask(__name__)

# --- Configuration ---
# Use a dedicated folder for temporary processing files
PROCESSING_FOLDER = 'processing'
UPLOAD_FOLDER = 'uploads'
os.makedirs(PROCESSING_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# No longer need Flask-Session, as we are managing state manually.
# app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# --- Credentials ---
GOOGLE_KEY_PATH = os.getenv('GOOGLE_CREDENTIALS_JSON')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
try:
    google_credentials = service_account.Credentials.from_service_account_file(GOOGLE_KEY_PATH)
except Exception as e:
    print(f"CRITICAL: Failed to load Google credentials. The application may not function. Error: {e}")
    google_credentials = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    """
    Initializes a new processing session by creating a unique directory.
    """
    session_id = str(uuid.uuid4())
    session_path = os.path.join(PROCESSING_FOLDER, session_id)
    os.makedirs(session_path, exist_ok=True)
    print(f"Started new session: {session_id}")
    return jsonify({"session_id": session_id})

def is_session_valid(session_id):
    """
    Checks if a session directory exists.
    """
    if not session_id or '..' in session_id or '/' in session_id: # Basic security check
        return False
    return os.path.isdir(os.path.join(PROCESSING_FOLDER, session_id))

@app.route('/upload_and_process_file', methods=['POST'])
def upload_and_process_file():
    """
    Processes a single file and saves its extracted data to a file in the session directory.
    """
    session_id = request.form.get('session_id')
    if not is_session_valid(session_id):
        return jsonify({"error": "Invalid or expired session ID"}), 400

    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    file.save(filepath)

    try:
        if not google_credentials:
            raise ConnectionError("Google credentials are not loaded.")
            
        # Stage 1: OCR and AI processing
        page_outputs = ocr_logic.run_stage1_for_file(filepath, google_credentials, OPENAI_API_KEY)

        # Save the result to its own JSON file within the session directory
        if page_outputs:
            result_filename = f"{os.path.splitext(filename)[0]}.json"
            result_filepath = os.path.join(PROCESSING_FOLDER, session_id, result_filename)
            with open(result_filepath, 'w') as f:
                json.dump(page_outputs, f)

        print(f"Processed file {filename} for session {session_id}")
        return jsonify({"success": True, "file": filename})

    except Exception as e:
        # Provide a more specific error message to the frontend
        return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500
    finally:
        # Clean up the uploaded file immediately after processing
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/aggregate_results', methods=['POST'])
def aggregate_results():
    """
    Aggregates all individual file results from a session directory.
    """
    data = request.get_json()
    session_id = data.get('session_id')
    entity_filter = data.get('entity_filter')

    if not is_session_valid(session_id):
        return jsonify({"error": "Invalid or expired session ID"}), 400
    
    session_path = os.path.join(PROCESSING_FOLDER, session_id)

    try:
        all_page_outputs = []
        # Read all the .json files from the session directory
        for json_file in os.listdir(session_path):
            if json_file.endswith('.json'):
                with open(os.path.join(session_path, json_file), 'r') as f:
                    all_page_outputs.extend(json.load(f))
        
        if not all_page_outputs:
            return jsonify({"error": "No data was extracted from the provided documents."}), 400

        # Stage 2: Aggregation logic
        final_report = ocr_logic.run_stage2_aggregation(all_page_outputs, OPENAI_API_KEY, entity_filter)

        return jsonify(final_report)

    except Exception as e:
        return jsonify({"error": f"Failed to aggregate results: {str(e)}"}), 500
    finally:
        # Clean up the entire session directory after aggregation
        if os.path.isdir(session_path):
            shutil.rmtree(session_path)
            print(f"Session {session_id} cleaned up.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
