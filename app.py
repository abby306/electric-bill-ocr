import os
import json # <--- Add this import
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google.oauth2 import service_account
import ocr_logic

# Load environment variables
load_dotenv(override=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODIFIED CREDENTIALS LOADING ---
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not GOOGLE_CREDENTIALS_JSON:
    raise ValueError("Missing GOOGLE_CREDENTIALS_JSON environment variable.")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

# Load credentials from the JSON string
try:
    google_credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)
    google_credentials = service_account.Credentials.from_service_account_info(google_credentials_info)
except json.JSONDecodeError:
    raise ValueError("Could not decode GOOGLE_CREDENTIALS_JSON. Make sure it's a valid JSON string.")

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and dispatch to the V3 processing logic."""
    entity_filter = request.form.get('entity_filter', '').strip() or None
    files = request.files.getlist('file')

    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    filepaths = []
    try:
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filepaths.append(filepath)

        # --- NEW V3 LOGIC ---
        # All requests now go through the same advanced V3 pipeline.
        result = ocr_logic.process_documents_v3(filepaths, google_credentials, OPENAI_API_KEY, entity_filter)
        
        return jsonify(result)

    except Exception as e:
        # Provide a more user-friendly error
        return jsonify({"error": f"An error occurred during processing: {e}"}), 500
    finally:
        # Clean up all temporary files
        for fp in filepaths:
            if os.path.exists(fp):
                os.remove(fp)

if __name__ == '__main__':
    app.run(debug=True)
