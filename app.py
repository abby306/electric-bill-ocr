import os
import uuid
import json # <-- ADDED: Required for parsing the JSON string
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google.oauth2 import service_account
import ocr_logic # Your refactored V6 functions

# Load environment variables
load_dotenv(override=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- REVERTED TO OLD CREDENTIAL LOADING METHOD ---
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
# --- END OF REVERTED BLOCK ---

# In-memory session storage (for demonstration purposes)
SESSIONS = {}

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

# NEW: Endpoint to start a session for a new batch of uploads.
@app.route('/start_session', methods=['POST'])
def start_session():
    """Initializes a new session to store intermediate results."""
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = []
    print(f"Started new session: {session_id}")
    return jsonify({"session_id": session_id})

# NEW: Endpoint to process one file at a time, preventing timeouts.
@app.route('/upload_and_process_file', methods=['POST'])
def upload_and_process_file():
    """Processes a single file and adds its Stage 1 results to the session."""
    session_id = request.form.get('session_id')
    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "Invalid or expired session ID"}), 400

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run Stage 1 processing for the single file
        # This now uses the 'google_credentials' object created with the old method
        page_outputs = ocr_logic.run_stage1_for_file(filepath, google_credentials, OPENAI_API_KEY)
        
        # Add the extracted data to the session
        SESSIONS[session_id].extend(page_outputs)
        
        print(f"Processed file {filename} for session {session_id}")
        return jsonify({"success": True, "file": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

# NEW: Endpoint to aggregate results after all files are processed.
@app.route('/aggregate_results', methods=['POST'])
def aggregate_results():
    """Aggregates all results from a session and returns the final report."""
    data = request.get_json()
    session_id = data.get('session_id')
    entity_filter = data.get('entity_filter')

    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "Invalid or expired session ID"}), 400

    try:
        all_page_outputs = SESSIONS[session_id]
        if not all_page_outputs:
             return jsonify({"error": "No data was extracted to aggregate."}), 400

        # Run Stage 2 aggregation
        final_report = ocr_logic.run_stage2_aggregation(all_page_outputs, OPENAI_API_KEY, entity_filter)
        
        return jsonify(final_report)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the session after aggregation
        if session_id in SESSIONS:
            del SESSIONS[session_id]
            print(f"Session {session_id} cleaned up.")

if __name__ == '__main__':
    app.run(debug=True)
