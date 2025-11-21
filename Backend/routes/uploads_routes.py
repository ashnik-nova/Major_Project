from flask import Blueprint, request, jsonify
import os
import uuid
import threading
import time
from datetime import datetime, timedelta

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ------------- CENTRALIZED AUTO-CLEANUP SYSTEM -------------
_CLEANUP_REGISTRY = []  # List of (filepath, deletion_time)
_CLEANUP_LOCK = threading.Lock()

def schedule_file_deletion(filepath, delay_seconds=60):
    """
    Schedule a file for deletion after delay_seconds.
    
    Args:
        filepath: Full path to the file
        delay_seconds: Seconds to wait before deletion (default: 60)
    """
    deletion_time = datetime.now() + timedelta(seconds=delay_seconds)
    with _CLEANUP_LOCK:
        _CLEANUP_REGISTRY.append((filepath, deletion_time))
    print(f"[SCHEDULED] {filepath} will be deleted at {deletion_time.strftime('%H:%M:%S')}")

def cleanup_worker():
    """
    Background thread that periodically checks and deletes expired files.
    Runs every 10 seconds.
    """
    print("[CLEANUP WORKER] Started for uploads")
    while True:
        time.sleep(10)  # Check every 10 seconds
        now = datetime.now()
        with _CLEANUP_LOCK:
            to_delete = []
            remaining = []
            
            for filepath, deletion_time in _CLEANUP_REGISTRY:
                if now >= deletion_time:
                    to_delete.append(filepath)
                else:
                    remaining.append((filepath, deletion_time))
            
            _CLEANUP_REGISTRY[:] = remaining
        
        # Delete files outside the lock to avoid blocking
        for filepath in to_delete:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"[CLEANUP] ✓ Deleted: {filepath}")
            except Exception as e:
                print(f"[CLEANUP ERROR] ✗ Failed to delete {filepath}: {e}")

# Start cleanup thread when module loads
_cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
_cleanup_thread.start()

# ------------- HELPER FUNCTIONS -------------
def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------- UPLOAD ENDPOINT -------------
@upload_bp.route("/", methods=["POST"])
def upload_image():
    """
    Upload an image file with automatic deletion after 60 seconds.
    
    Form data:
        - file: Image file (PNG, JPG, JPEG)
        - delete_after (optional): Seconds to wait before auto-deletion (default: 60)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PNG, JPG, JPEG allowed"}), 400

    # Get custom deletion delay if provided (default: 60 seconds)
    delete_after = int(request.form.get('delete_after', 60))

    # Generate unique filename (prevents overwriting)
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Save locally
        file.save(filepath)

        # Schedule deletion using centralized cleanup system
        schedule_file_deletion(filepath, delay_seconds=delete_after)

        return jsonify({
            "message": "File uploaded successfully",
            "file_path": filepath,
            "filename": filename,
            "auto_delete_info": f"File will be automatically deleted after {delete_after} seconds"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@upload_bp.route("/status", methods=["GET"])
def cleanup_status():
    """
    Get the status of the cleanup system (for debugging).
    Returns number of files scheduled for deletion.
    """
    with _CLEANUP_LOCK:
        scheduled_count = len(_CLEANUP_REGISTRY)
        files_info = [
            {
                "file": os.path.basename(fp),
                "deletes_at": dt.strftime('%Y-%m-%d %H:%M:%S')
            }
            for fp, dt in _CLEANUP_REGISTRY
        ]
    
    return jsonify({
        "status": "active",
        "scheduled_deletions": scheduled_count,
        "files": files_info
    }), 200