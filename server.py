"""
Face Recognition Server for Render
This server uses dlib for face recognition and connects to MongoDB
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import dlib
import numpy as np
import cv2
import os
import base64
import logging
from pymongo import MongoClient
import datetime
import json
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for Android app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Base directory: {BASE_DIR}")

# Check for models in data/data_dlib folder
models_dir = os.path.join(BASE_DIR, 'data', 'data_dlib')
predictor_path = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
face_reco_model_path = os.path.join(models_dir, 'dlib_face_recognition_resnet_model_v1.dat')

logger.info(f"Looking for models in: {models_dir}")

if not os.path.exists(predictor_path):
    logger.error(f"Shape predictor not found at: {predictor_path}")
    # Also check in current directory as fallback
    alt_predictor = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
    if os.path.exists(alt_predictor):
        logger.info(f"Found shape predictor at: {alt_predictor}")
        predictor_path = alt_predictor
    else:
        raise FileNotFoundError(f"shape_predictor_68_face_landmarks.dat not found in {models_dir} or {BASE_DIR}")

if not os.path.exists(face_reco_model_path):
    logger.error(f"Face recognition model not found at: {face_reco_model_path}")
    # Also check in current directory as fallback
    alt_model = os.path.join(BASE_DIR, 'dlib_face_recognition_resnet_model_v1.dat')
    if os.path.exists(alt_model):
        logger.info(f"Found face recognition model at: {alt_model}")
        face_reco_model_path = alt_model
    else:
        raise FileNotFoundError(f"dlib_face_recognition_resnet_model_v1.dat not found in {models_dir} or {BASE_DIR}")

logger.info(f"Using predictor: {predictor_path}")
logger.info(f"Using model: {face_reco_model_path}")

# Initialize dlib models
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)
    logger.info("✅ Dlib models loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading dlib models: {e}")
    raise

# MongoDB connection
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb+srv://hiraiginjialt4_db_user:l2CDwnX4pQdcEvI7@cluster0.sjtehir.mongodb.net/attendance_system?retryWrites=true&w=majority')

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    mongo_client.admin.command('ping')
    db = mongo_client["attendance_system"]
    employee_faces_collection = db["employee_faces"]
    attendance_collection = db["attendance_records"]
    logger.info("✅ Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    # Continue without MongoDB - will use local memory
    employee_faces_collection = None
    attendance_collection = None

# Load known faces into memory
face_features_known = []
face_names_known = []
last_load_time = None

def load_known_faces():
    """Load all known faces from MongoDB into memory"""
    global face_features_known, face_names_known, last_load_time
    
    if employee_faces_collection is None:
        logger.warning("⚠️ MongoDB not available, using empty face database")
        face_features_known = []
        face_names_known = []
        return 0
    
    try:
        employees = employee_faces_collection.find({})
        count = 0
        temp_features = []
        temp_names = []
        
        for emp in employees:
            # Try different possible field names
            name = emp.get('name') or emp.get('employee_name') or emp.get('full_name') or 'Unknown'
            features = emp.get('features') or emp.get('face_features') or []
            
            if features and len(features) == 128:
                # Convert to float array
                feature_array = [float(f) for f in features]
                temp_names.append(name)
                temp_features.append(feature_array)
                count += 1
                logger.info(f"  Loaded: {name}")
            else:
                logger.warning(f"  Invalid features for {name}: {len(features)} features")
        
        face_features_known = temp_features
        face_names_known = temp_names
        last_load_time = datetime.datetime.now()
        
        logger.info(f"✅ Successfully loaded {count} known faces")
        return count
        
    except Exception as e:
        logger.error(f"❌ Error loading faces: {e}")
        return 0

# Load faces on startup
load_known_faces()

def return_euclidean_distance(feature_1, feature_2):
    """Calculate Euclidean distance between two face features"""
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return float(dist)

def process_image(base64_image):
    """Convert base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_image)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return None
        
        return img
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "status": "online",
        "service": "Face Recognition Server",
        "faces_loaded": len(face_names_known),
        "mongodb": employee_faces_collection is not None,
        "models_loaded": True
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "faces_loaded": len(face_names_known),
        "timestamp": datetime.datetime.now().isoformat(),
        "mongodb": employee_faces_collection is not None
    })

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Main face recognition endpoint"""
    try:
        # Get image from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Process image
        img = process_image(data['image'])
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Detect faces
        faces = detector(img, 1)  # Upsample once for better detection
        
        if len(faces) == 0:
            return jsonify({
                "faces_detected": 0,
                "matches": [],
                "message": "No faces detected"
            })
        
        results = []
        for i, face in enumerate(faces):
            try:
                # Get face landmarks and descriptor
                shape = predictor(img, face)
                face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
                
                # Compare with known faces
                best_match = None
                min_distance = float('inf')
                
                for j, known_features in enumerate(face_features_known):
                    distance = return_euclidean_distance(face_descriptor, known_features)
                    if distance < min_distance:
                        min_distance = distance
                        if distance < 0.4:  # Threshold for recognition
                            best_match = face_names_known[j]
                
                # Get face location
                face_location = {
                    "left": int(face.left()),
                    "top": int(face.top()),
                    "right": int(face.right()),
                    "bottom": int(face.bottom()),
                    "width": int(face.width()),
                    "height": int(face.height())
                }
                
                confidence = 1 - (min_distance / 0.8) if min_distance < 0.8 else 0
                
                results.append({
                    "face_id": i + 1,
                    "name": best_match if best_match else "unknown",
                    "confidence": round(max(0, confidence), 3),
                    "distance": round(min_distance, 3),
                    "location": face_location
                })
                
                logger.info(f"Face {i+1}: {results[-1]['name']} (confidence: {results[-1]['confidence']})")
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        return jsonify({
            "faces_detected": len(results),
            "matches": results,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/attendance', methods=['POST'])
def record_attendance():
    """Record attendance in database"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        name = data.get('name')
        action = data.get('action', 'clock_in')
        timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        # Store in MongoDB if available
        if attendance_collection is not None:
            attendance_record = {
                "name": name,
                "action": action,
                "timestamp": timestamp,
                "source": "android_app",
                "created_at": datetime.datetime.now()
            }
            
            result = attendance_collection.insert_one(attendance_record)
            record_id = str(result.inserted_id)
            logger.info(f"✅ Attendance recorded in MongoDB: {name} - {action}")
        else:
            record_id = "local_only"
            logger.info(f"📝 Attendance recorded locally: {name} - {action}")
        
        return jsonify({
            "status": "success",
            "message": f"{action} recorded for {name}",
            "record_id": record_id
        })
        
    except Exception as e:
        logger.error(f"Error recording attendance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    """Reload known faces from database"""
    try:
        count = load_known_faces()
        return jsonify({
            "status": "success",
            "faces_loaded": count,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error reloading faces: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/faces', methods=['GET'])
def list_faces():
    """List all known faces"""
    return jsonify({
        "total": len(face_names_known),
        "faces": face_names_known,
        "last_load": last_load_time.isoformat() if last_load_time else None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)