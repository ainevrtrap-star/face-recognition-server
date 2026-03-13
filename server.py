from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import dlib
import cv2
import os
import base64
import datetime
from pymongo import MongoClient
import logging
import json
import pandas as pd

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[+] Server Base Directory: {BASE_DIR}")

# Initialize Dlib models with correct paths from your structure
detector = dlib.get_frontal_face_detector()

# Path to model files in your data/data_dlib structure
predictor_path = os.path.join(BASE_DIR, 'data', 'data_dlib', 'shape_predictor_68_face_landmarks.dat')
face_reco_model_path = os.path.join(BASE_DIR, 'data', 'data_dlib', 'dlib_face_recognition_resnet_model_v1.dat')
csv_features_path = os.path.join(BASE_DIR, 'data', 'features_all.csv')

print(f"[+] Looking for predictor at: {predictor_path}")
print(f"[+] Looking for model at: {face_reco_model_path}")
print(f"[+] Looking for CSV at: {csv_features_path}")

# Check if model files exist
if not os.path.exists(predictor_path):
    print(f"[-] ERROR: Predictor file not found at {predictor_path}")
    print("[+] Make sure shape_predictor_68_face_landmarks.dat is in data/data_dlib/")
    exit(1)

if not os.path.exists(face_reco_model_path):
    print(f"[-] ERROR: Model file not found at {face_reco_model_path}")
    print("[+] Make sure dlib_face_recognition_resnet_model_v1.dat is in data/data_dlib/")
    exit(1)

# Load the models
predictor = dlib.shape_predictor(predictor_path)
face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)

# MongoDB Connection
MONGO_URI = "mongodb+srv://hiraiginjialt4_db_user:l2CDwnX4pQdcEvI7@cluster0.sjtehir.mongodb.net/attendance_system?retryWrites=true&w=majority"
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    mongo_client.admin.command('ping')
    db = mongo_client["attendance_system"]
    employee_faces_collection = db["employee_faces"]
    attendance_collection = db["attendance_records"]
    print("[+] Connected to MongoDB successfully")
except Exception as e:
    print(f"[-] MongoDB connection failed: {e}")
    print("[+] Continuing with local CSV only")
    mongo_client = None
    employee_faces_collection = None
    attendance_collection = None

# Load known faces into memory on server start
face_features_known = []
face_names_known = []

def return_euclidean_distance(feature_1, feature_2):
    """Compute Euclidean distance between two feature vectors"""
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist

def load_from_csv():
    """Load face features from local CSV file"""
    global face_features_known, face_names_known
    
    try:
        if not os.path.exists(csv_features_path):
            print(f"[-] CSV file not found at {csv_features_path}")
            return 0
        
        print("[+] Loading face features from local CSV...")
        
        # Read CSV file
        csv_rd = pd.read_csv(csv_features_path, header=None)
        
        if csv_rd.shape[0] == 0:
            print("[-] CSV file has no rows")
            return 0
        
        loaded_count = 0
        for i in range(csv_rd.shape[0]):
            try:
                name = str(csv_rd.iloc[i][0])
                features_someone_arr = []
                
                # Check if name is valid
                if not name or pd.isna(name) or name == 'nan':
                    print(f"[-] Invalid name at row {i}")
                    continue
                
                # Extract 128 features
                valid_row = True
                for j in range(1, 129):
                    value = csv_rd.iloc[i][j]
                    if pd.isna(value) or value == '':
                        features_someone_arr.append(0.0)
                        valid_row = False
                    else:
                        try:
                            features_someone_arr.append(float(value))
                        except:
                            features_someone_arr.append(0.0)
                            valid_row = False
                
                if valid_row and len(features_someone_arr) == 128:
                    face_names_known.append(name)
                    face_features_known.append(features_someone_arr)
                    loaded_count += 1
                    print(f"[+] Loaded from CSV: {name}")
                else:
                    print(f"[-] Invalid features for {name} at row {i}")
                    
            except Exception as e:
                print(f"[-] Error processing row {i} in CSV: {e}")
        
        print(f"[+] Successfully loaded {loaded_count} faces from local CSV")
        return loaded_count
        
    except Exception as e:
        print(f"[-] Error loading from CSV: {e}")
        return 0

def load_from_mongodb():
    """Load face features from MongoDB"""
    global face_features_known, face_names_known
    
    if employee_faces_collection is None:
        return 0
    
    try:
        print("[+] Loading face features from MongoDB...")
        employee_records = employee_faces_collection.find({})
        
        mongo_count = 0
        for record in employee_records:
            try:
                name = record.get('name', 'Unknown')
                if not name or name == 'Unknown':
                    name = record.get('employee_name', 'Unknown')
                
                features = record.get('features', [])
                
                if features and len(features) == 128:
                    features_arr = [float(f) for f in features]
                    face_names_known.append(name)
                    face_features_known.append(features_arr)
                    mongo_count += 1
                    print(f"[+] Loaded from MongoDB: {name}")
                else:
                    print(f"[-] Invalid features for {name}")
            except Exception as e:
                print(f"[-] Error processing MongoDB record: {e}")
        
        print(f"[+] Successfully loaded {mongo_count} faces from MongoDB")
        return mongo_count
        
    except Exception as e:
        print(f"[-] Error loading from MongoDB: {e}")
        return 0

def load_known_faces():
    """Load all known faces from database (MongoDB first, then CSV fallback)"""
    global face_features_known, face_names_known
    face_features_known = []
    face_names_known = []
    
    total_loaded = 0
    
    # Try MongoDB first if available
    if employee_faces_collection is not None:
        total_loaded = load_from_mongodb()
    
    # If MongoDB failed or returned 0, try CSV
    if total_loaded == 0:
        print("[+] Falling back to local CSV...")
        total_loaded = load_from_csv()
    
    print(f"[+] Total faces loaded: {total_loaded}")
    return total_loaded

# Load faces on startup
load_known_faces()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "faces_loaded": len(face_names_known),
        "mongodb": "connected" if mongo_client else "disconnected",
        "faces": face_names_known  # List all loaded faces
    })

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Receive image from client and recognize faces"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 encoding: {e}"}), 400
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Convert to RGB (Dlib expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector(img_rgb, 0)
        
        if len(faces) == 0:
            return jsonify({"faces": []})
        
        results = []
        for face in faces:
            try:
                # Get face features
                shape = predictor(img_rgb, face)
                face_features = face_reco_model.compute_face_descriptor(img_rgb, shape)
                face_features_array = list(face_features)
                
                # Compare with known faces
                best_match = "Unknown"
                min_distance = 0.4  # Threshold - lower is stricter
                confidence = 0.0
                
                for i, known_features in enumerate(face_features_known):
                    distance = return_euclidean_distance(face_features_array, known_features)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = face_names_known[i]
                        confidence = 1.0 - distance  # Convert to confidence score
                
                # Get face coordinates
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                
                results.append({
                    "name": best_match,
                    "confidence": float(confidence),
                    "distance": float(min_distance),
                    "bbox": [left, top, right, bottom]
                })
                
            except Exception as e:
                print(f"[-] Error processing individual face: {e}")
                continue
        
        return jsonify({"faces": results})
    
    except Exception as e:
        logging.error(f"Recognition error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/attendance', methods=['POST'])
def record_attendance():
    """Record attendance from client"""
    try:
        data = request.json
        name = data.get('name')
        action = data.get('action')  # 'in' or 'out'
        
        if not name or not action:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get current time
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        date = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        # If MongoDB is available, use it
        if attendance_collection is not None:
            # Check if record exists for today
            existing = attendance_collection.find_one({
                "name": name,
                "date": date
            })
            
            if action == 'in':
                if existing:
                    return jsonify({
                        "message": "Already clocked in", 
                        "record": {
                            "name": name,
                            "clock_in_time": existing.get('clock_in_time'),
                            "date": date
                        }
                    }), 200
                else:
                    record = {
                        "name": name,
                        "clock_in_time": time_str,
                        "clock_out_time": None,
                        "date": date,
                        "timestamp": now,
                        "device_id": data.get('device_id', 'unknown'),
                        "location": data.get('location', 'unknown')
                    }
                    result = attendance_collection.insert_one(record)
                    record['_id'] = str(result.inserted_id)
                    return jsonify({
                        "message": "Clocked in successfully", 
                        "record": record
                    }), 201
            
            elif action == 'out':
                if existing:
                    attendance_collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {"clock_out_time": time_str}}
                    )
                    existing["clock_out_time"] = time_str
                    existing['_id'] = str(existing['_id'])
                    return jsonify({
                        "message": "Clocked out successfully", 
                        "record": existing
                    }), 200
                else:
                    return jsonify({"error": "No clock-in record found for today"}), 404
        
        # If MongoDB is not available, return success anyway (client can store locally)
        return jsonify({
            "message": f"Attendance {action} recorded (offline mode)",
            "name": name,
            "time": time_str,
            "date": date
        }), 200
    
    except Exception as e:
        logging.error(f"Attendance error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/faces/reload', methods=['POST'])
def reload_faces():
    """Reload known faces from database"""
    count = load_known_faces()
    return jsonify({"message": f"Reloaded {count} faces", "count": count})

@app.route('/faces/list', methods=['GET'])
def list_faces():
    """List all known faces"""
    return jsonify({
        "count": len(face_names_known),
        "faces": face_names_known
    })

@app.route('/faces/add', methods=['POST'])
def add_face():
    """Add a new face to the database"""
    try:
        data = request.json
        name = data.get('name')
        features = data.get('features')
        
        if not name or not features:
            return jsonify({"error": "Missing name or features"}), 400
        
        if len(features) != 128:
            return jsonify({"error": "Features must be 128-dimensional"}), 400
        
        # Add to MongoDB if available
        if employee_faces_collection is not None:
            # Check if face already exists
            existing = employee_faces_collection.find_one({"name": name})
            if existing:
                # Update existing
                employee_faces_collection.update_one(
                    {"name": name},
                    {"$set": {"features": features, "updated_at": datetime.datetime.now()}}
                )
                message = f"Updated face for {name}"
            else:
                # Insert new
                employee_faces_collection.insert_one({
                    "name": name,
                    "features": features,
                    "created_at": datetime.datetime.now()
                })
                message = f"Added face for {name}"
            
            # Reload faces
            load_known_faces()
            
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": "MongoDB not available"}), 503
            
    except Exception as e:
        logging.error(f"Add face error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)