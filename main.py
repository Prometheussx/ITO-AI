import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import face_recognition
import re
from datetime import datetime

app = Flask(__name__, static_folder='.')

# Define faces directory
FACES_DIR = 'faces'

# Ensure faces directory exists
os.makedirs(FACES_DIR, exist_ok=True)

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load all face images from the faces directory and compute their encodings."""
    global known_face_encodings, known_face_names
    
    # Clear existing data
    known_face_encodings = []
    known_face_names = []
    
    # List all files in the faces directory
    print("Loading known faces...")
    
    # Supported image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    for filename in os.listdir(FACES_DIR):
        # Check if the file is an image
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in valid_extensions:
            continue
            
        # Extract the person's name from the filename (remove extension)
        name = os.path.splitext(filename)[0]
        
        # Load the image
        filepath = os.path.join(FACES_DIR, filename)
        face_image = face_recognition.load_image_file(filepath)
        
        # Try to find faces in the image
        face_encodings = face_recognition.face_encodings(face_image)
        
        # If at least one face is found, use the first one
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)
            print(f"Loaded face: {name}")
        else:
            print(f"Warning: No face found in {filename}")
    
    print(f"Loaded {len(known_face_encodings)} faces in total")

# Load known faces at startup
load_known_faces()

@app.route('/')
def index():
    """Serve the index.html file."""
    return send_from_directory('.', 'index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a frame from the webcam to recognize faces."""
    # Get the image data from the request
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'found': False, 'message': 'No image data provided'})
    
    # Extract the base64 encoded image
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image_bytes = base64.b64decode(image_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert from BGR to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Using HOG for faster performance
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # If no faces are found
    if not face_encodings:
        return jsonify({'found': False, 'message': 'No faces detected'})
    
    # Check if we have any known faces loaded
    if not known_face_encodings:
        return jsonify({'found': False, 'message': 'No known faces loaded'})
    
    # Process each detected face
    best_match = None
    best_match_name = "Unknown"
    best_match_confidence = 0
    
    for face_encoding in face_encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        if len(face_distances) > 0 and any(matches):
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                # Calculate confidence as percentage (lower distance = higher confidence)
                confidence = (1 - face_distances[best_match_index]) * 100
                
                if confidence > best_match_confidence:
                    best_match_confidence = confidence
                    best_match_name = known_face_names[best_match_index]
                    best_match = {
                        'name': best_match_name,
                        'confidence': round(best_match_confidence, 2)
                    }
    
    # Return the result
    if best_match:
        return jsonify({
            'found': True,
            'name': best_match['name'],
            'confidence': best_match['confidence']
        })
    else:
        return jsonify({'found': False, 'message': 'Face detected but not recognized'})

@app.route('/reload_faces', methods=['GET'])
def reload_faces():
    """Reload the known faces from the faces directory."""
    load_known_faces()
    return jsonify({'success': True, 'message': f'Loaded {len(known_face_encodings)} faces'})

if __name__ == '__main__':
    print("Starting Face Recognition Server...")
    print(f"Please place your face images in the '{FACES_DIR}' directory")
    print("Each image file should contain one face and be named after the person")
    print("Example: 'john_doe.jpg'")
    app.run(host='0.0.0.0', port=5000, debug=True)