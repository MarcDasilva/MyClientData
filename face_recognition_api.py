from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import numpy as np
import cv2
import dlib
import pickle
from pymongo import MongoClient
import os

# Set up absolute paths for the images folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Initialize FastAPI
app = FastAPI()

# Load Face Detector and Recognizer
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["face_db"]
collection = db["faces"]

# Function to get face embeddings
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    shape = sp(gray, faces[0])
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# API to register a new face with additional info
@app.post("/register")
async def register_face(
    name: str = Form(...), 
    info: str = Form(...),
    file: UploadFile = File(...)
):
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    embedding = get_face_embedding(image)
    if embedding is None:
        return {"error": "No face detected"}
    
    # Save image to disk in the "images" folder.
    # We use a simple filename based on the provided name.
    filename = f"{name}.jpg"
    image_path = os.path.join("images", filename)  # stored relative path
    full_path = os.path.join(IMAGES_DIR, filename)   # full absolute path
    cv2.imwrite(full_path, image)

    # Store data in MongoDB.
    collection.insert_one({
        "name": name, 
        "info": info,
        "embedding": pickle.dumps(embedding),
        "imageName": image_path  # save relative path (e.g., "images/John.jpg")
    })
    
    return {"message": f"Face registered for {name}", "info": info}

# API to recognize a face
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    new_embedding = get_face_embedding(image)
    if new_embedding is None:
        return {"error": "No face detected"}
    
    recognized_name = "Unknown"
    recognized_info = "No additional info"
    recognized_image = "default.jpg"
    min_distance = float("inf")
    
    for doc in collection.find():
        stored_embedding = pickle.loads(doc["embedding"])
        distance = np.linalg.norm(new_embedding - stored_embedding)
        if distance < 0.6 and distance < min_distance:
            min_distance = distance
            recognized_name = doc["name"]
            recognized_info = doc.get("info", "No additional info")
            recognized_image = doc.get("imageName", "default.jpg")

    return {
        "name": recognized_name, 
        "info": recognized_info,
        "image_name": recognized_image
    }

# API to list all registered faces
@app.get("/list_entries")
async def list_entries():
    entries = []
    for doc in collection.find():
        entries.append({
            "id": str(doc["_id"]),
            "name": doc["name"],
            "info": doc.get("info", "No info available"),
            "imageName": doc.get("imageName", "default.jpg")
        })
    return entries

# Endpoint to serve images
@app.get("/get_image/{image_path:path}")
async def get_image(image_path: str):
    # If the path starts with "images/", remove it.
    if image_path.startswith("images/"):
        image_path = image_path[len("images/"):]
    file_path = os.path.join(IMAGES_DIR, image_path)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    else:
        return {"error": "Image not found"}
