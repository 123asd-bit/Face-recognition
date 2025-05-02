import face_recognition
import numpy as np
import os
from typing import List, Tuple, Optional
import shutil
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.cluster import DBSCAN, KMeans
import pickle
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response
import uvicorn
import io
import random
from scipy.cluster.hierarchy import linkage, fcluster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Face recognition parameters
SIMILARITY_THRESHOLD = 0.6  # Original value
CLUSTERING_EPS = 0.6  # Original value
FACE_MATCH_THRESHOLD = 0.6  # Original value
USE_CLUSTERING = True  # Enable clustering as original
USE_HIERARCHICAL = False  # Disable hierarchical clustering
MIN_FACE_SIZE = 10  # Smaller minimum size for faster detection
MIN_GROUP_SIZE = 1  # Allow single-face groups
HIGH_CONFIDENCE_THRESHOLD = 0.6  # Original value
MIN_SAMPLE_SIZE = 1  # Minimum sample size for comparison
MAX_SAMPLE_SIZE = 3  # Maximum sample size for comparison
GROUP_MERGE_THRESHOLD = 0.6  # Original value

# Quality thresholds for face detection
BLUR_THRESHOLD = 30  # Higher threshold to allow slightly blurrier human faces while still filtering very blurry images
MIN_FACE_WIDTH = 30   # Reduced minimum face width
MIN_FACE_HEIGHT = 30  # Reduced minimum face height
FACE_CONFIDENCE_THRESHOLD = 0.75  # Slightly reduced threshold for face detection model

# Initialize paths
FACES_DIR = "faces"
UPLOADS_DIR = "uploads"
STATIC_DIR = "static"
DB_PATH = "faces.db"

# Create necessary directories
UPLOAD_DIR = Path(UPLOADS_DIR)
FACES_DIR = Path(FACES_DIR)
STATIC_DIR = Path(STATIC_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins temporarily for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Drop existing tables if they exist
    c.execute('DROP TABLE IF EXISTS faces')
    c.execute('DROP TABLE IF EXISTS face_groups')
    c.execute('DROP TABLE IF EXISTS images')
    
    # Create tables with proper schema
    c.execute('''
        CREATE TABLE face_groups
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         name TEXT,
         created_at TIMESTAMP,
         is_named BOOLEAN DEFAULT 0,
         representative_face TEXT)
    ''')
    c.execute('''
        CREATE TABLE faces
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         group_id INTEGER,
         image_path TEXT,
         face_encoding BLOB,
         original_image_path TEXT,
         face_location TEXT,
         face_image TEXT,
         created_at TIMESTAMP,
         FOREIGN KEY (group_id) REFERENCES face_groups (id))
    ''')
    c.execute('''
        CREATE TABLE images
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         file_path TEXT UNIQUE,
         uploaded_at TIMESTAMP)
    ''')
    c.execute('CREATE INDEX idx_faces_group_id ON faces (group_id)')
    c.execute('CREATE INDEX idx_images_path ON images (file_path)')
    conn.commit()
    conn.close()
    
    logger.info("Database initialized with proper schema and indexes")

# Mount static files in the correct order (most specific first)
# Make sure the faces directory comes before the catch-all routes
app.mount("/faces", StaticFiles(directory=str(FACES_DIR)), name="faces")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads") 
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def detect_blur(image_array) -> Tuple[bool, float]:
    """
    Detect if an image is blurry using Laplacian variance.
    Optimized for speed.
    """
    try:
        if image_array is None or image_array.size == 0:
            return True, 0.0
        
        # Resize image for faster processing
        if image_array.shape[0] > 200 or image_array.shape[1] > 200:
            resized = cv2.resize(image_array, (200, 200))
        else:
            resized = image_array
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # Calculate Laplacian variance (measure of image sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Determine if image is blurry based on threshold
        is_blurry = laplacian_var < BLUR_THRESHOLD
        logger.info(f"Image blur detection: variance={laplacian_var:.2f}, threshold={BLUR_THRESHOLD}, is_blurry={is_blurry}")
        
        return is_blurry, laplacian_var
    except Exception as e:
        logger.error(f"Error detecting blur: {str(e)}")
        return True, 0.0

def is_human_face(image_array, face_location) -> Tuple[bool, float]:
    """
    Validate if a detected face region is likely a human face.
    Simplified for speed - only basic checks.
    """
    try:
        if image_array is None or face_location is None:
            return False, 0.0
        
        # Get face dimensions
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        
        # Basic size check
        if face_width < MIN_FACE_WIDTH or face_height < MIN_FACE_HEIGHT:
            logger.info(f"Face too small: {face_width}x{face_height}")
            return False, 0.0
            
        # Quick aspect ratio check only
        aspect_ratio = face_width / face_height
        if aspect_ratio < 0.5 or aspect_ratio > 1.5:
            logger.info(f"Invalid face aspect ratio: {aspect_ratio:.2f}")
            return False, 0.0
            
        # Skip the slower landmark detection for speed
        # We'll trust the face_recognition library's detection
        
        return True, 0.8  # Assume it's human if it passes the basic checks
        
    except Exception as e:
        logger.error(f"Error validating human face: {str(e)}")
        return False, 0.0

def detect_faces(image_array) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces using face_recognition library with improved accuracy.
    Returns list of face locations in (top, right, bottom, left) format.
    """
    try:
        # Convert BGR to RGB for face_recognition library
        if len(image_array.shape) == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_array
            
        # Resize image for faster processing
        height, width = rgb_image.shape[:2]
        if height > 800 or width > 800:
            scale = min(800/width, 800/height)
            rgb_image = cv2.resize(rgb_image, (0, 0), fx=scale, fy=scale)
            logger.info(f"Resized image from {width}x{height} to {int(width*scale)}x{int(height*scale)} for faster processing")
            
        # Try HOG model first (faster)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        logger.info(f"Detected {len(face_locations)} faces in image")
        
        # Scale back the locations if we resized the image
        if height > 800 or width > 800:
            scale = min(800/width, 800/height)
            face_locations = [(int(top/scale), int(right/scale), 
                               int(bottom/scale), int(left/scale)) 
                               for top, right, bottom, left in face_locations]
        
        return face_locations
        
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []

def save_face_image(face_img, output_path):
    """Save a face image to the specified path"""
    try:
        # Resize for faster processing and smaller storage
        if face_img.shape[0] > 200 or face_img.shape[1] > 200:
            face_img = cv2.resize(face_img, (200, 200))
            
        # Convert OpenCV BGR to RGB format
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Use PIL to save the image (better compatibility than OpenCV)
        pil_img = Image.fromarray(face_img_rgb)
        
        # Create directories if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image with proper quality - reduced for speed
        pil_img.save(output_path, quality=80)
        
        # Verify the file was actually created
        if os.path.exists(output_path):
            logger.info(f"Successfully saved face image to {output_path}")
            return True
        else:
            logger.error(f"Failed to save face image - file doesn't exist after save: {output_path}")
            return False
    except Exception as e:
        logger.error(f"Error saving face image: {str(e)}")
        return False

def crop_face(image: np.ndarray, face_location: Tuple[int, int, int, int], padding: int = 30) -> Optional[np.ndarray]:
    """Extract face region from image with padding and aspect ratio preservation"""
    try:
        top, right, bottom, left = face_location
        height, width = image.shape[:2]
        
        # Calculate face dimensions
        face_width = right - left
        face_height = bottom - top
        
        # Add padding
        pad_x = int(face_width * (padding/100))
        pad_y = int(face_height * (padding/100))
        
        # Calculate padded boundaries while keeping aspect ratio
        left = max(0, left - pad_x)
        right = min(width, right + pad_x)
        top = max(0, top - pad_y)
        bottom = min(height, bottom + pad_y)
        
        # Ensure square crop by taking the larger dimension
        crop_size = max(right - left, bottom - top)
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        # Adjust boundaries to make square crop
        left = max(0, center_x - crop_size // 2)
        right = min(width, center_x + crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        bottom = min(height, center_y + crop_size // 2)
        
        # Crop the face region
        face_img = image[top:bottom, left:right]
        
        # Ensure we got a valid crop
        if face_img is None or face_img.size == 0:
            logger.error("Invalid face crop dimensions")
            return None
            
        return face_img
        
    except Exception as e:
        logger.error(f"Error cropping face: {str(e)}")
        return None

def get_face_embedding(image_array: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Get face embedding using face_recognition library.
    Returns 128-D face embedding vector.
    """
    try:
        if face_location is None:
            return None
            
        # Convert to RGB if needed
        if len(image_array.shape) == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_array
            
        # Get face encoding with reduced jitters for faster processing
        face_encodings = face_recognition.face_encodings(rgb_image, [face_location], num_jitters=1)
        
        if not face_encodings:
            logger.warning("No face encoding found")
            return None
            
        # Get the first encoding and ensure it's the right format
        face_encoding = face_encodings[0]
        if not isinstance(face_encoding, np.ndarray):
            logger.error(f"Invalid embedding format: {type(face_encoding)}")
            return None
            
        logger.info("Successfully encoded face")
        return face_encoding
        
    except Exception as e:
        logger.error(f"Error getting face embedding: {str(e)}")
        return None

def is_same_person(embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[bool, float]:
    """
    Compare two face embeddings using face_recognition's built-in comparison.
    Returns (is_match, confidence_score).
    """
    try:
        if embedding1 is None or embedding2 is None:
            return False, 0.0
        
        # Convert to correct shape if needed
        if embedding1.shape != (128,):
            embedding1 = embedding1.reshape(128,)
        if embedding2.shape != (128,):
            embedding2 = embedding2.reshape(128,)
            
        # Use face_recognition's compare_faces with a threshold
        distance = face_recognition.face_distance([embedding1], embedding2)[0]
        similarity = 1 - distance  # Convert distance to similarity score
        
        # More accurate face matching
        is_match = similarity >= SIMILARITY_THRESHOLD
        logger.info(f"Face comparison: similarity={similarity:.3f}, threshold={SIMILARITY_THRESHOLD}, match={is_match}")
        return is_match, float(similarity)
        
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        return False, 0.0

def perform_dbscan_clustering(face_encodings: List[np.ndarray], eps=CLUSTERING_EPS, min_samples=1):
    """Perform DBSCAN clustering on face encodings"""
    try:
        if not USE_CLUSTERING or len(face_encodings) < 2:
            # Simply assign each face to its own individual cluster
            cluster_labels = np.array(range(len(face_encodings)))
            logger.info(f"Assigned {len(set(cluster_labels))} individual clusters for {len(face_encodings)} faces")
            return cluster_labels
            
        # Reshape to 2D array if needed
        face_encodings_np = np.array(face_encodings)
        if len(face_encodings_np.shape) == 1:
            face_encodings_np = face_encodings_np.reshape(1, -1)
            
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(face_encodings_np)
        cluster_labels = clustering.labels_
        logger.info(f"DBSCAN clustering found {len(set(cluster_labels))} clusters for {len(face_encodings)} faces")
        return cluster_labels
    except Exception as e:
        logger.error(f"Error performing clustering: {str(e)}")
        return np.array(range(len(face_encodings)))

def find_matching_group(face_embedding: np.ndarray, conn: sqlite3.Connection) -> Optional[int]:
    """Find a matching group for a face based on similarity - original algorithm"""
    try:
        c = conn.cursor()

        # Get all face encodings from database
        c.execute("SELECT id, group_id, face_encoding FROM faces")
        results = c.fetchall()

        if not results:
            logger.info("No existing faces to compare with")
            return None

        # Group face encodings by group_id
        groups = defaultdict(list)
        group_face_ids = defaultdict(list)
        
        for face_id, group_id, face_encoding_blob in results:
            try:
                encoding = pickle.loads(face_encoding_blob)
                groups[group_id].append(encoding)
                group_face_ids[group_id].append(face_id)
            except Exception as e:
                logger.error(f"Error unpickling encoding: {str(e)}")
                continue

        # Original algorithm for face matching
        matches = []
        
        # Compare with all faces and find matches
        for group_id, encodings in groups.items():
            for encoding in encodings:
                is_match, similarity = is_same_person(face_embedding, encoding)
                if is_match:
                    matches.append((group_id, similarity))
        
        # If we have matches, use the highest similarity one
        if matches:
            # Sort by similarity (highest first)
            matches.sort(key=lambda x: x[1], reverse=True)
            best_group_id, best_similarity = matches[0]
            logger.info(f"Found matching group {best_group_id} with similarity {best_similarity:.3f}")
            return best_group_id
        else:
            logger.info("No matching group found")
            return None
            
    except Exception as e:
        logger.error(f"Error finding matching group: {str(e)}")
        return None

def is_valid_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate if an image is suitable for face detection.
    Returns (is_valid, reason).
    """
    try:
        if image is None or image.size == 0:
            return False, "Empty or invalid image"
        
        # Minimal dimension check for speed
        height, width = image.shape[:2]
        if width < 50 or height < 50:
            return False, f"Image too small: {width}x{height}, min required: 50x50"
        
        # Skip blur detection for validation - faster
        return True, "Image is valid"
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False, f"Error validating image: {str(e)}"

@app.post("/upload")
@app.post("/upload/")
async def upload_photo(file: UploadFile = File(...)):
    """Upload and process a photo for face detection and grouping"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Add to images table
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT OR IGNORE INTO images (file_path, uploaded_at) VALUES (?, ?)",
            (str(file_path), datetime.now())
        )
        conn.commit()
            
        # Read image
        image = cv2.imread(str(file_path))
        if image is None:
            raise HTTPException(
                status_code=422,
                detail="Invalid image file"
            )
            
        # Validate image quality
        is_valid, reason = is_valid_image(image)
        if not is_valid:
            logger.warning(f"Invalid image quality: {reason}")
            return {"message": f"Image quality check failed: {reason}", "groups": await get_face_groups()}
            
        # Detect faces in the image
        face_locations = detect_faces(image)
        logger.info(f"Detected {len(face_locations)} faces in image")
        
        if not face_locations:
            logger.warning(f"No faces detected in {file.filename}")
            return {"message": "No faces detected in image", "groups": await get_face_groups()}
            
        # Process each detected face
        processed_faces = []
        
        try:
            # Process each face individually - Google Photos style
            for i, face_location in enumerate(face_locations):
                try:
                    # Extract face region with padding
                    face_img = crop_face(image, face_location)
                    if face_img is None:
                        logger.warning(f"Failed to crop face {i}")
                        continue
                    
                    # Check if face is extremely blurry
                    is_blurry, blur_score = detect_blur(face_img)
                    if is_blurry and blur_score < 5:  # Only filter extremely blurry faces
                        logger.warning(f"Skipping extremely blurry face {i} with blur score {blur_score}")
                        continue
                        
                    # Check if face is human - only filter obvious non-human objects
                    is_human, human_confidence = is_human_face(image, face_location)
                    if not is_human:
                        logger.warning(f"Skipping obvious non-human object {i}")
                        continue
                    
                    # Generate unique filename for face image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    face_filename = f"face_{timestamp}_{i}.jpg"
                    face_path = FACES_DIR / face_filename
                    
                    # Save face image
                    if not save_face_image(face_img, str(face_path)):
                        logger.warning(f"Failed to save face image {i}")
                        continue
                    
                    # Get face encoding
                    face_embedding = get_face_embedding(image, face_location)
                    if face_embedding is None:
                        logger.warning(f"Failed to encode face {i}")
                        continue
                    
                    # Try to find a matching group - using Google Photos-like strict criteria
                    matching_group_id = find_matching_group(face_embedding, conn)
                    
                    if matching_group_id:
                        # Use existing group if there's an extremely high confidence match
                        group_id = matching_group_id
                        logger.info(f"Assigning face {i} to existing group {group_id} - extremely high confidence match")
                    else:
                        # Create a new group - Google Photos style
                        c = conn.cursor()
                        logger.info(f"Creating new group for face {i} - no extremely high confidence match found")
                        
                        # Get the next available group number for better naming
                        c.execute("SELECT MAX(id) FROM face_groups")
                        result = c.fetchone()[0]
                        next_id = result + 1 if result else 1
                        
                        c.execute(
                            "INSERT INTO face_groups (name, representative_face, created_at) VALUES (?, ?, ?)",
                            (f"Person {next_id}", face_filename, datetime.now())
                        )
                        conn.commit()
                        group_id = c.lastrowid
                        logger.info(f"Created new group {group_id} for face {i}")
                    
                    # Add face to database
                    c = conn.cursor()
                    logger.info(f"Adding face {i} to group {group_id}")
                    c.execute(
                        """INSERT INTO faces 
                           (group_id, image_path, face_encoding, original_image_path, face_location, face_image, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            group_id,
                            str(face_path),
                            pickle.dumps(face_embedding),
                            str(file_path),
                            str(face_location),
                            face_filename,
                            datetime.now()
                        )
                    )
                    conn.commit()
                    logger.info(f"Successfully added face {i} to group {group_id}")
                    
                    processed_faces.append({
                        "face_id": i,
                        "group_id": group_id,
                        "face_image": face_filename
                    })
                except Exception as e:
                    logger.error(f"Error processing face {i}: {str(e)}")
                    continue
            
            # Force update representative faces
            update_group_representative_faces(conn)
            logger.info("Updated representative faces for all groups")
            
            # Consolidate similar groups - Google Photos uses very high thresholds to only merge identical faces
            consolidate_similar_groups(conn)
            
            # Get the updated groups
            groups = await get_face_groups()
            logger.info(f"Retrieved {len(groups)} groups for response")
            
            # Check if any faces were successfully processed
            if not processed_faces:
                logger.warning("No valid faces were detected in the image - all detected faces failed quality checks")
                return {
                    "message": "No valid faces were detected. The image may contain blurry faces or non-human objects.",
                    "groups": groups
                }
            
            return {
                "message": f"Successfully processed {len(processed_faces)} faces",
                "faces": processed_faces,
                "groups": groups
            }
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/groups")
@app.get("/groups/")
async def get_face_groups():
    """Get all face groups with their details"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get all groups
        c.execute("""
            SELECT g.id, g.name, g.representative_face 
            FROM face_groups g
            ORDER BY g.id
        """)
        
        groups_data = c.fetchall()
        if not groups_data:
            logger.warning("No groups found in database")
            return []
            
        logger.info(f"Found {len(groups_data)} groups in database")
        
        # Process each group to include counts and details
        result = []
        for group_id, name, representative_face in groups_data:
            # Get count of faces in this group
            c.execute("SELECT COUNT(*) FROM faces WHERE group_id = ?", (group_id,))
            face_count = c.fetchone()[0]
            
            # Get count of unique original images in this group
            c.execute("SELECT COUNT(DISTINCT original_image_path) FROM faces WHERE group_id = ?", (group_id,))
            image_count = c.fetchone()[0]
            
            # Get face images for this group (limited to 5 for performance)
            c.execute("SELECT face_image FROM faces WHERE group_id = ? LIMIT 5", (group_id,))
            face_images = [face_img for (face_img,) in c.fetchall()]
            
            # Get a better representative face if none is set
            if not representative_face and face_images:
                representative_face = face_images[0]
                
                # Update the group's representative face
                c.execute("UPDATE face_groups SET representative_face = ? WHERE id = ?", 
                          (representative_face, group_id))
                conn.commit()
            
            # Create group object with all data
            group_data = {
                "id": group_id,
                "name": name,
                "face_count": face_count,
                "image_count": image_count,
                "representative_face": representative_face,
                "face_images": face_images
            }
            
            logger.info(f"Processing group {group_id}: {name} with {face_count} faces across {image_count} images")
            result.append(group_data)
            
        logger.info(f"Returning {len(result)} groups with {sum(g['face_count'] for g in result)} total faces")
        
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error getting face groups: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return []

@app.get("/group/{group_id}")
async def get_group_details(group_id: int):
    """Get detailed information for a specific face group"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get group info
        c.execute("SELECT id, name, representative_face FROM face_groups WHERE id = ?", (group_id,))
        group = c.fetchone()
        
        if not group:
            raise HTTPException(
                status_code=404,
                detail=f"Group {group_id} not found"
            )
            
        group_id, name, representative_face = group
        
        # Get all faces in this group
        c.execute("""
            SELECT id, face_image, original_image_path, created_at
            FROM faces
            WHERE group_id = ?
            ORDER BY created_at DESC
        """, (group_id,))
        
        face_rows = c.fetchall()
        
        # Get all unique images containing faces in this group
        c.execute("""
            SELECT DISTINCT original_image_path 
            FROM faces 
            WHERE group_id = ?
        """, (group_id,))
        
        image_paths = [row[0] for row in c.fetchall()]
        
        # Create response object
        response = {
            "id": group_id,
            "name": name,
            "representative_face": representative_face,
            "face_count": len(face_rows),
            "image_count": len(image_paths),
            "faces": [
                {
                    "id": face_id,
                    "face_image": face_image,
                    "original_image": os.path.basename(original_image_path) if original_image_path else None,
                    "created_at": created_at
                }
                for face_id, face_image, original_image_path, created_at in face_rows
            ],
            "images": [os.path.basename(path) for path in image_paths if path]
        }
        
        conn.close()
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting group details: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving group details: {str(e)}"
        )

@app.get("/image/{image_name}")
async def get_original_image(image_name: str):
    """Serve original images"""
    try:
        # First check in uploads directory
        file_path = UPLOAD_DIR / image_name
        
        if not os.path.exists(file_path):
            # If not found, try absolute path (stored in database)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT file_path FROM images WHERE file_path LIKE ?", (f"%{image_name}",))
            result = c.fetchone()
            conn.close()
            
            if result:
                file_path = Path(result[0])
                if not os.path.exists(file_path):
                    # Try to find the file by looking up the basename in uploads folder
                    base_name = os.path.basename(file_path)
                    possible_path = UPLOAD_DIR / base_name
                    if os.path.exists(possible_path):
                        file_path = possible_path
                    else:
                        logger.error(f"Original image not found at path {file_path}")
                        raise HTTPException(
                            status_code=404,
                            detail=f"Image file not found on disk: {image_name}"
                        )
            else:
                logger.error(f"Original image not found in database: {image_name}")
                # Try one more time with direct file lookup
                for file in os.listdir(UPLOAD_DIR):
                    if image_name in file or file == image_name:
                        file_path = UPLOAD_DIR / file
                        break
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Image not found: {image_name}"
                    )
            
        # Determine the correct content type
        content_type = "image/jpeg"  # Default
        if file_path.suffix.lower() in ['.png']:
            content_type = "image/png"
        elif file_path.suffix.lower() in ['.gif']:
            content_type = "image/gif"
        
        # Get the real filename for the Content-Disposition header
        filename = os.path.basename(file_path)
        
        # Serve the image with proper download headers
        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""}
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving original image {image_name}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

# Add catch-all route for undefined paths AFTER all other routes
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path_name: str):
    logger.warning(f"Undefined route accessed: {path_name}")
    raise HTTPException(status_code=404, detail="Route not found")

# Custom 404 handler
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": f"Path {request.url.path} not found"}
    )

def update_group_representative_faces(conn: sqlite3.Connection):
    """Update representative faces for face groups"""
    try:
        c = conn.cursor()
        
        # Get all groups
        c.execute("SELECT id FROM face_groups")
        groups = c.fetchall()
        
        for (group_id,) in groups:
            # Get the most recent face for this group
            c.execute("""
                SELECT face_image 
                FROM faces 
                WHERE group_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (group_id,))
            
            result = c.fetchone()
            if result:
                rep_face = result[0]
                # Update the group's representative face
                c.execute("""
                    UPDATE face_groups 
                    SET representative_face = ? 
                    WHERE id = ?
                """, (rep_face, group_id))
        
        conn.commit()
        logger.info("Updated representative faces for all groups")
        
    except Exception as e:
        logger.error(f"Error updating representative faces: {str(e)}")
        conn.rollback()

def consolidate_similar_groups(conn: sqlite3.Connection):
    """Consolidate similar groups based on face similarity - original algorithm"""
    try:
        c = conn.cursor()

        # Get all groups
        c.execute("""
            SELECT g.id, 
                (SELECT COUNT(*) FROM faces WHERE group_id = g.id) as face_count
            FROM face_groups g
            WHERE (SELECT COUNT(*) FROM faces WHERE group_id = g.id) > 0
            ORDER BY face_count DESC
        """)
        
        groups = c.fetchall()
        
        if len(groups) < 2:
            # No groups to consolidate
            logger.info("Less than 2 groups, no consolidation needed")
            return
            
        logger.info(f"Checking {len(groups)} groups for consolidation")
        
        # Original algorithm: Compare each group with all other groups
        merged_groups = set()
        
        # Process groups from smallest to largest (to merge smaller into larger)
        for i in range(len(groups) - 1, 0, -1):
            small_group_id, small_count = groups[i]
            
            # Skip if already merged
            if small_group_id in merged_groups:
                continue
                
            # Get face encodings for this group
            c.execute("SELECT face_encoding FROM faces WHERE group_id = ?", (small_group_id,))
            small_group_faces = c.fetchall()
            
            if not small_group_faces:
                continue
                
            small_encodings = []
            for encoding_blob, in small_group_faces:
                try:
                    encoding = pickle.loads(encoding_blob)
                    small_encodings.append(encoding)
                except Exception as e:
                    logger.error(f"Error unpickling encoding: {str(e)}")
                    continue
            
            if not small_encodings:
                continue
            
            # Compare with all larger groups
            for j in range(i):
                large_group_id, large_count = groups[j]
                
                # Skip if same group or if already merged
                if small_group_id == large_group_id or large_group_id in merged_groups:
                    continue
                
                # Get face encodings for larger group
                c.execute("SELECT face_encoding FROM faces WHERE group_id = ?", (large_group_id,))
                large_group_faces = c.fetchall()
                
                if not large_group_faces:
                    continue
                
                large_encodings = []
                for encoding_blob, in large_group_faces:
                    try:
                        encoding = pickle.loads(encoding_blob)
                        large_encodings.append(encoding)
                    except Exception as e:
                        logger.error(f"Error unpickling encoding: {str(e)}")
                        continue
                
                if not large_encodings:
                    continue
                
                # Original algorithm: Count matches between groups
                # Need at least 50% of faces to match for merging
                match_count = 0
                total_comparisons = min(len(small_encodings), len(large_encodings))
                
                # Only compare a sample if there are many faces
                max_comparisons = 10
                if total_comparisons > max_comparisons:
                    # Sample evenly from both groups
                    sample_small = small_encodings[:max_comparisons]
                    sample_large = large_encodings[:max_comparisons]
                else:
                    sample_small = small_encodings
                    sample_large = large_encodings
                
                # Compare samples
                for small_enc in sample_small:
                    for large_enc in sample_large:
                        try:
                            is_match, similarity = is_same_person(small_enc, large_enc)
                            if is_match:
                                match_count += 1
                                # Early exit if we have enough matches
                                if match_count >= len(sample_small) / 2:
                                    break
                        except Exception as e:
                            logger.error(f"Error comparing faces: {str(e)}")
                
                # Check if we have enough matches
                should_merge = match_count >= len(sample_small) / 2
                
                if should_merge:
                    logger.info(f"Merging group {small_group_id} into {large_group_id}, {match_count} matches out of {len(sample_small)} samples")
                    
                    try:
                        # Update faces to belong to the larger group
                        c.execute("UPDATE faces SET group_id = ? WHERE group_id = ?", (large_group_id, small_group_id))
                        
                        # Delete the now-empty small group
                        c.execute("DELETE FROM face_groups WHERE id = ?", (small_group_id,))
                        
                        # Add to merged set
                        merged_groups.add(small_group_id)
                        
                        # Commit changes
                        conn.commit()
                        
                        # Early exit - no need to compare with other groups
                        break
                    except Exception as e:
                        logger.error(f"Error merging groups: {str(e)}")
                        conn.rollback()
                else:
                    logger.info(f"Not merging groups {small_group_id} and {large_group_id}, only {match_count} matches out of {len(sample_small)} samples")
        
        # Commit any pending changes
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error consolidating groups: {str(e)}")
        conn.rollback()

# Add face_image endpoint
@app.get("/face_image/{face_image_name}")
async def get_face_image(face_image_name: str):
    """Serve cropped face images"""
    try:
        # Get the absolute path to the faces directory
        faces_dir_path = os.path.abspath(FACES_DIR)
        face_path = os.path.join(faces_dir_path, face_image_name)
        
        if os.path.exists(face_path):
            logger.info(f"Serving face image from: {face_path}")
            return FileResponse(face_path, media_type="image/jpeg")
        else:
            # If file doesn't exist, try direct redirect to faces
            redirect_url = f"/faces/{face_image_name}"
            logger.info(f"Redirecting to: {redirect_url}")
            return Response(status_code=307, headers={"Location": redirect_url})
    except Exception as e:
        logger.error(f"Error serving face image {face_image_name}: {str(e)}")
        # Try redirecting as a fallback
        return Response(status_code=307, headers={"Location": f"/faces/{face_image_name}"})

if __name__ == "__main__":
    # Initialize database
    init_db()
    logger.info("Created new face cache")
    logger.info("Initialized FaceGroupManager with improved face recognition")
    
    port = 8092  # Use a different port to avoid conflicts
    
    print("="*50)
    print("Server starting! Open this link in your browser:")
    print(f"http://localhost:{port}")
    print("="*50)
    
    # Start the server with proper host binding and error handling
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=port,
        log_level="info",
        access_log=True,
        reload=False  # Disable auto-reload to prevent port conflicts
    ) 