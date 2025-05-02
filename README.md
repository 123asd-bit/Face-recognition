# Face Grouping System

A Google Photos-like face grouping system that automatically groups photos by detected faces. Built with FastAPI and modern web technologies.

## Features

- Upload multiple photos
- Automatic face detection and grouping
- View all photos where a person appears
- Modern, responsive UI
- Real-time face grouping updates

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd face-grouping-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

## Usage

1. Click the "Choose Files" button to select photos
2. Click "Upload Photos" to process the images
3. The system will automatically detect faces and group them
4. Click on any group to view all photos where that person appears

## Technical Details

- Backend: FastAPI with face_recognition library
- Frontend: HTML, CSS, and JavaScript
- Database: SQLite
- Face Detection: face_recognition library (based on dlib)

## Notes

- The face_recognition library requires dlib, which may need additional system dependencies depending on your OS
- For best results, use clear, well-lit photos
- Processing time depends on the number and size of uploaded photos

## License

MIT License 