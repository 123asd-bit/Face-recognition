#!/usr/bin/env python3
"""
This script patches main.py to fix the dict_keys error that occurs during face processing.
"""
import re
import sys
import os

def main():
    print("Patching main.py to fix dict_keys error...")
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("Error: main.py not found in current directory.")
        return 1
    
    # Read the content of main.py
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix for dict_keys issue
    pattern1 = r"(def create_new_group_for_cluster\(.*?\).*?\n\s+)(?:try:\s*\n)?"
    replacement1 = r"""\1try:
        # Convert face_indices to a list if it's a dict_keys object
        if hasattr(face_indices, 'keys'):
            face_indices = list(face_indices)
            
        """
    
    # Fix error handling in upload endpoint
    pattern2 = r"(@app\.post\(\"/upload\"\).*?async def upload_file.*?\n.*?process_image\(file_location\))"
    replacement2 = r"""\1
        try:
            face_group_manager.process_image(file_location)
            # Ensure we update all groups after processing
            face_group_manager.update_representative_faces()
            # Consolidate similar groups
            face_group_manager.consolidate_similar_groups()
        except Exception as e:
            logging.error(f"Error processing upload: {str(e)}")
            # Return a 200 response even on error to not block the UI
            return {"status": "error", "message": f"Error processing faces: {str(e)}"}"""
    
    # Apply the patches
    patched_content = content
    patched_content = re.sub(pattern1, replacement1, patched_content, flags=re.DOTALL)
    patched_content = re.sub(pattern2, replacement2, patched_content, flags=re.DOTALL)
    
    # Write the patched content back to main.py
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(patched_content)
    
    print("Patching complete. Please restart the server.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 