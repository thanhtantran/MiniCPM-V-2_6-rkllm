import os
from pathlib import Path
from huggingface_hub import snapshot_download
import streamlit as st

# Configuration
MODEL_DIR = "model"
HF_REPO = "thanhtantran/MiniCPM-V-2_6-rkllm"
REQUIRED_FILES = ["qwen.rkllm", "vision_transformer.rknn"]
TEMP_DIR = "temp_images"

class ModelManager:
    def __init__(self):
        self.model_dir = Path(MODEL_DIR)
        # Fix: Create temp_dir relative to current working directory
        self.temp_dir = Path.cwd() / TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)
        
    def check_model_files(self):
        """Check if required model files exist"""
        if not self.model_dir.exists():
            return False, []
        
        existing_files = []
        missing_files = []
        
        for file in REQUIRED_FILES:
            file_path = self.model_dir / file
            if file_path.exists():
                existing_files.append(file)
            else:
                missing_files.append(file)
        
        return len(missing_files) == 0, existing_files
    
    def download_models(self, progress_callback=None):
        """Download model files from Hugging Face"""
        try:
            if progress_callback:
                progress_callback("Starting download...")
            
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=self.model_dir,
                allow_patterns=REQUIRED_FILES
            )
            
            if progress_callback:
                progress_callback("Download completed!")
            
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Download failed: {e}")
            return False
    
    def save_uploaded_image(self, uploaded_file):
        """Save uploaded image to temp directory"""
        try:
            # Create a temporary file with the original extension
            file_extension = Path(uploaded_file.name).suffix
            temp_file = self.temp_dir / f"image{file_extension}"
            
            # Save the uploaded file
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return str(temp_file)
        except Exception as e:
            st.error(f"Failed to save image: {e}")
            return None