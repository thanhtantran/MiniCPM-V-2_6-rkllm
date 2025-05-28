import streamlit as st
import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image
import tempfile
import subprocess
import threading
import queue
import re
import cv2
import numpy as np

# Configuration
MODEL_DIR = "model"
HF_REPO = "thanhtantran/MiniCPM-V-2_6-rkllm"
REQUIRED_FILES = ["qwen.rkllm", "vision_transformer.rknn"]
TEMP_DIR = "temp_images"

class StreamlitSubprocessManager:
    def __init__(self):
        self.process = None
        self.is_ready = False
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        
    def start_inference_process(self):
        """Start the multiprocess_inference.py as a subprocess"""
        try:
            # Start the subprocess
            self.process = subprocess.Popen(
                ['python', 'multiprocess_inference.py'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start threads to handle I/O
            self.output_thread = threading.Thread(target=self._read_output, daemon=True)
            self.input_thread = threading.Thread(target=self._write_input, daemon=True)
            
            self.output_thread.start()
            self.input_thread.start()
            
            # Wait for the "ready for inference" message
            start_time = time.time()
            while time.time() - start_time < 120:  # 2 minute timeout
                try:
                    output = self.output_queue.get(timeout=1)
                    if "ready for inference" in output.lower():
                        self.is_ready = True
                        return True
                except queue.Empty:
                    continue
                    
            return False
            
        except Exception as e:
            st.error(f"Failed to start inference process: {e}")
            return False
    
    def _read_output(self):
        """Read output from the subprocess"""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line.strip())
        except Exception as e:
            print(f"Error reading output: {e}")
    
    def _write_input(self):
        """Write input to the subprocess"""
        try:
            while self.process and self.process.poll() is None:
                try:
                    input_text = self.input_queue.get(timeout=1)
                    if input_text:
                        self.process.stdin.write(input_text + '\n')
                        self.process.stdin.flush()
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error writing input: {e}")
    
    def send_question(self, question_with_image):
        """Send a question to the inference process"""
        if not self.is_ready or not self.process:
            return "Error: Inference process not ready"
        
        try:
            # Send the question
            self.input_queue.put(question_with_image)
            
            # Send three empty lines to signal end of input (as expected by original script)
            for _ in range(3):
                self.input_queue.put("")
            
            # Collect the response
            response_lines = []
            start_time = time.time()
            
            while time.time() - start_time < 60:  # 60 second timeout
                try:
                    output = self.output_queue.get(timeout=1)
                    
                    # Skip debug/status messages
                    if any(skip in output.lower() for skip in ['loading', 'ready', 'input:', 'formatted prompt']):
                        continue
                    
                    # Look for the actual response
                    if output.strip() and not output.startswith('==='):
                        response_lines.append(output)
                        
                    # Check if we got a complete response
                    if len(response_lines) > 0 and "ready for inference" in output.lower():
                        break
                        
                except queue.Empty:
                    continue
            
            if response_lines:
                return '\n'.join(response_lines)
            else:
                return "Error: No response received"
                
        except Exception as e:
            return f"Error during inference: {e}"
    
    def stop_process(self):
        """Stop the inference process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
        self.is_ready = False

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

def main():
    st.set_page_config(
        page_title="MiniCPM-V-2.6 RKLLM Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 MiniCPM-V-2.6 RKLLM Chat")
    st.markdown("Chat with images using MiniCPM-V-2.6 on RKLLM")
    
    # Initialize managers
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    if 'inference_manager' not in st.session_state:
        st.session_state.inference_manager = StreamlitSubprocessManager()
    
    # Model status section
    with st.expander("📁 Model Status", expanded=True):
        model_exists, existing_files = st.session_state.model_manager.check_model_files()
        
        if model_exists:
            st.success(f"✅ All required model files found: {', '.join(existing_files)}")
        else:
            st.warning("⚠️ Model files not found")
            
            if st.button("📥 Download Models"):
                progress_placeholder = st.empty()
                
                def update_progress(message):
                    progress_placeholder.info(message)
                
                success = st.session_state.model_manager.download_models(update_progress)
                
                if success:
                    st.success("✅ Models downloaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to download models")
    
    # Inference section
    if model_exists:
        with st.expander("🚀 Inference Control", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                if not st.session_state.inference_manager.is_ready:
                    if st.button("🔄 Start Inference Process"):
                        with st.spinner("Starting inference process..."):
                            success = st.session_state.inference_manager.start_inference_process()
                            
                        if success:
                            st.success("✅ Inference process started!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to start inference process")
                else:
                    st.success("✅ Inference process is ready")
            
            with col2:
                if st.session_state.inference_manager.is_ready:
                    if st.button("🛑 Stop Inference Process"):
                        st.session_state.inference_manager.stop_process()
                        st.success("✅ Inference process stopped")
                        st.rerun()
        
        # Chat interface
        if st.session_state.inference_manager.is_ready:
            st.markdown("---")
            st.subheader("💬 Chat Interface")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload an image to analyze"
            )
            
            if uploaded_file is not None:
                # Display the image
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    # Save the image
                    image_path = st.session_state.model_manager.save_uploaded_image(uploaded_file)
                    
                    if image_path:
                        # Question input
                        question = st.text_area(
                            "Ask a question about the image:",
                            placeholder="Describe what you see in this image...",
                            height=100
                        )
                        
                        if st.button("🔍 Analyze Image", type="primary"):
                            if question.strip():
                                # Format the question with image placeholder
                                formatted_question = f"{question} {{{{{image_path}}}}}"
                                
                                with st.spinner("Analyzing image..."):
                                    response = st.session_state.inference_manager.send_question(formatted_question)
                                
                                st.subheader("🤖 Response:")
                                st.write(response)
                            else:
                                st.warning("Please enter a question about the image.")
    
    # Cleanup on app termination
    if hasattr(st.session_state, 'inference_manager'):
        import atexit
        atexit.register(st.session_state.inference_manager.stop_process)

if __name__ == "__main__":
    main()