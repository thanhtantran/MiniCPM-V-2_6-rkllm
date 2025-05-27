import streamlit as st
import os
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from huggingface_hub import snapshot_download
import queue
import signal
import sys

# Configuration
MODEL_DIR = "model"
HF_REPO = "thanhtantran/MiniCPM-V-2_6-rkllm"
REQUIRED_FILES = ["qwen.rkllm", "vision_transformer.rknn"]
TEMP_DIR = "temp_images"

class ModelManager:
    def __init__(self):
        self.model_dir = Path(MODEL_DIR)
        self.temp_dir = Path(TEMP_DIR)
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
                
        return len(missing_files) == 0, {"existing": existing_files, "missing": missing_files}
    
    def download_models(self, progress_callback=None):
        """Download model files from Hugging Face"""
        try:
            if progress_callback:
                progress_callback("Starting download...")
            
            # Download only the model folder with specific file patterns
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=".",
                allow_patterns=["model/*.rkllm", "model/*.rknn"],
                local_dir_use_symlinks=False
            )
            
            if progress_callback:
                progress_callback("Download completed!")
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Download failed: {str(e)}")
            return False

class InferenceManager:
    def __init__(self):
        self.process = None
        self.output_queue = queue.Queue()
        self.is_ready = False
        
    def start_inference_process(self, progress_callback=None):
        """Start the multiprocess_inference.py subprocess"""
        try:
            if progress_callback:
                progress_callback("Starting inference process...")
            
            # Start the subprocess
            self.process = subprocess.Popen(
                [sys.executable, "multiprocess_inference.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start output monitoring thread
            self.output_thread = threading.Thread(
                target=self._monitor_output,
                args=(progress_callback,),
                daemon=True
            )
            self.output_thread.start()
            
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Failed to start process: {str(e)}")
            return False
    
    def _monitor_output(self, progress_callback=None):
        """Monitor subprocess output"""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    self.output_queue.put(line)
                    
                    if progress_callback:
                        progress_callback(line)
                    
                    # Check if models are loaded and ready
                    if "All models loaded, starting interactive mode..." in line:
                        self.is_ready = True
                        if progress_callback:
                            progress_callback("✅ Models loaded! Ready for inference.")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Output monitoring error: {str(e)}")
    
    def send_question(self, question, image_path):
        """Send question to the inference process"""
        if not self.is_ready or not self.process:
            return "Process not ready"
        
        try:
            # Format the question with image path
            formatted_question = f"{question} {{./{image_path}}}"
            
            # Send question followed by 3 empty lines
            self.process.stdin.write(formatted_question + "\n\n\n\n")
            self.process.stdin.flush()
            
            # Collect response
            response_lines = []
            start_time = time.time()
            timeout = 120  # 2 minutes timeout
            
            while time.time() - start_time < timeout:
                try:
                    line = self.output_queue.get(timeout=1)
                    response_lines.append(line)
                    
                    # Check for completion markers
                    if "(finished)" in line:
                        break
                except queue.Empty:
                    continue
            
            # Filter and clean response
            response = self._clean_response(response_lines)
            return response
            
        except Exception as e:
            return f"Error during inference: {str(e)}"
    
    def _clean_response(self, lines):
        """Clean and format the response"""
        # Remove system messages and keep only the actual response
        response_lines = []
        capturing = False
        
        for line in lines:
            # Skip system messages
            if any(skip in line.lower() for skip in [
                "start vision inference", "vision encoder inference time",
                "prefill", "generate", "stage", "tokens", "time per token"
            ]):
                continue
            
            # Start capturing after vision inference
            if "vision encoder inference time" in line:
                capturing = True
                continue
            
            # Stop at finished marker
            if "(finished)" in line:
                break
                
            if capturing and line.strip():
                response_lines.append(line.strip())
        
        return "\n".join(response_lines) if response_lines else "No response received"
    
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

# Initialize managers
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'inference_manager' not in st.session_state:
    st.session_state.inference_manager = InferenceManager()

# Streamlit UI
st.title("🤖 MiniCPM-V-2.6-RKLLM Web Interface")
st.markdown("Run the powerful MiniCPM-V-2.6 Visual Language Model with a web interface!")

# Sidebar for model management
with st.sidebar:
    st.header("📁 Model Management")
    
    # Check model status
    models_exist, file_status = st.session_state.model_manager.check_model_files()
    
    if models_exist:
        st.success("✅ All model files are ready!")
        for file in file_status["existing"]:
            st.text(f"✓ {file}")
    else:
        st.warning("⚠️ Model files missing")
        if file_status["existing"]:
            st.text("Existing files:")
            for file in file_status["existing"]:
                st.text(f"✓ {file}")
        if file_status["missing"]:
            st.text("Missing files:")
            for file in file_status["missing"]:
                st.text(f"✗ {file}")
        
        # Download button
        if st.button("📥 Download Models", type="primary"):
            progress_placeholder = st.empty()
            
            def update_progress(message):
                progress_placeholder.text(message)
            
            with st.spinner("Downloading models..."):
                success = st.session_state.model_manager.download_models(update_progress)
                
            if success:
                st.success("Models downloaded successfully!")
                st.rerun()
            else:
                st.error("Failed to download models")

# Main interface
if models_exist:
    # Process management
    st.header("🚀 Inference Process")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.inference_manager.is_ready:
            if st.button("▶️ Start Inference Process", type="primary"):
                progress_placeholder = st.empty()
                
                def update_progress(message):
                    progress_placeholder.text(message)
                
                with st.spinner("Starting inference process..."):
                    success = st.session_state.inference_manager.start_inference_process(update_progress)
                
                if not success:
                    st.error("Failed to start inference process")
        else:
            st.success("✅ Inference process is ready!")
    
    with col2:
        if st.session_state.inference_manager.process:
            if st.button("⏹️ Stop Process", type="secondary"):
                st.session_state.inference_manager.stop_process()
                st.success("Process stopped")
                st.rerun()
    
    # Image upload and inference
    if st.session_state.inference_manager.is_ready:
        st.header("🖼️ Image Upload & Inference")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Save uploaded file
            temp_image_path = st.session_state.model_manager.temp_dir / "image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Image saved to {temp_image_path}")
            
            # Predefined questions
            st.header("❓ Ask Questions")
            
            questions = [
                "Describe this image in detail.",
                "What objects can you see in this image?",
                "What is happening in this image?"
            ]
            
            # Custom question input
            custom_question = st.text_input("Or ask your own question:")
            if custom_question:
                questions.append(custom_question)
            
            # Question buttons
            for i, question in enumerate(questions):
                if st.button(f"🤔 {question}", key=f"q_{i}"):
                    with st.spinner(f"Processing: {question}"):
                        response = st.session_state.inference_manager.send_question(
                            question, 
                            temp_image_path.relative_to(Path.cwd())
                        )
                    
                    st.header("💬 Response")
                    st.write(response)
else:
    st.info("Please download the model files first using the sidebar.")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit | "
    "Model: [MiniCPM-V-2.6-RKLLM](https://huggingface.co/thanhtantran/MiniCPM-V-2_6-rkllm)"
)