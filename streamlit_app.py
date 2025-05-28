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
from multiprocessing import Process, Queue, Event
import cv2
import numpy as np
from rkllm_binding import *
from rknnlite.api.rknn_lite import RKNNLite
import signal

# Configuration
MODEL_DIR = "model"
HF_REPO = "thanhtantran/MiniCPM-V-2_6-rkllm"
REQUIRED_FILES = ["qwen.rkllm", "vision_transformer.rknn"]
TEMP_DIR = "temp_images"

# Import the functions from multiprocess_inference.py
from multiprocess_inference import vision_encoder_process, llm_process

class StreamlitMultiprocessManager:
    def __init__(self):
        self.load_ready_queue = Queue()
        self.embedding_queue = Queue()
        self.img_path_queue = Queue()
        self.prompt_queue = Queue()
        self.inference_done_queue = Queue()
        self.start_event = Event()
        
        self.vision_process = None
        self.lm_process = None
        self.is_ready = False
        self.response_text = ""
        
    def start_processes(self):
        """Start the vision and LLM processes"""
        self.vision_process = Process(target=vision_encoder_process,
                                    args=(self.load_ready_queue, self.embedding_queue, 
                                         self.img_path_queue, self.start_event))
        self.lm_process = Process(target=llm_process,
                                args=(self.load_ready_queue, self.embedding_queue, 
                                     self.prompt_queue, self.inference_done_queue, self.start_event))
        
        self.vision_process.start()
        self.lm_process.start()
        
        # Wait for models to load
        ready_count = 0
        while ready_count < 2:
            status = self.load_ready_queue.get()
            print(f"Received ready signal: {status}")
            ready_count += 1
        
        print("All models loaded, ready for inference...")
        self.start_event.set()
        self.is_ready = True
        
    def send_question(self, question, image_path):
        """Send a question with image for inference"""
        if not self.is_ready:
            return "Error: Inference processes not ready"
        
        # Print debug info
        print(f"\n=== USER INPUT ===")
        print(f"Question: {question}")
        print(f"Image Path: {image_path}")
        print("==================")
        
        # Format the input like the original multiprocess_inference.py
        full_input = f"{question} {{{{{image_path}}}}}"
        
        # Parse input like in original code
        img_match = re.search(r'\{\{(.+?)\}\}', full_input)
        if not img_match:
            return "Error: No image path found in input"
            
        img_path = img_match.group(1)
        
        # Replace image marker with <image> tag
        image_placeholder = '<image_id>0</image_id><image>\n'
        prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{full_input.replace(img_match.group(0), image_placeholder)}<|im_end|>
<|im_start|>assistant
"""
        
        print(f"\n=== FORMATTED PROMPT ===")
        print(prompt)
        print("========================\n")
        
        # Send to processes
        self.img_path_queue.put(img_path)
        self.prompt_queue.put(prompt)
        
        # Wait for inference to complete
        try:
            status = self.inference_done_queue.get(timeout=60)
            if status == "DONE":
                return "Inference completed successfully. Check console for response."
            else:
                return "Error: Inference failed"
        except:
            return "Error: Inference timeout"
            
    def stop_processes(self):
        """Stop the inference processes"""
        if self.is_ready:
            self.img_path_queue.put("STOP")
            self.prompt_queue.put("STOP")
            
        if self.vision_process:
            self.vision_process.join(timeout=5)
            if self.vision_process.is_alive():
                self.vision_process.terminate()
                
        if self.lm_process:
            self.lm_process.join(timeout=5)
            if self.lm_process.is_alive():
                self.lm_process.terminate()
                
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
                
        return len(missing_files) == 0, {"existing": existing_files, "missing": missing_files}
    
    def download_models(self, progress_callback=None):
        """Download model files from Hugging Face"""
        try:
            if progress_callback:
                progress_callback("Starting download...")
            
            snapshot_download(
                repo_id=HF_REPO,
                allow_patterns=["model/*.rkllm", "model/*.rknn"],
                local_dir=".",
                local_dir_use_symlinks=False
            )
            
            if progress_callback:
                progress_callback("Download completed!")
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Download failed: {str(e)}")
            return False

# Initialize Streamlit
st.set_page_config(
    page_title="MiniCPM-V-2.6-RKLLM",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MiniCPM-V-2.6-RKLLM Multimodal AI")
st.markdown("Upload an image and ask questions about it! **Check the console/terminal for detailed responses.**")

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
    
if 'inference_manager' not in st.session_state:
    st.session_state.inference_manager = None
    
if 'questions_asked' not in st.session_state:
    st.session_state.questions_asked = []
    
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Sidebar for model management
with st.sidebar:
    st.header("📦 Model Management")
    
    # Check model status
    models_exist, file_status = st.session_state.model_manager.check_model_files()
    
    if models_exist:
        st.success("✅ All model files are available!")
        for file in file_status["existing"]:
            st.write(f"✓ {file}")
    else:
        st.warning("⚠️ Some model files are missing")
        if file_status["existing"]:
            st.write("**Available:**")
            for file in file_status["existing"]:
                st.write(f"✓ {file}")
        if file_status["missing"]:
            st.write("**Missing:**")
            for file in file_status["missing"]:
                st.write(f"✗ {file}")
        
        if st.button("📥 Download Missing Models"):
            with st.spinner("Downloading models from Hugging Face..."):
                success = st.session_state.model_manager.download_models()
                if success:
                    st.success("Models downloaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to download models")
    
    st.markdown("---")
    
    # Process management
    st.header("🔧 Process Management")
    
    if models_exist:
        if st.session_state.inference_manager is None:
            if st.button("🚀 Start Inference Process"):
                with st.spinner("Starting inference processes..."):
                    try:
                        st.session_state.inference_manager = StreamlitMultiprocessManager()
                        st.session_state.inference_manager.start_processes()
                        st.success("✅ Inference processes started!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start processes: {str(e)}")
        else:
            st.success("✅ Inference process is running")
            if st.button("🛑 Stop Inference Process"):
                st.session_state.inference_manager.stop_processes()
                st.session_state.inference_manager = None
                st.session_state.questions_asked = []
                st.session_state.responses = []
                st.success("Inference process stopped")
                st.rerun()

# Main content
if models_exist:
    # Image upload section
    st.header("🖼️ Image Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "bmp"],
        disabled=st.session_state.inference_manager is None
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.session_state.inference_manager is not None:
                # Save uploaded file
                temp_image_path = st.session_state.model_manager.temp_dir / "image.jpg"
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ Image saved and ready for analysis")
                
                # Questions section
                st.header("❓ Ask Questions")
                
                # Predefined questions
                st.subheader("Quick Questions")
                predefined_questions = [
                    "Describe this image {{./temp_images/image.jpg}} in detail.",
                    "What objects can you see in this image {{./temp_images/image.jpg}}?",
                    "What is happening in this image {{./temp_images/image.jpg}}?",
                    "What colors are prominent in this image {{./temp_images/image.jpg}}?",
                    "Are there any people in this image {{./temp_images/image.jpg}}?"
                ]
                
                cols = st.columns(2)
                for i, question in enumerate(predefined_questions):
                    col = cols[i % 2]
                    with col:
                        if st.button(f"🤔 {question}", key=f"pred_q_{i}"):
                            with st.spinner(f"Processing: {question}"):
                                try:
                                    response = st.session_state.inference_manager.send_question(
                                        question, 
                                        str(temp_image_path)
                                    )
                                    st.session_state.questions_asked.append(question)
                                    st.session_state.responses.append(response)
                                    st.info("✅ Question sent! Check the console/terminal for the detailed response.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error during inference: {str(e)}")
                
                # Custom question input
                st.subheader("Custom Question")
                custom_question = st.text_input(
                    "Ask your own question about the image:",
                    placeholder="What do you want to know about this image?"
                )
                
                if st.button("🚀 Ask Custom Question", disabled=not custom_question):
                    with st.spinner(f"Processing: {custom_question}"):
                        try:
                            custom_question = question.replace(" {{./temp_images/image.jpg}}", "")
                            response = st.session_state.inference_manager.send_question(
                                custom_question, 
                                str(temp_image_path)
                            )
                            st.session_state.questions_asked.append(custom_question)
                            st.session_state.responses.append(response)
                            st.info("✅ Question sent! Check the console/terminal for the detailed response.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during inference: {str(e)}")
            else:
                st.warning("🚫 Please start the inference process first.")
    
    # Display conversation history
    if st.session_state.questions_asked:
        st.header("💬 Question History")
        
        for i, (question, response) in enumerate(zip(st.session_state.questions_asked, st.session_state.responses)):
            with st.expander(f"Q{i+1}: {question}", expanded=False):
                st.write(f"**Question:** {question}")
                st.write(f"**Status:** {response}")
                st.info("💡 **Tip:** The detailed AI response is displayed in the console/terminal where you started Streamlit.")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.questions_asked = []
            st.session_state.responses = []
            st.rerun()

else:
    st.info("Please download the model files first using the sidebar.")

# Footer
st.markdown("---")
st.markdown("**Note:** ")  
st.markdown("- You can ask multiple questions about the same image")
st.markdown("- **Detailed AI responses are shown in the console/terminal** where you started Streamlit")
st.markdown("- The Streamlit interface shows question status and history")