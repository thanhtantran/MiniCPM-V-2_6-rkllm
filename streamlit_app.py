import os
import streamlit as st
from PIL import Image
import atexit

# Import the extracted modules
from model_manager import ModelManager
from subprocess_manager import StreamlitSubprocessManager

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
                            st.error("❌ Failed to start inference process. Check console for details.")
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
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
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
                                with st.spinner("Analyzing image..."):
                                    # Use the updated send_question method with separate parameters
                                    response = st.session_state.inference_manager.send_question(question, image_path)
                                
                                st.subheader("🤖 Response:")
                                st.write(response)
                            else:
                                st.warning("Please enter a question about the image.")
    
    # Cleanup on app termination
    if hasattr(st.session_state, 'inference_manager'):
        atexit.register(st.session_state.inference_manager.stop_process)

if __name__ == "__main__":
    main()