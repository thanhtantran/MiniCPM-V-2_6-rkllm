import streamlit as st
from PIL import Image
import atexit
import os

# Import the extracted modules
from model_manager import ModelManager
from subprocess_manager import StreamlitSubprocessManager

def load_css():
    """Load external CSS file"""
    css_file = "css/styles.css"
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline CSS if file doesn't exist
        st.markdown(
            """
            <style>
            .logo-container {
                position: fixed;
                top: 15px;
                right: 15px;
                z-index: 999;
                background: rgba(255, 255, 255, 0.95);
                padding: 8px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }
            .logo-container:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            }
            .logo-container img {
                height: 50px;
                width: auto;
                display: block;
                border-radius: 4px;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                color: #495057;
                text-align: center;
                padding: 12px 0;
                border-top: 2px solid #ff6600;
                z-index: 999;
                font-size: 14px;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            }
            .footer a {
                color: #ff6600;
                text-decoration: none;
                font-weight: 600;
                transition: color 0.3s ease;
            }
            .footer a:hover {
                color: #e55a00;
                text-decoration: underline;
            }
            .main .block-container {
                padding-bottom: 70px;
            }
            @media (max-width: 768px) {
                .logo-container {
                    top: 10px;
                    right: 10px;
                    padding: 6px;
                }
                .logo-container img {
                    height: 40px;
                }
                .footer {
                    font-size: 12px;
                    padding: 10px 5px;
                }
            }
            </style>
            """,
            unsafe_allow_html=True
        )

def add_logo_and_header():
    """Add Orange Pi logo to the top right corner"""
    logo_path = "static/logo2.png"
    
    # Check if logo exists, fallback to logo1.png if logo2.png is not found
    if not os.path.exists(logo_path):
        logo_path = "logo2.png"  # Direct path if not in static folder yet
        if not os.path.exists(logo_path):
            logo_path = "logo1.png"  # Final fallback
    
    st.markdown(
        f"""
        <div class="logo-container">
            <a href="http://orangepi.net" target="_blank" title="Visit Orange Pi">
                <img src="{logo_path}" alt="Orange Pi Logo">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

def add_footer():
    """Add Orange Pi footer"""
    st.markdown(
        """
        <div class="footer">
            <p>Copyright 2025 by <a href="https://orangepi.vn" target="_blank" title="Visit Orange Pi Vietnam">Orange Pi Vietnam</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(
        page_title="MiniCPM-V-2.6 RKLLM Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    # Load CSS and add UI elements
    load_css()
    add_logo_and_header()
    add_footer()
    
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
                            success = st.session_state.inference_manager.start_process()
                            
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