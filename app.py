import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Human Detection App",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-bottom: 10px;
    }
    .detection-stats {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        background-color: #1f1f1f;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load and cache the HOG detector"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def process_frame(frame, hog, detection_confidence=0.3):
    """Process frame with human detection"""
    try:
        # Resize frame for better performance
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            frame = cv2.resize(frame, (800, int(height * scale)))
        
        # Detect humans
        boxes, weights = hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        
        # Filter detections based on confidence
        confident_detections = [
            (box, weight) for box, weight in zip(boxes, weights)
            if weight > detection_confidence
        ]
        
        # Draw detections
        num_detections = len(confident_detections)
        for (box, _) in confident_detections:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Human', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame, num_detections
    
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return frame, 0

def check_camera_access(camera_index):
    """Check if camera is accessible"""
    cap = cv2.VideoCapture(camera_index)
    if cap is None or not cap.isOpened():
        return False
    cap.release()
    return True

def get_available_cameras():
    """Get list of available cameras"""
    available_cameras = []
    for i in range(5):  # Check first 5 camera indexes
        if check_camera_access(i):
            available_cameras.append(i)
    return available_cameras if available_cameras else [0]  # Return [0] as default if none found

def main():
    st.title("👥 Real-time Human Detection")
    
    # Get available cameras
    available_cameras = get_available_cameras()
    
    # Sidebar settings
    with st.sidebar:
        st.title("⚙️ Settings")
        detection_confidence = st.slider(
            "Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
        
        # Only show available cameras
        camera_index = st.selectbox(
            "Select Camera",
            options=available_cameras,
            index=0,
            help="Choose available camera"
        )
        
        fps_display = st.checkbox("Show FPS", value=True)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📷 Image Upload", "ℹ️ About"])
    
    with tab1:
        st.header("Real-time Human Detection")
        
        # Initialize session state for camera
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        # Camera controls
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button('▶️ Start Camera')
        with col2:
            stop_button = st.button('⏹️ Stop Camera')
        
        # Handle camera controls
        if start_button:
            if check_camera_access(camera_index):
                st.session_state.camera_running = True
            else:
                st.error(f"""
                    Cannot access camera {camera_index}. Please check:
                    1. Camera is properly connected
                    2. Camera permissions are granted
                    3. No other application is using the camera
                    4. Try a different camera index
                """)
        if stop_button:
            st.session_state.camera_running = False
        
        # Placeholder for video feed
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Load detector
        hog = load_detector()
        
        if st.session_state.camera_running:
            try:
                cap = cv2.VideoCapture(camera_index)
                
                # Check if camera opened successfully
                if not cap.isOpened():
                    st.error("Failed to open camera")
                    st.session_state.camera_running = False
                else:
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    # FPS calculation variables
                    fps_start_time = time.time()
                    fps_counter = 0
                    fps = 0
                    
                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame")
                            break
                        
                        # Process frame
                        processed_frame, num_detections = process_frame(
                            frame, hog, detection_confidence)
                        
                        # Calculate and display FPS
                        fps_counter += 1
                        if (time.time() - fps_start_time) > 1:
                            fps = fps_counter / (time.time() - fps_start_time)
                            fps_counter = 0
                            fps_start_time = time.time()
                        
                        if fps_display:
                            cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        # Convert BGR to RGB
                        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame and stats
                        video_placeholder.image(processed_frame, channels="RGB")
                        stats_placeholder.markdown(f"""
                            <div class="detection-stats">
                                🎯 Detected Humans: {num_detections} | ⚡ FPS: {fps:.1f}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Small delay to reduce CPU usage
                        time.sleep(0.01)
                
                cap.release()
                
            except Exception as e:
                st.error(f"""
                    Camera Error: {str(e)}
                    
                    Troubleshooting steps:
                    1. Check if camera is connected
                    2. Try restarting the application
                    3. Make sure no other application is using the camera
                    4. Try a different camera index
                """)
                st.session_state.camera_running = False
    
    with tab2:
        st.header("Upload Image for Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Process image
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            processed_img, num_humans = process_frame(img_array, hog, detection_confidence)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Detected Humans")
                st.image(processed_img, use_column_width=True)
            
            st.success(f"Found {num_humans} humans in the image!")
    
    with tab3:
        st.markdown("""
        ### 🎯 About This App
        
        This application provides real-time human detection using computer vision technology.
        
        ### 🚀 Features
        - Real-time webcam detection
        - Image upload and processing
        - Adjustable detection sensitivity
        - FPS monitoring
        - Multiple camera support
        
        ### 📝 How to Use
        1. Select your preferred camera from the sidebar
        2. Adjust the detection confidence if needed
        3. Click 'Start Camera' to begin real-time detection
        4. Or upload an image for static detection
        
        ### 🛠️ Technologies Used
        - Streamlit
        - OpenCV (Computer Vision)
        - Python
        - NumPy
        
        ### 👨‍💻 Created By
        [Your Name/Company]
        """)

if __name__ == "__main__":
    main()
