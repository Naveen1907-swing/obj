import streamlit as st
import cv2
import numpy as np
from PIL import Image

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
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .detection-stats {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .upload-section {
        padding: 2rem;
        border-radius: 1rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load and cache the HOG detector"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def detect_humans(image, confidence_threshold=0.3):
    """Detect humans in the image"""
    hog = load_detector()
    
    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Resize image for better performance while maintaining aspect ratio
    max_dimension = 800
    height, width = image_cv.shape[:2]
    scale = min(max_dimension/width, max_dimension/height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_cv = cv2.resize(image_cv, (new_width, new_height))
    
    # Detect humans
    try:
        boxes, weights = hog.detectMultiScale(
            image_cv,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        
        # Filter detections based on confidence
        confident_detections = [
            box for box, weight in zip(boxes, weights) 
            if weight > confidence_threshold
        ]
        
        # Scale boxes back if image was resized
        if scale < 1:
            confident_detections = [
                [int(x/scale), int(y/scale), int(w/scale), int(h/scale)]
                for (x, y, w, h) in confident_detections
            ]
        
        # Draw boxes
        for (x, y, w, h) in confident_detections:
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_cv, 'Human', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert back to RGB
        result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        return result_image, len(confident_detections)
        
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return image, 0

def main():
    st.title("👥 Human Detection App")
    
    # Sidebar settings
    with st.sidebar:
        st.title("⚙️ Settings")
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Adjust the confidence threshold for detection"
        )
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📷 Detection", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Image for Human Detection")
        
        with st.container():
            uploaded_file = st.file_uploader(
                "Choose an image file...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image to detect humans"
            )
            
            if uploaded_file is not None:
                try:
                    # Load and process image
                    image = Image.open(uploaded_file)
                    
                    # Create columns for before/after
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_column_width=True)
                    
                    with st.spinner('Detecting humans...'):
                        processed_image, num_humans = detect_humans(image, confidence)
                    
                    with col2:
                        st.subheader("Detected Humans")
                        st.image(processed_image, use_column_width=True)
                    
                    # Show results
                    if num_humans > 0:
                        st.success(f"✨ Found {num_humans} human{'s' if num_humans > 1 else ''} in the image!")
                    else:
                        st.info("No humans detected in the image.")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    with tab2:
        st.markdown("""
        ### 🎯 About This App
        
        This application uses computer vision to detect humans in images using the HOG 
        (Histogram of Oriented Gradients) detector combined with SVM (Support Vector Machine) 
        classification.
        
        ### 🚀 Features
        - Upload and process images
        - Adjust detection sensitivity
        - Real-time results
        - Support for multiple image formats
        
        ### 📝 How to Use
        1. Upload an image using the file uploader
        2. Adjust the confidence threshold if needed
        3. View the results with detected humans highlighted
        
        ### 🛠️ Technologies Used
        - Streamlit
        - OpenCV (Computer Vision)
        - Python
        - NumPy
        
        ### 👨‍💻 Created By
        [Your Name/Company]
        
        ### 📮 Contact
        For questions or feedback, please contact: [Your Contact Info]
        """)

if __name__ == "__main__":
    main()
