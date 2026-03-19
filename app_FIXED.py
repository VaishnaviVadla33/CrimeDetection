import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import EfficientNetB0
import cv2
import numpy as np
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration
import os

# --- 1. Define Constants ---
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (112, 112)
# Updated to match your error log
MODEL_PATH = 'gpu_violence_model_vaishu123.h5' 

# --- 2. Rebuild Model Architecture (CORRECTED) ---
def build_model_architecture():
    """Rebuild the exact model architecture from training"""
    input_shape = (SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    cnn_input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    # 1. Define Augmentation (This was missing from your old app.py)
    # This must be IDENTICAL to your training notebook
    data_augmentation = Sequential([
      Input(shape=input_shape), 
      TimeDistributed(RandomFlip("horizontal")),
      TimeDistributed(RandomRotation(0.1)),
      TimeDistributed(RandomZoom(0.1))
    ], name="data_augmentation")

    # 2. CNN base
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=cnn_input_shape
    )
    base_model.trainable = False
    
    # 3. Build model (Using Functional API to match notebook)
    model_input = Input(shape=input_shape)
    
    # 4. Add Augmentation Layer
    x = data_augmentation(model_input) 
    
    # 5. Add the CNN Base
    x = TimeDistributed(base_model)(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    
    # 6. Add "thinking" layer
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.3))(x)

    # 7. Add Bidirectional LSTM
    x = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2))(x)
    
    # 8. Add Final Classifier
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    model_output = Dense(1, activation='sigmoid')(x)

    # 9. Create the model
    model = Model(model_input, model_output)
    
    return model

# --- 3. Load Model (cached) ---
@st.cache_resource
def load_model_with_weights(model_path):
    """Load model by rebuilding architecture and loading weights"""
    try:
        # Build architecture
        model = build_model_architecture()
        
        # Load weights
        model.load_weights(model_path)
        print("Model weights loaded successfully!")
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- 4. Define the VideoTransformer Class ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Load the model when the class is created
        self.model = load_model_with_weights(MODEL_PATH)
        # Create a list to store the frames
        self.frame_buffer = []
        # Store the last prediction
        self.prediction_prob = 0.0
        self.prediction_text = "NON-VIOLENCE"
        # Frame counter to control prediction frequency
        self.frame_counter = 0
        self.prediction_interval = 5  # Predict every 5 frames for better performance

    def preprocess_frame(self, frame):
        """Resizes and normalizes a single frame."""
        frame_resized = cv2.resize(frame, IMAGE_SIZE)
        frame_normalized = frame_resized.astype('float32') / 255.0
        return frame_normalized

    def recv(self, frame):
        """This method is called for every frame from the webcam."""
        # Convert the frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocess the frame
        processed_frame = self.preprocess_frame(img)
        
        # Add the frame to our buffer
        self.frame_buffer.append(processed_frame)
        
        # --- Sliding Window ---
        if len(self.frame_buffer) >= SEQUENCE_LENGTH:
            # Keep only the last SEQUENCE_LENGTH frames
            self.frame_buffer = self.frame_buffer[-SEQUENCE_LENGTH:]
            
            # Only predict every N frames
            self.frame_counter += 1
            if self.frame_counter >= self.prediction_interval:
                self.frame_counter = 0
                
                # 1. Prepare data for the model
                input_data = np.expand_dims(np.array(self.frame_buffer), axis=0)
                
                # 2. Make prediction (if model is loaded)
                if self.model:
                    try:
                        self.prediction_prob = self.model.predict(input_data, verbose=0)[0][0]
                    except Exception as e:
                        print(f"Prediction error: {e}")

        # 3. Determine prediction text and color
        if self.prediction_prob > 0.5:
            self.prediction_text = "VIOLENCE"
            color = (0, 0, 255)  # Red
        else:
            self.prediction_text = "NON-VIOLENCE"
            color = (0, 255, 0)  # Green
        
        # 4. Draw the prediction on the frame
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        cv2.putText(
            img, 
            f"{self.prediction_text}", 
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            color, 
            3
        )
        
        confidence_text = f"Confidence: {self.prediction_prob:.2%}"
        cv2.putText(
            img, 
            confidence_text, 
            (10, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 255), 
            2
        )
        
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{SEQUENCE_LENGTH}"
        text_size = cv2.getTextSize(buffer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(
            img,
            buffer_text,
            (img.shape[1] - text_size[0] - 10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        bar_width = 300
        bar_height = 20
        bar_x = 10
        bar_y = img.shape[0] - 40
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        fill_width = int(bar_width * self.prediction_prob)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. Streamlit App Interface ---
st.set_page_config(
    page_title="Live Violence Detection",
    page_icon="📹",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>📹 Live Violence Detection (Webcam)</h1>", 
            unsafe_allow_html=True)

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: `{MODEL_PATH}`")
    st.info("Please place the trained model file in the same directory as this script.")
    st.stop()

# Try to load model first
with st.spinner("Loading model..."):
    test_model = load_model_with_weights(MODEL_PATH)

if test_model is None:
    st.error(f"❌ Failed to load model from `{MODEL_PATH}`")
    st.info("""
    **Troubleshooting:**
    1. Make sure the model file exists
    2. Verify the file is not corrupted
    3. Check that you have the correct TensorFlow version
    """)
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# Instructions
with st.expander("ℹ️ How to use", expanded=True):
    st.markdown(f"""
    **Instructions:**
    1. Click **START** below to activate your webcam
    2. Allow browser access to your camera when prompted
    3. The system will analyze video in real-time
    4. Wait for the buffer to fill ({SEQUENCE_LENGTH} frames)
    5. Predictions will update automatically
    6. Click **STOP** to end the session
    
    **What you'll see:**
    - 🟢 **Green**: Non-violent activity detected
    - 🔴 **Red**: Violent activity detected
    - **Confidence bar**: Shows prediction confidence
    - **Buffer status**: Shows how many frames are loaded
    """)

# Sidebar with settings and info
with st.sidebar:
    st.header("⚙️ Settings & Info")
    
    st.markdown("### 📊 Model Details")
    st.info(f"""
    - **Frames per prediction**: {SEQUENCE_LENGTH}
    - **Frame size**: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels
    - **Architecture**: EfficientNetB0 + BiLSTM
    - **Parameters**: ~4.8M
    """)
    
    st.markdown("### 🎯 Prediction Threshold")
    st.write("Values above 50% are classified as violence")
    st.progress(0.5)
    
    st.markdown("### ⚠️ Important Notes")
    st.warning("""
    - This is an AI model and may not be 100% accurate
    - Lighting and video quality affect performance
    - Wait for buffer to fill for initial prediction
    - Predictions update every few frames
    """)
    
    st.markdown("### 🔒 Privacy")
    st.info("All processing happens locally. No video data is uploaded or stored.")

# Main content area
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Webcam Feed")
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Start webcam streamer
    webrtc_ctx = webrtc_streamer(
        key="violence-detection",
        video_processor_factory=VideoTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 1280, "max": 1920},
                "height": {"min": 480, "ideal": 720, "max": 1080},
            },
            "audio": False
        },
        async_processing=True,
    )

with col2:
    st.markdown("### 📈 Status")
    
    # Status indicators
    status_container = st.container()
    
    with status_container:
        if webrtc_ctx.state.playing:
            st.success("🔴 Live")
            st.metric("Status", "Active")
        else:
            st.info("⚪ Standby")
            st.metric("Status", "Inactive")
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ensure good lighting
    - Keep camera steady
    - Position subjects clearly in frame
    - Allow time for buffer to fill
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Violence Detection System | Powered by Deep Learning</p>
    <p style='font-size: 0.8rem;'>EfficientNetB0 + Bidirectional LSTM Architecture</p>
</div>
""", unsafe_allow_html=True)