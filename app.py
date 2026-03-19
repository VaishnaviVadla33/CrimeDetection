import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
import cv2
import numpy as np
import os

# --- 1. Define Constants ---
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (112, 112)
MODEL_PATH = 'best_capsnet_model.h5'

# --- 2. Rebuild Model Architecture ---
def build_model_architecture():
    """Rebuild the exact model architecture from training"""
    input_shape = (SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    # CNN base
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape[1:]
    )
    base_model.trainable = False
    
    # Build model
    model = Sequential([
        Input(shape=input_shape),
        tf.keras.layers.TimeDistributed(base_model),
        tf.keras.layers.TimeDistributed(GlobalAveragePooling2D()),
        tf.keras.layers.TimeDistributed(Dense(256, activation='relu')),
        tf.keras.layers.TimeDistributed(Dropout(0.3)),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    return model

# --- 3. Frame Extraction ---
def extract_frames(video_path):
    """Extract, resize, and normalize frames from video"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
    else:
        st.error(f"Error: Could not read video {video_path} or it has 0 frames.")
        return None

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            frame = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        
        frame = cv2.resize(frame, IMAGE_SIZE)
        frame = frame.astype('float32') / 255.0
        frames.append(frame)
        
    cap.release()
    
    if len(frames) == SEQUENCE_LENGTH:
        return np.array(frames)
    else:
        st.error(f"Error: Extracted {len(frames)} frames, but expected {SEQUENCE_LENGTH}.")
        return None

# --- 4. Load Model ---
@st.cache_resource
def load_model_with_weights(model_path):
    """Load model by rebuilding architecture and loading weights"""
    try:
        with st.spinner("Building model architecture..."):
            model = build_model_architecture()
        
        with st.spinner("Loading trained weights..."):
            # Try to load just the weights
            try:
                model.load_weights(model_path)
                st.success("Model weights loaded successfully!")
            except:
                # If that fails, try loading the whole model
                loaded_model = tf.keras.models.load_model(model_path, compile=False)
                model.set_weights(loaded_model.get_weights())
                st.success("Model loaded successfully!")
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- 5. Streamlit Interface ---
st.set_page_config(page_title="Violence Detection", page_icon="🎥", layout="wide")

st.title("🎥 Violence Detection System")
st.markdown("""
This application uses deep learning to analyze video content and detect potential violence.
Upload a video file to get started.
""")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown(f"""
    **Model Details:**
    - Sequence Length: {SEQUENCE_LENGTH} frames
    - Image Size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}
    - Architecture: EfficientNetB0 + BiLSTM
    
    **How it works:**
    1. Extract {SEQUENCE_LENGTH} frames from video
    2. Process each frame
    3. Analyze temporal patterns
    4. Predict violence probability
    """)

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: `{MODEL_PATH}`")
    st.info("Please place the trained model file in the same directory as this script.")
    st.stop()

# Load model
model = load_model_with_weights(MODEL_PATH)

if model is not None:
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            st.video(uploaded_file)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### Video Info")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
    
    # Prediction
    if uploaded_file is not None:
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            # Save temp file
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Process video
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Extracting frames...")
                progress_bar.progress(30)
                video_frames = extract_frames(temp_file_path)
                
                if video_frames is not None:
                    status_text.text("Preparing data...")
                    progress_bar.progress(60)
                    video_frames_batch = np.expand_dims(video_frames, axis=0)
                    
                    status_text.text("Making prediction...")
                    progress_bar.progress(80)
                    prediction_prob = model.predict(video_frames_batch, verbose=0)[0][0]
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    # Create columns for results
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        if prediction_prob > 0.5:
                            st.error("### 🚨 VIOLENCE DETECTED")
                            st.metric("Violence Confidence", f"{prediction_prob:.1%}")
                        else:
                            st.success("### ✅ NO VIOLENCE DETECTED")
                            st.metric("Non-Violence Confidence", f"{(1-prediction_prob):.1%}")
                    
                    with res_col2:
                        st.markdown("#### Confidence Distribution")
                        st.progress(float(prediction_prob), text=f"Violence: {prediction_prob:.1%}")
                        st.progress(float(1-prediction_prob), text=f"Non-Violence: {(1-prediction_prob):.1%}")
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
            
            finally:
                # Cleanup
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
else:
    st.error("Failed to load the model. Please check the error messages above.")