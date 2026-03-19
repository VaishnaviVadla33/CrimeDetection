import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
import cv2
import numpy as np
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration
import os
from collections import deque
import threading
import queue

# --- Configuration ---
SEQUENCE_LENGTH = 20
IMAGE_SIZE = (96, 96)
MODEL_PATH = 'best_violence_model_500.h5'
THRESHOLD_PATH = 'optimal_threshold_500.npy'

# --- Build Model Architecture ---
def build_model_architecture():
    input_shape = (SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        alpha=0.75
    )
    base_model.trainable = False
    
    model_input = Input(shape=input_shape)
    x = TimeDistributed(base_model)(model_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.4))(x)
    x = Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.3))(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    model_output = Dense(1, activation='sigmoid')(x)
    
    model = Model(model_input, model_output)
    return model

# --- Load Model (with progress) ---
@st.cache_resource
def load_model_and_threshold():
    try:
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
        
        model = build_model_architecture()
        model.load_weights(MODEL_PATH)
        
        # Compile with mixed precision for speed
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Warm up model with dummy prediction
        dummy_input = np.zeros((1, SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        threshold = 0.5
        if os.path.exists(THRESHOLD_PATH):
            threshold = float(np.load(THRESHOLD_PATH))
        
        return model, threshold
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 0.5

# --- Optimized VideoTransformer with Threading ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_prob = 0.0
        self.is_ready = False
        self.lock = threading.Lock()
        self.frame_count = 0
        self.skip_frames = 2  # Process every 3rd frame for speed
        
        # Prediction queue for async processing
        self.prediction_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self._prediction_worker, daemon=True)
        self.prediction_thread.start()
        
    def _prediction_worker(self):
        """Background thread for predictions"""
        while True:
            try:
                input_data = self.prediction_queue.get(timeout=1)
                if input_data is not None:
                    pred = float(self.model.predict(input_data, verbose=0)[0][0])
                    # Update result if queue is empty (don't block)
                    if self.result_queue.empty():
                        self.result_queue.put(pred)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prediction error: {e}")

    def preprocess_frame(self, frame):
        """Fast preprocessing"""
        frame_resized = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return frame_resized.astype('float32') / 255.0

    def recv(self, frame):
        """Process frame"""
        img = frame.to_ndarray(format="bgr24")
        
        # Increment frame counter
        self.frame_count += 1
        
        # Only process every Nth frame to reduce lag
        if self.frame_count % (self.skip_frames + 1) == 0:
            # Preprocess and add to buffer
            processed_frame = self.preprocess_frame(img)
            
            with self.lock:
                self.frame_buffer.append(processed_frame)
            
            # Once buffer is full, send to prediction thread
            if len(self.frame_buffer) == SEQUENCE_LENGTH:
                if not self.is_ready:
                    self.is_ready = True
                
                # Send to prediction queue (non-blocking)
                if self.prediction_queue.empty():
                    input_data = np.expand_dims(np.array(self.frame_buffer), axis=0)
                    try:
                        self.prediction_queue.put_nowait(input_data)
                    except queue.Full:
                        pass
        
        # Always try to get latest result (non-blocking)
        try:
            self.prediction_prob = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Determine label and color
        # High probability = Violence detected
        if self.prediction_prob > self.threshold:
            label = "⚠️ VIOLENCE DETECTED"
            color = (0, 0, 255)  # Red
            bg_color = (0, 0, 120)
            confidence_label = "DANGER"
        else:
            label = "✓ SAFE"
            color = (0, 255, 0)  # Green
            bg_color = (0, 120, 0)
            confidence_label = "SAFE"
        
        # Draw overlay
        h, w = img.shape[:2]
        
        # Top bar with label
        cv2.rectangle(img, (0, 0), (w, 80), bg_color, -1)
        cv2.putText(img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Show violence probability (not just generic confidence)
        violence_text = f"Violence: {self.prediction_prob*100:.1f}%"
        cv2.putText(img, violence_text, (w - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Buffer status
        if not self.is_ready:
            status = f"Buffering: {len(self.frame_buffer)}/{SEQUENCE_LENGTH}"
            cv2.putText(img, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Confidence bar
        bar_h = 8
        bar_y = h - bar_h
        fill_w = int(w * self.prediction_prob)
        cv2.rectangle(img, (0, bar_y), (w, h), (40, 40, 40), -1)
        cv2.rectangle(img, (0, bar_y), (fill_w, h), color, -1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.set_page_config(page_title="Violence Detection", page_icon="🚨", layout="wide")

st.markdown("""
<style>
    .stApp {max-width: 100%; padding: 0.5rem;}
    h1 {text-align: center; margin: 0.5rem 0;}
    .stAlert {margin: 0.3rem 0; padding: 0.5rem;}
</style>
""", unsafe_allow_html=True)

st.title("🚨 Real-Time Violence Detection")

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.info("Please place the model file in the same directory.")
    st.stop()

# Initialize session state first
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.threshold = 0.5

# Load model BEFORE webrtc_streamer
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading model... Please wait..."):
        model, threshold = load_model_and_threshold()
        
        if model is None:
            st.error("❌ Failed to load model")
            st.stop()
        
        st.session_state.model = model
        st.session_state.threshold = threshold
        st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully!")

model = st.session_state.model
threshold = st.session_state.threshold

# Safety check
if model is None:
    st.error("❌ Model not loaded. Please refresh the page.")
    st.stop()

# Show info
col1, col2, col3, col4 = st.columns(4)
col1.metric("Threshold", f"{threshold:.3f}")
col2.metric("Frames", SEQUENCE_LENGTH)
col3.metric("Size", f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
col4.metric("Status", "✅ Ready")

st.info("""
**Instructions:** Click START → Allow camera → Wait for buffer (20 frames) → Detection active!

**🎬 Actions to Test Violence Detection:**

**HIGH VIOLENCE (Should show RED):**
- 👊 **Punch motions** - Fast, aggressive arm movements toward camera
- 🤜 **Hitting motions** - Swing arms like hitting someone
- 🦵 **Kicking motions** - Fast leg movements
- ✊ **Fighting gestures** - Two people mock fighting
- 🤛 **Grabbing/choking** - Aggressive grabbing motions
- 💥 **Slapping** - Fast hand movements toward face area

**LOW VIOLENCE (Should show GREEN):**
- 👋 **Waving** - Normal, friendly waving
- 🚶 **Walking** - Normal walking around
- 💺 **Sitting still** - No movement
- 📱 **Using phone** - Normal phone usage
- 🗣️ **Talking** - Just talking to camera
- ✋ **Slow hand movements** - Gentle, non-aggressive

**💡 Tips for Testing:**
- Be **sudden and aggressive** for violence detection
- Use **fast, jerky movements** (like real fighting)
- Multiple **rapid punches** work best
- Try **2 people** mock fighting for better results
""")

st.markdown("---")

# WebRTC configuration
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Capture model and threshold in local scope for factory
current_model = model
current_threshold = threshold

# Create video transformer factory (using closure to capture variables)
def video_processor_factory():
    return VideoTransformer(current_model, current_threshold)

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="violence-detection-v2",
    video_processor_factory=video_processor_factory,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    async_processing=True,
    sendback_audio=False,
)

# Status
st.markdown("---")
if webrtc_ctx.state.playing:
    st.success("🔴 LIVE - Detection Active")
else:
    st.warning("⚪ Click START above to begin detection")

# Compact sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.code(f"""
Threshold: {threshold:.4f}
Frames: {SEQUENCE_LENGTH}
Resolution: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}
Processing: Every 3rd frame
""")
    
    st.markdown("### 🎯 Performance")
    st.success("""
✅ Optimized for speed
- Skips 2 frames between processing
- Async predictions
- Lower resolution (96x96)
""")
    
    st.markdown("### 📹 Testing Tips")
    st.warning("""
**For RED Alert:**
- Fast punching motions
- Aggressive hitting
- Rapid arm swings
- Mock fighting (2 people)

**Should stay GREEN:**
- Normal movements
- Talking, waving
- Slow gestures
""")
    
    if st.button("🔄 Reload Model"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

st.caption("Continuous real-time detection | Optimized for speed")