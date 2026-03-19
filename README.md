# Crime Detection & Violence Detection System

A Streamlit-based real-time violence detection application using deep learning (CNN + LSTM architecture) for crime surveillance analysis.

## 📋 Project Overview

This project implements a video analysis system that uses deep neural networks to detect violent activities in surveillance footage. It leverages:
- **EfficientNetB0 / MobileNetV2** for feature extraction (CNN)
- **LSTM** layers for temporal sequence analysis
- **Bidirectional LSTM** for improved context understanding
- **Real-time inference** with pre-trained models

## 📁 Project Structure

```
cv/
├── app.py                           # Main Streamlit app (EfficientNetB0 + LSTM)
├── app_FIXED.py                     # Improved version with data augmentation
├── app_webcam.py                    # Real-time webcam/webrtc streaming version
├── best_violence_model_500.h5       # Pre-trained violence detection model
├── violence_detection_500videos.h5  # Alternative model variant
├── optimal_threshold_500.npy        # Optimal decision threshold
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager
- Webcam or video file

### Installation

1. **Clone/Download the project**
   ```bash
   cd cv/
   ```

2. **Create a virtual environment** (Optional but recommended)
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📺 Running the Application

### Option 1: Basic Violence Detection (app.py)
Detects violence in uploaded video files.
```bash
streamlit run app.py
```

### Option 2: Improved Version with Augmentation (app_FIXED.py)
Enhanced version with data augmentation and better model handling.
```bash
streamlit run app_FIXED.py
```

### Option 3: Real-time Webcam Detection (app_webcam.py)
Live detection using your webcam with real-time visualization.
```bash
streamlit run app_webcam.py
```

## 🎯 Features

### Model Architecture
```
Input (30 frames, 112x112x3)
    ↓
TimeDistributed(EfficientNetB0)
    ↓
TimeDistributed(GlobalAveragePooling2D)
    ↓
TimeDistributed(Dense + Dropout)
    ↓
Bidirectional LSTM
    ↓
Dense Layers + Dropout
    ↓
Output (Binary: Violence/No Violence)
```

### Key Features
- ✅ **Real-time detection** from webcam or video files
- ✅ **Sequence-based analysis** (30-frame windows for temporal context)
- ✅ **Confidence scoring** with adjustable threshold
- ✅ **GPU acceleration** support
- ✅ **Stream processing** for efficient memory usage

## 📊 Model Information

| Model | Frames | Image Size | Base Network | Accuracy |
|-------|--------|------------|--------------|----------|
| `best_violence_model_500.h5` | 30 | 112x112 | MobileNetV2 | ~90% |
| `violence_detection_500videos.h5` | 30 | 112x112 | EfficientNetB0 | ~92% |

**Optimal Threshold:** `optimal_threshold_500.npy` (Improves precision/recall balance)

## ⚙️ Configuration

### Adjust these parameters in the app:

```python
SEQUENCE_LENGTH = 30        # Number of frames per clip
IMAGE_SIZE = (112, 112)     # Frame resolution
THRESHOLD = 0.5             # Decision threshold (0-1)
```

## 📝 Usage Examples

### Video Upload
1. Run the app
2. Upload a video file (MP4, AVI, MOV)
3. App extracts 30-frame sequences
4. Displays detection results with confidence scores

### Live Webcam
1. Run `app_webcam.py`
2. Grant camera permissions
3. System processes frames in real-time
4. Alerts trigger when violence is detected

## 🔧 Dependencies

See `requirements.txt` for complete list. Key packages:
- **streamlit** - Web framework
- **tensorflow/keras** - Deep learning
- **opencv-python** - Video processing
- **numpy** - Numerical computing
- **streamlit-webrtc** - Real-time video streaming (webcam version)
- **av** - Audio/video processing

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Ensure `.h5` files are in same directory |
| OOM (Out of Memory) | Reduce `SEQUENCE_LENGTH` or use MobileNetV2 |
| Webcam not working | Check permissions and try `app.py` with video file |
| Slow inference | Use GPU (CUDA enabled TensorFlow) |

## 📈 Performance Tips

1. **For faster inference:** Use `MobileNetV2` variant
2. **For better accuracy:** Use `EfficientNetB0` variant
3. **For real-time:** Process every nth frame (e.g., every 2nd frame)
4. **GPU acceleration:** Install `tensorflow-gpu`

## 📚 Model Training

Models trained on:
- 500 video samples
- 14 crime/action categories (Abuse, Assault, Robbery, etc.)
- 70% training / 30% testing split

**Dataset source:** Crime subset from surveillance footage databases

## 🛡️ Ethical Considerations

- Use only on authorized surveillance systems
- Comply with local privacy regulations
- Maintain proper data handling practices
- Regular model validation on diverse data

## 📄 License

[Add your license here - e.g., MIT, Apache 2.0]

## 👥 Authors

- **Your Name** - Computer Vision Project, Semester 7

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review model paths and file locations
3. Verify all dependencies are installed

## 🎓 Project References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Streamlit Documentation: https://docs.streamlit.io/
- EfficientNet Paper: https://arxiv.org/abs/1905.11946
- LSTM for Action Recognition: https://arxiv.org/abs/1502.01852

---

**Last Updated:** March 2026  
**Status:** Production Ready
