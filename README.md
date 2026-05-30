# 😊 Emotion Detection using Deep Learning

A real-time facial emotion recognition system built with TensorFlow and MobileNetV2 (via TensorFlow Hub). The model classifies human facial expressions into **7 emotion categories** and can run live predictions through a webcam.

---

## 🎭 Emotions Detected

| Emotion   | Label       |
|-----------|-------------|
| 😊 Happy    | `happy`     |
| 😲 Surprised| `surprised` |
| 😐 Neutral  | `neutral`   |
| 😢 Sad      | `sad`       |
| 😠 Angry    | `angry`     |
| 😨 Fearful  | `fear`      |
| 🤢 Disgusted| `disgust`   |

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (130 width multiplier, 224×224 input) via [TensorFlow Hub](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4)
- **Fine-tuning:** Full model is trainable (`trainable=True`)
- **Data Augmentation:** Random horizontal flip, rotation (±10%), zoom (±10%)
- **Output Layer:** Dense layer with Softmax activation (7 classes)
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam (`lr=1e-5`)
- **Input Size:** 224 × 224 × 3 (RGB)

---

## 📊 Dataset

The model was trained on a dataset with the following structure:

```
archive-2/
├── train/
│   ├── happy/
│   ├── surprised/
│   ├── neutral/
│   ├── sad/
│   ├── angry/
│   ├── fearful/
│   └── disgusted/
└── test/
    └── (same structure as train/)
```
Taken from kaggle --> https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
- **Training samples used:** 8,000 (stratified split, 80/20 train-val)
- **Class imbalance handling:** Balanced class weights via `sklearn.utils.class_weight`
- **Image format:** 48×48 px grayscale images, resized to 224×224 RGB for the model

---

## ⚙️ Setup & Installation

### Prerequisites

```bash
pip install tensorflow tensorflow-hub tensorflow-keras
pip install pandas numpy matplotlib seaborn scikit-learn
pip install opencv-python   # for webcam script
```

### Clone & Run

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

---

## 🏋️ Training (Notebook)

Open `emotion_detection-notebook.ipynb` in **Google Colab** (GPU recommended).

1. Mount Google Drive and place the dataset at `MyDrive/archive-2/`
2. Run all cells sequentially
3. The trained model is saved to `MyDrive/emotion_detection_model.h5`

**Training configuration:**
- Epochs: up to 50 (with early stopping)
- Batch size: 32
- Early stopping: monitors `val_accuracy`, patience = 5, restores best weights
- TensorBoard logging: enabled

---

## 📷 Real-Time Webcam Detection

Run the webcam script after the model has been trained and saved:

```bash
python live_emotion.py
```

This opens your webcam feed and overlays the predicted emotion label in real time on detected faces.

---

## 🔮 Inference (Single Image)

```python
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as legacy_keras
import numpy as np

# Load model
loaded_model = legacy_keras.models.load_model(
    "emotion_detection_model.h5",
    custom_objects={"KerasLayer": hub.KerasLayer}
)

unique_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

def predict_emotion(image_path):
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, [224, 224]) / 255.0
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    predictions = loaded_model.predict(image)
    emotion = unique_emotions[np.argmax(predictions)]
    confidence = predictions[0][np.argmax(predictions)]
    print(f"Predicted: {emotion} ({confidence:.2%} confidence)")

predict_emotion("path/to/face.jpg")
```

---

## 📈 Training Monitoring

TensorBoard logs are saved during training. To visualize:

```bash
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/log
```

---

## 🛠️ Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| MobileNetV2 backbone | Lightweight yet accurate; fast inference for real-time use |
| `trainable=True` | Full fine-tuning for better emotion-specific feature learning |
| Class weights | Dataset is imbalanced (disgust has fewer samples); weights compensate |
| Low learning rate (1e-5) | Prevents catastrophic forgetting during fine-tuning |
| Early stopping | Avoids overfitting; restores best-performing checkpoint |

---

## 📝 Notes

- The notebook is designed to run on **Google Colab** with Google Drive mounted.
- Images are originally 48×48 grayscale but are forced to 3-channel RGB for MobileNetV2 compatibility.
- The `disgust` class has significantly fewer samples; class weights and data augmentation help address this.

---

## 📄 License

This project is for educational purposes. Dataset and pretrained weights are subject to their respective licenses.

## veiw Notebook
[![View Notebook](https://img.shields.io/badge/Rendered%20with-NBViewer-orange?style=for-the-badge&logo=jupyter)](https://nbviewer.org/github/omprakash20071125-a11y/emotion_detection_deeplearning_project/blob/main/emotion_detection-notebook.ipynb)
