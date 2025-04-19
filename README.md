# Optical Character Recognition System with Deep Learning

## Project Overview
This project implements a complete OCR pipeline combining computer vision and natural language processing techniques. The system detects text regions using the EAST model, recognizes text via CRNN, and optionally translates/summarizes content using sequence-to-sequence models.

## Development Timeline & Technical Components

### Phase 1: Foundations
#### Autoencoders and Recurrent Neural Networks
**Objective**: Establish fundamental understanding of feature extraction and sequential data processing.

**Key Implementations**:
1. **Basic Autoencoder(MNIST Digit Classifier)**:
   - Implemented on MNIST dataset to learn compressed representations
   - Architecture:
     ```python
     class Autoencoder(nn.Module):
         def __init__(self):
             super().__init__()
             self.encoder = nn.Sequential(
                 nn.Linear(784, 256),
                 nn.ReLU(),
                 nn.Linear(256, 64))
             self.decoder = nn.Sequential(
                 nn.Linear(64, 256),
                 nn.ReLU(),
                 nn.Linear(256, 784))
     ```
![image](https://github.com/user-attachments/assets/e80305d9-61f0-48fb-ba03-0acabd81e3a4)

2. **Word Embeddings (Word2Vec)**:
   - Implemented both CBOW and Skip-gram architectures
   - Trained on custom OCR corpus containing 5,000+ text samples
   - Key hyperparameters:
     - Embedding dimension: 300
     - Window size: 5
     - Negative sampling: 10
![image](https://github.com/user-attachments/assets/5495798d-e21f-4980-97d9-677f79ab5ded)

3. **Seq2Seq Model**:
   - Built with attention mechanism for text translation
   - Architecture highlights:
     ```python
     class Seq2Seq(nn.Module):
         def __init__(self, input_size, hidden_size):
             self.encoder = nn.LSTM(input_size, hidden_size)
             self.attention = BahdanauAttention(hidden_size)
             self.decoder = nn.LSTM(hidden_size*2, hidden_size)
     ```
   - Achieved BLEU-4 score of 0.58 on English-French translation task

### Phase 2: Text Detection - EAST Model
**Objective**: Localize text regions in images with arbitrary orientations.

**Technical Implementation**:
- Used OpenCV's EAST implementation (frozen_graph.pb)
- Key components:
  - **Score Map**: Pixel-level text/non-text classification
  - **Geometry Map**: Predicts:
    - Bounding box offsets (4 channels)
    - Rotation angle (1 channel)
- Detection code snippet:
  ```python
  def detect_text(image):
      blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320), 
                                 (123.68, 116.78, 103.94))
      net.setInput(blob)
      scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                    "feature_fusion/concat_3"])
  ```
### Phase 3: Text Recognition - CRNN
**Objective**: Recognize text from detected regions using hybrid CNN-RNN architecture.
![image](https://github.com/user-attachments/assets/7a28ccf4-421f-4da4-8b80-d935678fe0e4)

**Model Architecture**:

- CNN Backbone (Feature Extraction):

```python
x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
x = MaxPool2D(pool_size=(2,2))(x)
```
- Sequence Modeling (Bidirectional LSTMs):

```python
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
```
- CTC Decoding:

```python
decoded = tf.keras.backend.ctc_decode(
    y_pred, 
    input_length=np.ones(y_pred.shape[0])*y_pred.shape[1],
    greedy=True)
```
