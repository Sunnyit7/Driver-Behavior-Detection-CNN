# Driver Behavior Detection Using CNN and VGG16

A deep learning-based driver behavior classification system that identifies driver activities from images. The project compares a **Custom Convolutional Neural Network (CNN)** with a **VGG16 transfer-learning model** and deploys the best-performing model using a Flask web application.

The trained models are evaluated using test accuracy, precision, recall, F1-score, classification reports, and confusion matrices. Based on the evaluation results, **VGG16 was selected for deployment**.

---

## Project Overview

Distracted driving is a major road-safety concern. This project uses computer vision and deep learning to classify driver behavior into five activity categories.

The complete workflow includes:

1. Dataset validation and preprocessing
2. Train-validation-test dataset splitting
3. Custom CNN model development
4. VGG16 transfer learning
5. Model training using callbacks
6. Model evaluation
7. Performance comparison
8. Best-model selection
9. Flask-based web deployment

---

## Driver Behavior Classes

The model classifies images into the following five categories:

- Other Activities
- Safe Driving
- Talking Phone
- Texting Phone
- Turning

---

## Models Implemented

### 1. Custom CNN

A custom Convolutional Neural Network was developed from scratch using:

- Convolutional layers
- Max-pooling layers
- Flatten layer
- Fully connected dense layer
- Dropout regularization
- Softmax output layer

### 2. VGG16 Transfer Learning

The VGG16 architecture was initialized using pretrained ImageNet weights.

The convolutional base was frozen and a custom classification head was added using:

- Flatten layer
- Dense layer with 256 neurons
- Dropout regularization
- Softmax output layer for five-class classification

---

## Dataset Information

The dataset contains:

| Dataset Split | Number of Images |
|---|---:|
| Training | 7,525 |
| Validation | 1,613 |
| Testing | 1,613 |
| Total | 10,751 |

The training dataset uses image augmentation techniques including:

- Rotation
- Zoom
- Width shifting
- Height shifting
- Horizontal flipping

All images are resized to `224 × 224` pixels and normalized to the range `[0, 1]`.

---

## Model Evaluation

Both models were evaluated on the same unseen test dataset using:

- Test loss
- Test accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Classification report
- Confusion matrix

### VGG16 Test Results

| Metric | Score |
|---|---:|
| Test Loss | 0.3016 |
| Test Accuracy | 91.51% |
| Precision | 91.56% |
| Recall | 91.51% |
| Weighted F1-score | 91.39% |

VGG16 achieved the best overall performance and was selected for Flask deployment.

---

## Model Comparison

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 67.82% |
| VGG16 | 91.51% |

VGG16 improved test accuracy by approximately **23.69 percentage points** compared with the Custom CNN.

The complete comparison results are available in:

```text
outputs/comparison/
├── model_comparison.csv
└── model_comparison.png
```

---

## Flask Web Application

The Flask application allows users to:

1. Upload a driver image
2. Validate the uploaded image
3. Preprocess the image
4. Generate a prediction using the trained VGG16 model
5. Display the predicted driver behavior
6. Display the model confidence score
7. Preview the uploaded image

Application workflow:

```text
Upload Driver Image
        ↓
Validate Image
        ↓
Resize to 224 × 224
        ↓
Normalize Pixel Values
        ↓
VGG16 Prediction
        ↓
Display Behavior and Confidence
```

---

## Project Structure

```text
Driver_Behavior_Detection_CNN/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── custom_cnn.py
│   ├── vgg16_model.py
│   ├── model_factory.py
│   ├── train.py
│   ├── evaluate.py
│   └── compare_models.py
│
├── saved_models/
│   └── .gitkeep
│
├── outputs/
│   ├── custom_cnn/
│   │   ├── metrics.json
│   │   ├── classification_report.txt
│   │   └── confusion_matrix.png
│   │
│   ├── vgg16/
│   │   ├── metrics.json
│   │   ├── classification_report.txt
│   │   └── confusion_matrix.png
│   │
│   └── comparison/
│       ├── model_comparison.csv
│       └── model_comparison.png
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    │   └── style.css
    │
    └── uploads/
        └── .gitkeep
```

---

## Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- Convolutional Neural Networks
- VGG16
- Transfer Learning

### Data Processing and Evaluation

- NumPy
- Pandas
- Scikit-learn

### Data Visualization

- Matplotlib

### Web Development

- Flask
- HTML
- CSS

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Driver-Behavior-Detection-CNN
```

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained VGG16 model

Place the trained deployment model at:

```text
saved_models/vgg16_best.keras
```

The trained model is not included in the repository because its file size exceeds GitHub's standard file-size limit.

### 5. Run the Flask application

```bash
python app.py
```

Open the local address displayed in the terminal, typically:

```text
http://127.0.0.1:5000
```

---

## Training

Select the required model in `src/train.py`.

For the Custom CNN:

```python
MODEL_NAME = "custom_cnn"
```

For VGG16:

```python
MODEL_NAME = "vgg16"
```

Run:

```bash
python src/train.py
```

The best and final trained models are saved inside:

```text
saved_models/
```

---

## Evaluation

Select the model in `src/evaluate.py`:

```python
MODEL_NAME = "vgg16"
```

Run:

```bash
python src/evaluate.py
```

The evaluation script generates:

- Test metrics
- Classification report
- Confusion matrix

---

## Compare Models

Run:

```bash
python src/compare_models.py
```

The comparison script generates:

```text
outputs/comparison/
├── model_comparison.csv
└── model_comparison.png
```

The best-performing model is selected using the weighted F1-score.

---

## Future Improvements

- Add real-time webcam-based driver behavior detection
- Add video-stream inference
- Deploy the Flask application to a cloud platform
- Add temporal behavior analysis across video frames
- Improve generalization using additional driver datasets
- Add alert generation for distracted-driving behavior

---

## Conclusion

The project demonstrates an end-to-end deep learning workflow for driver behavior classification. A Custom CNN and VGG16 transfer-learning model were trained and evaluated using the same dataset. VGG16 achieved a test accuracy of **91.51%** and a weighted F1-score of **91.39%**, making it the best-performing model for deployment.

The selected VGG16 model was integrated into a Flask web application that accepts driver images and displays the predicted behavior with a confidence score.