# Driver Distraction Detection using Deep Learning

## Overview

This project focuses on detecting distracted driving behaviors using a deep learning-based image classification model. The system identifies whether a driver is engaged in safe driving or distracted activities such as using a phone, drinking.

A Streamlit web application allows users to upload an image and receive predictions along with confidence scores.

---

## Features

- Driver activity classification using deep learning
- Streamlit-based interactive web application
- Confidence score for predictions
- Detection of distracted driving behaviors
- Image upload support for testing predictions

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/buvana-ML/Driver-Distraction-Detection.git
cd Driver-Distraction-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## Results

### Home Interface

![Home](images/homescreen.png)

### Example Prediction

![Prediction](images/prediction_phone.png)
![Prediction](images/prediction_drinking.png)

### Safe Driving Prediction

![Safe Driving](images/prediction_safe.png)

---

## Dataset

This project uses the State Farm Distracted Driver Detection dataset available on Kaggle.

---

## Limitations

- Model performance depends on similarity to training data
- Real-world generalization is limited
- Safe driving predictions may show lower confidence due to fewer distinctive visual features

---

## Future Improvements

- Improve model generalization with better augmentation
- Add real-time webcam detection
- Use transfer learning for higher accuracy
- Deploy as a cloud-based application

---

## Author

Buvana Sri Namachivayam
````
