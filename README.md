# 🌿 Plant Disease Detection using CNN

## 📋 Project Overview

This project aims to **detect plant diseases** from leaf images using a **Convolutional Neural Network (CNN)**.  
It classifies images of crops (like **Potato, Tomato, Corn**) into their respective disease categories, helping farmers identify diseases early and take corrective actions.

---

## 🚀 Tech Stack

- **Python 3**
- **TensorFlow**
- **NumPy, Matplotlib, Pandas**
- **Google Colab**

---

## 🏗️ Project Architecture

- **Input**: Images resized to **128 × 128** pixels.
- **Convolution Layers**: For feature extraction.
- **ReLU Activation**: To introduce non-linearity.
- **MaxPooling Layers**: For dimensionality reduction.
- **Dropout Layers**: To prevent overfitting.
- **Flatten Layer**: To transform features into a 1D vector.
- **Fully Connected Dense Layers**: For classification.
- **Softmax Activation**: To predict the class probabilities.

---

## 📁 Dataset

- **Source**: PlantVillage Dataset
- **Classes**: Healthy and various disease classes.
- **Preprocessing**:  
  - Image resizing to 128×128
  - Normalization (pixel values between 0 and 1)
  - Data Augmentation (optional)

---
## 📊 Results

- **Training Accuracy**: 97.03%
- **Validation Accuracy**: 93.83%
- **Test Accuracy**: 93.6%
- **Loss Curves and Accuracy Curves** plotted.

---

## 🔥 Key Features

- Simple and efficient CNN architecture
- Works well even on small datasets
- Supports multiclass disease classification
- Lightweight and easy to deploy

---

## ✨ Future Improvements

- Fine-tuning with Transfer Learning (like using VGG16, MobileNet)
- Model deployment as a Web App
- Real-time detection via mobile camera

---

## 🤝 Contributing

Feel free to submit issues or pull requests if you'd like to contribute!  
Every contribution is appreciated 💖

---
