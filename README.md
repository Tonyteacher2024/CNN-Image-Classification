# CNN for Image Classification

This repository contains the code and documentation for the image classification project using Convolutional Neural Networks (CNN).

## 1. Installations
This project is implemented in Python using Jupyter Notebook. The required Python packages for this project are as follows:

- TensorFlow (including Keras)
- NumPy
- Matplotlib


---

## 2. Project Motivation
This project aims to leverage the power of CNNs to perform binary image classification. Specifically, it addresses the problem of classifying images of cats and dogs. The objectives of this project are:

1. Build and train a CNN model on labeled image data.
2. Evaluate the model on unseen test data to verify its accuracy and performance.
3. Use the trained model to predict new images provided in the `single_prediction` directory.

By achieving these goals, the project demonstrates the practical application of CNNs in real-world classification tasks.

---

## 3. Dataset
The dataset used in this project consists of:

- **Training Data**: 4,000 images of cats and 4,000 images of dogs.
- **Testing Data**: 1,000 images of cats and 1,000 images of dogs.

The dataset was preprocessed to ensure all images are of consistent size. Additionally, data augmentation techniques were applied to enhance the generalization ability of the model.

---

## 4. Model Architecture
The CNN model was built using TensorFlow's integrated Keras API. The architecture includes:

1. **Convolutional Layers**: Extract spatial features from input images.
2. **Max-Pooling Layers**: Reduce dimensionality and retain important features.
3. **Dropout Layers**: Prevent overfitting by randomly deactivating neurons during training.
4. **Fully Connected Layers**: Combine extracted features and enable final classification.

Hyperparameters, including the number of layers, kernel size, activation functions, and dropout rates, were optimized for better performance.

---

## 5. Results <a name="results"></a>
The main findings of the code can be found at the post on Medium available here

---

## 6. File Descriptions
The repository contains the following files:

- **`CNN for Image Classification.ipynb`**: Jupyter Notebook with the code for building, training, and testing the CNN model.
- **`Read Me.txt`**: Documentation providing an overview of the project, model usage, and execution steps.
- **`training_set`**: A directory containing images for model training.
- **`test_set`**: A directory containing images used both as a validation set and a test set.
- **`single_prediction`**: A directory containing sample images for single-input predictions.

---

## 7. Licensing, Authors, Acknowledgements, etc.
This project was developed for educational purposes, utilizing TensorFlow and Keras. All dataset images are used solely for demonstrating the capabilities of CNNs in binary classification tasks.

---
