# Image Classification: Classic ML vs. Multi-Layer Perceptron (MLP)

## Project Overview and Purpose
This project explores the task of image classification using the CIFAR-10 dataset, which contains 60,000 color images across 10 distinct classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The primary objective is to compare the performance of "Classic" Machine Learning algorithms against a "Modern" Deep Learning Multi-Layer Perceptron (MLP).

## Key Technologies and Libraries
- **Deep Learning**: `TensorFlow`, `Keras`
- **Machine Learning**: `scikit-learn`
- **Data Handling**: `NumPy`, `Pandas`
- **Visualization**: `Matplotlib`, `Plotly`

## Methodology and Models
### Dataset Preparation
- **Loading**: Data is loaded directly via `tensorflow.keras.datasets.cifar10`.
- **Preprocessing**: Images are normalized to a range of 0 to 1, and the 32x32x3 image matrices are flattened into 1D vectors for compatibility with the classic ML models.

### Evaluated Models
1. **K-Nearest Neighbors (KNN)**: A baseline instance-based learning algorithm.
2. **Random Forest Classifier**: An ensemble learning method using multiple decision trees.
3. **Multi-Layer Perceptron (MLP)**: A fully connected neural network built with Keras, featuring multiple hidden layers and Dropout for regularization.


## Results and Insights
- **Comparative Analysis**: The project includes detailed classification reports and confusion matrices for each model.
- **Performance**: The MLP typically outperforms classic ML models on this dataset due to its ability to learn complex non-linear features from raw pixel data.
- **Visualizations**: Interactive confusion matrices are generated using `Plotly` to identify which classes (e.g., cat vs. dog) are most frequently confused by the models.

## How to Run
1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
