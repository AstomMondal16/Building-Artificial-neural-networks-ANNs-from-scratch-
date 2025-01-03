# Building-Artificial-neural-networks-ANNs-from-scratch-
This repository contains a custom implementation of a neural network to predict kidney stone cases using a dataset. The neural network is built from scratch in Python without the use of any deep learning frameworks. It supports adding multiple layers, forward propagation, backpropagation, and weight updates.

---

## Features

- **Custom Neural Network Implementation**:
  - Supports multiple layers with configurable sizes.
  - Implements sigmoid activation function and its derivative.
  - Includes methods for forward propagation, backpropagation, and gradient descent.

- **Dataset**:
  - Uses the `kidney-stone-dataset.csv` for predictions.
  - Automatically handles missing data and splits the dataset into training and testing sets.

- **Evaluation**:
  - Calculates accuracy on the test set.
  - Displays training loss during the training process.

---

## Prerequisites

1. **Hardware**:
   - A system capable of running Python (e.g., a PC or server).

2. **Software**:
   - Python 3.6 or later.
   - Required libraries:
     - `numpy`
     - `pandas`
     - `scikit-learn`

---

## Installation

1. **Clone this repository**:
```bash   
git clone https://github.com/your_username/kidney-stone-prediction.git
cd kidney-stone-prediction
```
Install dependencies: Use pip to install the required libraries:

    pip install numpy pandas scikit-learn
Dataset: Ensure the kidney-stone-dataset.csv file is in the same directory as the code.

## Dataset

The `kidney-stone-dataset.csv` file is expected to have the following structure:

- **Feature Columns**: Contains numerical data representing features relevant to kidney stone prediction.
- **Target Column**: A binary column (`1` for positive cases, `0` for negative cases).

### Notes:
- Ensure the dataset does not contain any missing or invalid values.
- If the dataset contains additional columns (e.g., an index column), they will be ignored during processing.

   
