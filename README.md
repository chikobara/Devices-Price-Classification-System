# Devices Price Classification System

This project implements a Devices Price Classification System using machine learning techniques such as Logistic Regression and SVM to predict the price range of devices based on their specifications. The project focuses solely on data exploration, preprocessing, and model training.

# Table of Contents

- [Devices Price Classification System](#devices-price-classification-system)
- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Testing and Results](#testing-and-results)

# Project Overview

The Devices Price Classification System aims to predict the price category of a device using various features such as battery power, RAM, screen size, and camera specifications. The system predicts one of four price ranges:

- 0: Low cost
- 1: Medium cost
- 2: High cost
- 3: Very high cost

This repository focuses on data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning model training using Logistic Regression and Support Vector Machines (SVM).

# Project Structure

```
├── dataset
│   ├── cleaned_train_data.csv          # Cleaned training data
│   ├── test_data.csv                   # Test data for predictions
│   └── train_data.csv                  # Raw training data
├── EDA_Figures                         # Figures generated during EDA
│   ├── battery_power_distribution.png  # Distribution of battery power
│   ├── blue_distribution.png           # Bluetooth feature distribution
│   ├── ...                             # Other EDA plots
│   └── wifi_distribution.png           # WiFi feature distribution
├── ml_modeling
│   ├── cleaning_and_eda.ipynb          # Notebook for data cleaning and EDA
│   ├── model.ipynb                     # Notebook for model training (Logistic Regression, SVM)
│   └── svc_model.pkl                   # Saved SVM model
├── .gitignore                          # Files and directories to ignore in version control
├── LICENSE                             # License for the project
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

# Setup Instructions

1. **Install Dependencies**: Make sure you have Python installed, then install the required dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```

2. **Data Cleaning and EDA**: Run the `cleaning_and_eda.ipynb` notebook to perform data cleaning and exploratory data analysis:

   ```bash
   jupyter notebook ml_modeling/cleaning_and_eda.ipynb
   ```

   This notebook will generate visualizations saved in the `EDA_Figures` directory.

3. **Model Training**: After cleaning the data, run the `model.ipynb` notebook to train the machine learning models:

   ```bash
   jupyter notebook ml_modeling/model.ipynb
   ```

   This will train Logistic Regression and SVM models and save the best model (`svc_model.pkl`).

# Feature Engineering

Several feature engineering techniques were applied to enhance the dataset:

- **Screen Area**: Calculated as the product of `sc_h` (screen height) and `sc_w` (screen width).
- **Pixel Density**: Calculated as the ratio of pixel area (`px_height * px_width`) to screen area.
- **Camera Megapixels Ratio**: Ratio of the front camera (`fc`) to primary camera (`pc`).
- **Log Transformations**: Applied to RAM, `battery_power`, and `internal_memory` to reduce skewness.
- **Binning**: Battery power, RAM, and internal memory were binned using quantiles.

# Model Training

Two machine learning models were used for the classification task:

1. **Logistic Regression**: This linear model is trained to classify devices into price ranges based on features.

   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train_scaled, y_train)
   ```

2. **Support Vector Machine (SVM)**: A non-linear model using the RBF kernel for more complex decision boundaries.

   ```python
   from sklearn.svm import SVC
   model = SVC(kernel='rbf', C=1, gamma='scale')
   model.fit(X_train_scaled, y_train)
   ```

The SVM model performed slightly better and was saved as `svc_model.pkl`.

# Testing and Results

Testing was conducted using the cleaned test dataset (`test_data.csv`). Key results from the model evaluation include:

- **Accuracy**:
  - Logistic Regression: 85%
  - SVM with RBF Kernel: 87%
- **Confusion Matrix**: Generated for both models to evaluate the classification performance. You can refer to the figures and charts generated during the testing phase in the `model.ipynb` notebook.

You can visualize the results of feature importance, confusion matrices, and accuracy plots by referring to the `model.ipynb` notebook.