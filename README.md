# Devices Price Classification System

This project implements a Devices Price Classification System using machine learning techniques such as Logistic Regression, Random Forest, and Support Vector Machine (SVM) to predict the price range of devices based on their specifications.

## Table of Contents
- [Devices Price Classification System](#devices-price-classification-system)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Setup Instructions](#setup-instructions)
  - [Feature Engineering](#feature-engineering)
  - [Data Split](#data-split)
  - [Model Training and Evaluation](#model-training-and-evaluation)
    - [Logistic Regression](#logistic-regression)
    - [Random Forest](#random-forest)
    - [Support Vector Machine (SVM)](#support-vector-machine-svm)

## Project Overview

The Devices Price Classification System aims to predict the price category of a device using various features. The system predicts one of four price ranges (0-3), representing low to very high cost.


## Project Structure

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


## Setup Instructions

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

## Feature Engineering

Several feature engineering techniques were applied to enhance the dataset:

* **Screen Area**: Calculated as the product of `sc_h` (screen height) and `sc_w` (screen width).
* **Pixel Density**: Calculated as the ratio of pixel area (`px_height * px_width`) to screen area.
* **Camera Megapixels Ratio**: Ratio of the front camera (`fc`) to primary camera (`pc`).
* **Log Transformations**: Applied to `RAM`, `battery_power`, and `internal_memory` to reduce skewness.
* **Binning**: Battery power, RAM, and internal memory were binned using quantiles.

## Data Split

The dataset was split into training and test sets:

- Train set: (1600, 27) features, (1600,) labels
- Test set: (400, 27) features, (400,) labels

## Model Training and Evaluation

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression().fit(X_train, y_train)

print("Accuracy of Logistic regression classifier on train set:", LR.score(X_train, y_train))
print("Accuracy of Logistic regression classifier on test set:", LR.score(X_test, y_test))
```

Results:
- Accuracy on train set: 0.88375
- Accuracy on test set: 0.86

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

model = RandomForestClassifier(random_state=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Results:

```
[[ 99   7   0   0]
 [  4  89   8   0]
 [  0   8  74   5]
 [  0   0   6 100]]

              precision    recall  f1-score   support

           0       0.96      0.93      0.95       106
           1       0.86      0.88      0.87       101
           2       0.84      0.85      0.85        87
           3       0.95      0.94      0.95       106

    accuracy                           0.91       400
   macro avg       0.90      0.90      0.90       400
weighted avg       0.91      0.91      0.91       400
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

clf = SVC(kernel="linear").fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(clf.score(X_test, y_test)))

y_pred_linear = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))
```

Results:
- Accuracy on training set: 0.97
- Accuracy on test set: 0.96

Confusion Matrix:

```
[[102   4   0   0]
 [  2  99   0   0]
 [  0   3  82   2]
 [  0   0   5 101]]
```

Classification Report:

```
              precision    recall  f1-score   support

           0       0.98      0.96      0.97       106
           1       0.93      0.98      0.96       101
           2       0.94      0.94      0.94        87
           3       0.98      0.95      0.97       106

    accuracy                           0.96       400
   macro avg       0.96      0.96      0.96       400
weighted avg       0.96      0.96      0.96       400
```

The SVM model with a linear kernel performed the best among the three models, achieving the highest accuracy on both the training and test sets.