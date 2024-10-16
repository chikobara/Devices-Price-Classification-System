# Devices Price Classification System

This project implements a Devices Price Classification System using Python for machine learning and Spring Boot for the backend API. The system predicts the price range of mobile devices based on their specifications.

## Project Structure

The project consists of two main components:

1. **Python Project**: Handles data preprocessing, exploratory data analysis (EDA), model training, and provides an API for price prediction.
2. **Spring Boot Project**: Manages device data and interacts with the Python API for price predictions.

## Python Project

### Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Model comparison and selection
- Machine learning model training for price range prediction
- Flask API for serving predictions

### Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- flask

### Setup and Running

1. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

2. Place your training and test datasets (`train_data.csv` and `test_data.csv`) in the project directory.

3. Run the Python script:

   ```
   python price_classification.py
   ```

4. The Flask API will start running on `http://localhost:5000`.

### API Endpoint

- **POST /predict**
  - Accepts device specifications as JSON
  - Returns predicted price range (0-3)

Example request:

```json
{
  "battery_power": 1000,
  "blue": true,
  "clock_speed": 1.2,
  "dual_sim": false,
  "fc": 8,
  "four_g": true,
  "int_memory": 32,
  "m_dep": 0.6,
  "mobile_wt": 150,
  "n_cores": 4,
  "pc": 12,
  "px_height": 1280,
  "px_width": 720,
  "ram": 3000,
  "sc_h": 12,
  "sc_w": 7,
  "talk_time": 20,
  "three_g": true,
  "touch_screen": true,
  "wifi": true
}
```

## Spring Boot Project

### Features

- RESTful API for managing device data
- Integration with Python API for price predictions
- Data persistence using JPA

### Requirements

- Java 11+
- Maven
- Spring Boot 2.5+

### Setup and Running

1. Navigate to the Spring Boot project directory.

2. Build the project:

   ```
   mvn clean install
   ```

3. Run the application:

   ```
   java -jar target/device-api-0.0.1-SNAPSHOT.jar
   ```

4. The Spring Boot API will start running on `http://localhost:8080`.

### API Endpoints

- **GET /api/devices**: Retrieve all devices
- **GET /api/devices/{id}**: Retrieve a specific device by ID
- **POST /api/devices**: Add a new device
- **POST /api/predict/{deviceId}**: Predict price range for a device

## Model Selection and Evaluation

We compared several machine learning models for this classification task:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM) with RBF kernel
4. XGBoost
5. Neural Network

After evaluation, XGBoost was selected as the best-performing model based on cross-validation accuracy.

## Future Improvements

- Implement hyperparameter tuning for the selected model
- Add more advanced feature engineering techniques
- Develop a user interface for easier interaction with the system
- Implement user authentication and authorization for the APIs

## Contributors

- Chikobara

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
