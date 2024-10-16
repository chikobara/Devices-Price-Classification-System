from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json


app = Flask(__name__)


def validate_data(data):
    # Perform validation here based on your requirements
    # For simplicity, let's assume we're just checking if all required fields are present
    required_fields = [
        "battery_power",
        "blue",
        "clock_speed",
        "dual_sim",
        "fc",
        "four_g",
        "int_memory",
        "m_dep",
        "mobile_wt",
        "n_cores",
        "pc",
        "px_height",
        "px_width",
        "ram",
        "sc_h",
        "sc_w",
        "talk_time",
        "three_g",
        "touch_screen",
        "wifi",
    ]
    for field in required_fields:
        if field not in data:
            return False
    return True


def prepare_data(data):
    # cast the data type of test data to  match the training data
    test_df = pd.DataFrame.from_dict([data])
    train_df = pd.read_csv("data/train.csv")
    discrete_features = [
        "blue",
        "wifi",
        "three_g",
        "touch_screen",
        "four_g",
        "dual_sim",
    ]
    for col in train_df.drop("price_range", axis=1).columns:
        test_df[col] = test_df[col].astype(train_df[col].dtype)
    test_df[discrete_features] = test_df[discrete_features].astype("object")
    test_df = test_df.drop(["id", "price_range"], axis=1)
    return test_df


# Define a route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Assuming data is sent in JSON format
    if not validate_data(data):
        return (
            jsonify({"error": "Invalid data. Please provide all required fields."}),
            400,
        )

    # Prepare the data for prediction
    X = prepare_data(data)
    model = joblib.load("ml-service/model.joblib")
    prediction = model.predict(X).tolist()
    response = {"prediction": prediction}
    return jsonify(response), 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    app.run(debug=True)
