import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the saved model and preprocessing assets
ASSETS_PATH = "loan_model_assets.pkl"
if os.path.exists(ASSETS_PATH):
    with open(ASSETS_PATH, "rb") as f:
        assets = pickle.load(f)
    print("Model assets loaded successfully.")
else:
    raise FileNotFoundError(f"Required model assets file '{ASSETS_PATH}' not found. Run train_save_model.py first.")

@app.route("/")
def index():
    # Pass metrics to the UI so we can display them dynamically
    return render_template("index.html", metrics=assets["model_metrics"])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form request
        data = request.form.to_dict()
        
        # Extract selected model
        model_name = data.pop("model_name", "Naive Bayes")
        if model_name not in assets["models"]:
            return jsonify({"error": f"Invalid model name '{model_name}'"}), 400
        
        # Convert numeric strings to actual float values
        numeric_fields = assets["numerical_columns"]
        
        # Add Applicant_ID default if not provided
        if "Applicant_ID" not in data or not data["Applicant_ID"]:
            data["Applicant_ID"] = 500.0
            
        for col in numeric_fields:
            if col in data:
                data[col] = float(data[col])
        
        # Create input DataFrame (1 row)
        df_in = pd.DataFrame([data])
        
        # Reorder df_in to contain the expected columns in correct sequence before encoding
        all_expected_cols = assets["numerical_columns"] + ["Education_Level"] + assets["categorical_columns"]
        df_in = df_in[all_expected_cols]

        # 1. Impute missing values using the trained imputers
        df_in[assets["numerical_columns"]] = assets["num_imputer"].transform(df_in[assets["numerical_columns"]])
        df_in[assets["categorical_columns"]] = assets["cat_imputer"].transform(df_in[assets["categorical_columns"]])

        # 2. Label Encode Education_Level
        # Handle cases where value is unseen by returning default mapping
        try:
            df_in["Education_Level"] = assets["le_edu"].transform(df_in["Education_Level"])
        except ValueError:
            df_in["Education_Level"] = 0 # Default to Graduate

        # 3. One-hot encode categoricals
        encoded = assets["ohe"].transform(df_in[assets["categorical_columns"]])
        encodeddf = pd.DataFrame(encoded, columns=assets["ohe"].get_feature_names_out(assets["categorical_columns"]), index=df_in.index)
        
        # Concatenate encoded features and drop the original text columns
        df_in = pd.concat([df_in.drop(columns=assets["categorical_columns"]), encodeddf], axis=1)
        
        # Reorder columns to match the trained features exactly
        df_in = df_in[assets["feature_columns"]]
        
        # 4. Scale features
        df_in_scaled = assets["scaler"].transform(df_in)
        
        # 5. Predict using selected model
        model = assets["models"][model_name]
        pred = model.predict(df_in_scaled)[0]
        
        # Calculate prediction probability if available
        probability = None
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(df_in_scaled)[0]
            probability = float(prob_arr[pred])
            
        # Decode target label
        result_label = assets["le_approved"].inverse_transform([pred])[0]
        
        # Format metrics for response
        metrics = assets["model_metrics"][model_name]
        
        return jsonify({
            "status": "success",
            "prediction": result_label,
            "probability": round(probability * 100, 2) if probability is not None else None,
            "model_used": model_name,
            "metrics": {
                "accuracy": round(metrics["accuracy"] * 100, 2),
                "precision": round(metrics["precision"] * 100, 2),
                "recall": round(metrics["recall"] * 100, 2),
                "f1": round(metrics["f1"] * 100, 2)
            }
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
