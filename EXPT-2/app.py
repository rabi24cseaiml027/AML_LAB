from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    y_name = request.form.get('y_name', 'y').strip()
    y_input = request.form.get('y_values', '')
    
    try:
        y_vals = np.array([float(val) for val in y_input.split(",") if val.strip()])
        num_data_points = len(y_vals)
    except ValueError:
        return "Error: Invalid numerical values for y.", 400

    x_data = {}
    x_names = []
    x_values_raw = []
    
    idx = 0
    while f'x_name_{idx}' in request.form:
        feat_name = request.form.get(f'x_name_{idx}').strip()
        feat_input = request.form.get(f'x_values_{idx}', '')
        
        if not feat_input.strip():
            idx += 1
            continue
            
        try:
            feat_vals = np.array([float(val) for val in feat_input.split(",") if val.strip()])
            if len(feat_vals) != num_data_points:
                return f"Error: '{feat_name}' has {len(feat_vals)} data points, but '{y_name}' has {num_data_points}.", 400
            
            x_names.append(feat_name)
            x_values_raw.append(feat_input)
            x_data[f'x{idx+1}'] = feat_vals
        except ValueError:
            return f"Error: Invalid numerical values inside feature '{feat_name}'.", 400
        idx += 1

    # Model Calculation
    data = pd.DataFrame(x_data)
    model = LinearRegression()
    model.fit(data, y_vals)
    predictions = model.predict(data)

    mse = mean_squared_error(y_vals, predictions)
    mae = mean_absolute_error(y_vals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_vals, predictions)

    intercept = model.intercept_
    coefficients = model.coef_
    equation_terms = [f"({coef:.4f} * x{i+1})" for i, coef in enumerate(coefficients)]
    equation_str = f"y = {intercept:.4f} + " + " + ".join(equation_terms)
    slopes = {x_names[i]: f"{coef:.4f}" for i, coef in enumerate(coefficients)}

    plt.scatter(y_vals, predictions, color="#10b981", edgecolor="#065f46", linewidth=1, s=50, alpha=0.85)
    min_val = min(float(np.min(y_vals)), float(np.min(predictions)))
    max_val = max(float(np.max(y_vals)), float(np.max(predictions)))
    plt.plot([min_val, max_val], [min_val, max_val], color="#64748b", linestyle="-.", linewidth=1.5)
    plt.xlabel(f"Actual {y_name}")
    plt.ylabel(f"Predicted {y_name}")
    plt.title("Model Prediction Accuracy Evaluation")
    plt.grid(True)
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot1_base64 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close()

    metrics_names = ["R2 Score", "RMSE", "MAE", "MSE"]
    metrics_vals = [r2, rmse, mae, mse]
    colors = ["#4f46e5", "#3b82f6", "#60a5fa", "#93c5fd"]
    bars = plt.barh(metrics_names, metrics_vals, color=colors)
    for bar in bars:
        width = bar.get_width()
        plt.annotate(f" {width:.4f}", xy=(width, bar.get_y() + bar.get_height() / 2), ha='left', va='center', fontweight='bold')    
    plt.xlabel("Computed Values")
    plt.ylabel("Metric Value")
    plt.title("Linear Regression Performance Metrics")
    plt.grid(True)

    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    buf2.seek(0)
    plot2_base64 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close()

    x_raw_str = ";".join(x_values_raw)

    return render_template('result.html', 
                           equation=equation_str, 
                           intercept=f"{intercept:.4f}", 
                           slopes=slopes, 
                           y_name=y_name,
                           y_input=y_input,
                           x_names=x_names,
                           x_raw_str=x_raw_str,
                           plot1=plot1_base64, 
                           plot2=plot2_base64)

@app.route('/predict', methods=['POST'])
def predict():
    y_input = request.form['y_input']
    x_raw_str = request.form['x_raw_str']
    y_name = request.form['y_name']

    y_vals = np.array([float(val) for val in y_input.split(",") if val.strip()])
    feature_strings = x_raw_str.split(";")
    
    x_data = {}
    for i, f_str in enumerate(feature_strings):
        x_data[f'x{i+1}'] = [float(val) for val in f_str.split(",") if val.strip()]
        
    model = LinearRegression()
    model.fit(pd.DataFrame(x_data), y_vals)

    try:
        sample_dict = {}
        for idx in range(len(feature_strings)):
            val = float(request.form[f'pred_x_{idx}'])
            sample_dict[f'x{idx+1}'] = val

        predicted_val = model.predict(pd.DataFrame([sample_dict]))[0]
        return f"<h2>Predicted {y_name} (y): {predicted_val:.4f}</h2><br><a href='javascript:history.back()'>Go Back</a>"
    except Exception:
        return "Invalid input numbers entered for prediction.", 400

if __name__ == '__main__':
    app.run(debug=True)