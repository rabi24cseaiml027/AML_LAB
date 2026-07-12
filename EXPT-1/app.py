from flask import Flask, request, jsonify, send_from_directory
import math
import os
app = Flask(__name__, static_folder='.', static_url_path='')
def calculate_linear_regression(x, y):
    n = len(x)
    if n == 0:
        raise ValueError("Data lists cannot be empty.")
    if len(x) != len(y):
        raise ValueError("X and Y lists must have the same length.")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        raise ValueError("Cannot calculate slope (all X values are identical, resulting in a vertical line).")
    m = numerator / denominator
    c = mean_y - m * mean_x
    return m, c
def calculate_metrics(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        raise ValueError("Input lists cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lists must have the same length.")
    mean_y_true = sum(y_true) / n
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    ss_residual = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_total = sum((t - mean_y_true) ** 2 for t in y_true)
    if ss_total == 0:
        r_squared = 1.0 if ss_residual == 0 else float('-inf')
    else:
        r_squared = 1.0 - (ss_residual / ss_total)
    return mae, mse, rmse, r_squared
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')
@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No inputs provided.'}), 400
        x = data.get('x', [])
        y = data.get('y', [])
        m, c = calculate_linear_regression(x, y)
        y_pred = [m * xi + c for xi in x]
        mae, mse, rmse, r_squared = calculate_metrics(y, y_pred)
        return jsonify({
            'status': 'success',
            'm': m,
            'c': c,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r_squared': r_squared,
            'y_pred': y_pred
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)