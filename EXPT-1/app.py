from flask import Flask, request, jsonify, send_from_directory
import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='.', static_url_path='')

def calculate_linear_regression(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)
    if n == 0:
        raise ValueError("Data lists cannot be empty.")
    if len(x_arr) != len(y_arr):
        raise ValueError("X and Y lists must have the same length.")
    
    mean_x = np.mean(x_arr)
    mean_y = np.mean(y_arr)
    numerator = np.sum((x_arr - mean_x) * (y_arr - mean_y))
    denominator = np.sum((x_arr - mean_x) ** 2)
    if denominator == 0:
        raise ValueError("Cannot calculate slope (all X values are identical, resulting in a vertical line).")
    m = numerator / denominator
    c = mean_y - m * mean_x
    return float(m), float(c)

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        raise ValueError("Input lists cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lists must have the same length.")
    
    mean_y_true = np.mean(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - mean_y_true) ** 2)
    
    if ss_total == 0:
        r_squared = 1.0 if ss_residual == 0 else float('-inf')
    else:
        r_squared = 1.0 - (ss_residual / ss_total)
    
    return float(mae), float(mse), float(rmse), float(r_squared)

def generate_metrics_graph(x, y, m, c, mae, mse, rmse, r_squared):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ax1.scatter(x_arr, y_arr, color='#8b5cf6', edgecolor='#6d28d9', s=55, alpha=0.75, label='Actual Data Points')
    
    min_x = x_arr.min()
    max_x = x_arr.max()
    padding_x = (max_x - min_x) * 0.1 if max_x != min_x else 1.0
    x_line = np.linspace(min_x - padding_x, max_x + padding_x, 100)
    y_line = m * x_line + c
    
    c_sign = '+' if c >= 0 else '-'
    line_label = f'Regression Line\n$y = {m:.4f}x {c_sign} {abs(c):.4f}$'
    ax1.plot(x_line, y_line, color='#3b82f6', linestyle='-', linewidth=2.2, label=line_label)
    
    ax1.set_xlabel('X Values', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Y Values', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('Regression Fit Line', fontsize=13, fontweight='bold', pad=12, color='#1f2937')
    ax1.legend(loc='best', frameon=True, facecolor='#ffffff', edgecolor='#e5e7eb')
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    metrics_labels = ['MSE', 'MAE', 'RMSE', 'R2']
    metrics_values = [mse, mae, rmse, r_squared]
    
    ax2.plot(metrics_labels, metrics_values, 'o', color='#1f77b4', markersize=7)
    
    ax2.set_xlabel('Evaluation Metrics', fontsize=11, fontweight='normal')
    ax2.set_ylabel('Value', fontsize=11, fontweight='normal')
    ax2.set_title('Model Metrics Comparison', fontsize=13, fontweight='normal', pad=10)
    ax2.grid(True)
    ax2.set_xlim(-0.4, 3.6)
    
    for i, val in enumerate(metrics_values):
        if metrics_labels[i] == 'R2' and (r_squared == float('-inf') or np.isneginf(r_squared)):
            text_str = '-inf'
        else:
            text_str = f"{val:.4f}"
            while text_str.endswith('0') and len(text_str.split('.')[1]) > 2:
                text_str = text_str[:-1]
        
        ax2.text(i + 0.05, val, text_str, fontsize=10, va='center', ha='left')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64
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
        x_arr = np.asarray(x, dtype=float)
        y_pred_arr = m * x_arr + c
        y_pred = y_pred_arr.tolist()
        mae, mse, rmse, r_squared = calculate_metrics(y, y_pred)
        graph_base64 = generate_metrics_graph(x, y, m, c, mae, mse, rmse, r_squared)
        return jsonify({
            'status': 'success',
            'm': m,
            'c': c,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r_squared': r_squared,
            'y_pred': y_pred,
            'metrics_graph': graph_base64
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)