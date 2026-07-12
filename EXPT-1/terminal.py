import numpy as np
import matplotlib.pyplot as plt


def calculate_linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x) ** 2)

    m = numerator / denominator
    c = mean_y - m * mean_x

    return m, c


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_total == 0:
        r_squared = 1.0 if ss_residual == 0 else float("-inf")
    else:
        r_squared = 1 - (ss_residual / ss_total)

    return mae, mse, rmse, r_squared


def show_graph(x, y, m, c, mae, mse, rmse, r2):
    x = np.asarray(x)
    y = np.asarray(y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.scatter(x, y, color="blue", label="Actual Data")

    x_line = np.linspace(min(x), max(x), 100)
    y_line = m * x_line + c

    sign = "+" if c >= 0 else "-"
    ax1.plot(
        x_line,
        y_line,
        color="red",
        label=f"y = {m:.4f}x {sign} {abs(c):.4f}",
    )

    ax1.set_title("Linear Regression")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True)
    ax1.legend()

    labels = ["MSE", "MAE", "RMSE", "R²"]
    values = [mse, mae, rmse, r2]

    ax2.plot(labels, values, marker="o")

    for i, value in enumerate(values):
        ax2.text(i, value, f"{value:.4f}")

    ax2.set_title("Evaluation Metrics")
    ax2.set_xlabel("Metrics")
    ax2.set_ylabel("Value")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


x = np.array(list(map(float, input("Enter X values (comma-separated): ").split(","))))
y = np.array(list(map(float, input("Enter Y values (comma-separated): ").split(","))))

m, c = calculate_linear_regression(x, y)

y_pred = m * x + c

mae, mse, rmse, r2 = calculate_metrics(y, y_pred)

print("\n------ Linear Regression ------")

if c >= 0:
    print(f"Regression Equation : y = {m:.4f}x + {c:.4f}")
else:
    print(f"Regression Equation : y = {m:.4f}x - {abs(c):.4f}")

print("\nPredicted Y Values:")
print(np.round(y_pred, 4))

print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

show_graph(x, y, m, c, mae, mse, rmse, r2)