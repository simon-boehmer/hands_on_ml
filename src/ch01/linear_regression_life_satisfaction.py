import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import plot_regression_fit


def main() -> None:
    """Train a simple linear regression model and plot results."""

    # Resolve paths (repo root → data/sample)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(repo_root, "data", "sample")

    # Load dataset
    csv_path = os.path.join(data_dir, "gdp_life_satisfaction.csv")
    df = pd.read_csv(csv_path)

    # Inspect data
    print(df.head())
    print("-" * 40)

    # Features (X) and target (y)
    X = df["GDP_per_capita_USD"].to_numpy().reshape(-1, 1)
    y = df["Life_satisfaction"].to_numpy()

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Report model
    print(f"y = {model.coef_[0]:.6f} * x + {model.intercept_:.6f}")
    print(f"R^2: {model.score(X, y):.4f}")

    # Predictions for smooth regression line
    X_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred = model.predict(X_line)

    # Plot data + regression line
    plot_regression_fit(
        X,
        y,
        X_line,
        y_pred,
        x_label="GDP per capita (USD)",
        y_label="Life satisfaction (0–10)",
        title="GDP vs Life Satisfaction",
        y_lim=(0, 10),
        format_x_thousands=True,
    )


if __name__ == "__main__":
    main()
