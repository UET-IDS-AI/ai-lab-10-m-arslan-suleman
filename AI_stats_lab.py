"""
AI_stats_lab.py

Lab: Bias-Variance Tradeoff
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ============================================================
# Question 1: Model Complexity and Generalization
# ============================================================

def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """
    Generate nonlinear data:
    y = sin(2*pi*x) + Gaussian noise
    """
    np.random.seed(random_state)

    X = np.random.rand(n_samples, 1)
    y = np.sin(2 * np.pi * X).ravel() + np.random.normal(
        0, noise, n_samples
    )

    return X, y


def create_polynomial_model(degree):
    """
    Create polynomial regression pipeline.
    """
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear", LinearRegression())
    ])

    return model


def evaluate_polynomial_degrees(X, y, degrees, test_size=0.3, random_state=0):
    """
    Train polynomial models of different degrees.
    """
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_errors = []
    dev_errors = []

    for degree in degrees:
        model = create_polynomial_model(degree)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        dev_preds = model.predict(X_dev)

        train_mse = mean_squared_error(y_train, train_preds)
        dev_mse = mean_squared_error(y_dev, dev_preds)

        train_errors.append(train_mse)
        dev_errors.append(dev_mse)

    best_degree = degrees[np.argmin(dev_errors)]

    return {
        "degrees": degrees,
        "train_errors": train_errors,
        "dev_errors": dev_errors,
        "best_degree": best_degree
    }


def diagnose_from_errors(train_error, dev_error,
                         high_error_threshold=0.15,
                         gap_threshold=0.05):
    """
    Diagnose bias/variance from train and dev errors.
    """
    gap = dev_error - train_error

    if train_error > high_error_threshold and gap <= gap_threshold:
        diagnosis = "high_bias"

    elif train_error <= high_error_threshold and gap > gap_threshold:
        diagnosis = "high_variance"

    elif train_error > high_error_threshold and gap > gap_threshold:
        diagnosis = "high_bias_and_high_variance"

    else:
        diagnosis = "good_fit"

    return {
        "train_error": train_error,
        "dev_error": dev_error,
        "generalization_gap": gap,
        "diagnosis": diagnosis
    }


# ============================================================
# Question 2: Regularization and Model Improvement
# ============================================================

def regularization_comparison(X_train, y_train, X_dev, y_dev, alphas):
    """
    Compare Ridge models for different alpha values.
    """
    train_errors = []
    dev_errors = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        dev_preds = model.predict(X_dev)

        train_mse = mean_squared_error(y_train, train_preds)
        dev_mse = mean_squared_error(y_dev, dev_preds)

        train_errors.append(train_mse)
        dev_errors.append(dev_mse)

    best_alpha = alphas[np.argmin(dev_errors)]

    return {
        "alphas": alphas,
        "train_errors": train_errors,
        "dev_errors": dev_errors,
        "best_alpha": best_alpha
    }


def recommend_action(diagnosis):
    """
    Recommend next action based on diagnosis.
    """
    mapping = {
        "high_bias": "increase_model_complexity",
        "high_variance": "add_regularization_or_more_data",
        "high_bias_and_high_variance": "increase_complexity_then_regularize",
        "good_fit": "keep_model_or_minor_tuning"
    }

    return mapping.get(diagnosis, "unknown_diagnosis")


if __name__ == "__main__":
    print("Implement all required functions.")
