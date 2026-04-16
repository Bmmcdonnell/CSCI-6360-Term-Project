import os
import contextlib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from typing import List, Tuple, Union
from neural_network_classes import NoHiddenLayerNN, OneHiddenLayerNN, TwoHiddenLayerNN
from get_qof import get_qof
from get_cv_qof import get_cv_qof
from hyperparameter_tuning import tune_ridge_lasso_alpha, tune_box_cox_lambda, tune_nn_hyperparams
from save_plots import save_sorted_plot


def lin_reg(
    X: pd.DataFrame, 
    y: Union[pd.Series, pd.DataFrame], 
    X_test: pd.DataFrame, 
    X_train: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    y_train: Union[pd.Series, pd.DataFrame], 
    data_name: str, 
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Evaluates a standard Multiple Linear Regression model across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, and CV statistics.
    """
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    # Coerce to ndarray so get_qof receives Union[np.ndarray, pd.Series] — not DataFrame
    y_np = np.asarray(y, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)
    y_train_np_arr = np.asarray(y_train, dtype=float)

    model_is = sm.OLS(y, X).fit()
    yp_is = model_is.predict(X)
    k = X.shape[1] 
    
    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "Linear Regression", "Reg", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(y_train, X_train).fit()
    yp_oos = model_oos.predict(X_test)
    k = X_train.shape[1]
    
    # Pass None for model to enforce manual metric calculations on the test set
    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "Linear Regression", "Reg", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="linreg")

    return (qof_is, qof_oos, cv_stats)


def ridge_reg(
    X: pd.DataFrame, 
    y: Union[pd.Series, pd.DataFrame], 
    X_test: pd.DataFrame, 
    X_train: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    y_train: Union[pd.Series, pd.DataFrame], 
    data_name: str, 
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]], float]:
    """
    Evaluates a Ridge Regression model across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.
    
    Automatically tunes the alpha hyperparameter and applies StandardScaler 
    internally to properly penalize coefficients without data leakage.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, 
        CV statistics, and the best tuned alpha value.
    """
    (best_alpha, best_r_sq) = tune_ridge_lasso_alpha(X, y, method='ridge')
    
    y_np = np.asarray(y, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    X_scaler_is = StandardScaler()
    y_scaler_is = StandardScaler()
    
    # Regularization requires standardized data
    X_scaled_is = X_scaler_is.fit_transform(X)
    y_scaled_is = y_scaler_is.fit_transform(y_np.reshape(-1, 1)).flatten()

    model_is = sm.OLS(y_scaled_is, X_scaled_is).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_is_scaled: np.ndarray = np.asarray(model_is.predict(X_scaled_is), dtype=float)
    
    # Inverse transform to original scale to evaluate meaningful residuals
    yp_is = y_scaler_is.inverse_transform(yp_is_scaled.reshape(-1, 1)).flatten()
    k = X_scaled_is.shape[1]
    
    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "Ridge Regression", "Ridge", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    X_scaler_oos = StandardScaler()
    y_scaler_oos = StandardScaler()
    
    X_train_scaled = X_scaler_oos.fit_transform(X_train)
    # Transform test set using the training set parameters to avoid data leakage
    X_test_scaled = X_scaler_oos.transform(X_test)
    y_train_scaled = y_scaler_oos.fit_transform(y_train_np.reshape(-1, 1)).flatten()

    model_oos = sm.OLS(y_train_scaled, X_train_scaled).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_oos_scaled: np.ndarray = np.asarray(model_oos.predict(X_test_scaled), dtype=float)
    
    yp_oos = y_scaler_oos.inverse_transform(yp_oos_scaled.reshape(-1, 1)).flatten()
    k = X_train_scaled.shape[1]
    
    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "Ridge Regression", "Ridge", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="ridge", alpha=best_alpha)

    return (qof_is, qof_oos, cv_stats, best_alpha)


def lasso_reg(
    X: pd.DataFrame, 
    y: Union[pd.Series, pd.DataFrame], 
    X_test: pd.DataFrame, 
    X_train: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    y_train: Union[pd.Series, pd.DataFrame], 
    data_name: str, 
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]], float, List[str]]:
    """
    Evaluates a Lasso Regression model across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.
    
    Automatically tunes the alpha hyperparameter, applies StandardScaler internally, 
    and returns the surviving non-zero coefficients.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, 
        CV statistics, the best tuned alpha value, and a list of non-zero coefficients.
    """
    (best_alpha, best_r_sq) = tune_ridge_lasso_alpha(X, y, method='lasso')
    
    y_np = np.asarray(y, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    X_scaler_is = StandardScaler()
    y_scaler_is = StandardScaler()
    
    X_scaled_is = X_scaler_is.fit_transform(X)
    y_scaled_is = y_scaler_is.fit_transform(y_np.reshape(-1, 1)).flatten()

    model_is = sm.OLS(y_scaled_is, X_scaled_is).fit_regularized(alpha=best_alpha, L1_wt=1.0)
    yp_is_scaled: np.ndarray = np.asarray(model_is.predict(X_scaled_is), dtype=float)
    
    yp_is = y_scaler_is.inverse_transform(yp_is_scaled.reshape(-1, 1)).flatten()
    k = X_scaled_is.shape[1]
    
    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "Lasso", "Lasso", False)

    # Isolate features not zeroed out by L1 regularization
    coef_array = np.abs(model_is.params)
    non_zero_coeffs = list(X.columns[coef_array > 1e-6])

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    X_scaler_oos = StandardScaler()
    y_scaler_oos = StandardScaler()
    
    X_train_scaled = X_scaler_oos.fit_transform(X_train)
    X_test_scaled = X_scaler_oos.transform(X_test)
    y_train_scaled = y_scaler_oos.fit_transform(y_train_np.reshape(-1, 1)).flatten()

    model_oos = sm.OLS(y_train_scaled, X_train_scaled).fit_regularized(alpha=best_alpha, L1_wt=1.0)
    yp_oos_scaled: np.ndarray = np.asarray(model_oos.predict(X_test_scaled), dtype=float)
    
    yp_oos = y_scaler_oos.inverse_transform(yp_oos_scaled.reshape(-1, 1)).flatten()
    k = X_train_scaled.shape[1]
    
    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "Lasso", "Lasso", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="lasso", alpha=best_alpha)

    return (qof_is, qof_oos, cv_stats, best_alpha, non_zero_coeffs)


def sqrt_reg(
    X: pd.DataFrame, 
    y: Union[pd.Series, pd.DataFrame], 
    X_test: pd.DataFrame, 
    X_train: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    y_train: Union[pd.Series, pd.DataFrame], 
    data_name: str, 
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Evaluates a Transformed Regression model applying a Square Root transformation 
    to the target variable to stabilize variance across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, and CV statistics.
    """
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    # Coerce to ndarray so get_qof receives Union[np.ndarray, pd.Series] — not DataFrame
    y_np = np.asarray(y, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    model_is = sm.OLS(np.sqrt(y), X).fit()
    yp_is = (model_is.predict(X)) ** 2
    k = X.shape[1]
    
    # Pass None for model to ensure get_qof computes un-transformed residuals
    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "Sqrt Transformation", "Sqrt", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(np.sqrt(y_train), X_train).fit()
    yp_oos = (model_oos.predict(X_test)) ** 2
    k = X_train.shape[1]
    
    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "Sqrt Transformation", "Sqrt", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="sqrt")

    return (qof_is, qof_oos, cv_stats)


def log1p_reg(
    X: pd.DataFrame, 
    y: Union[pd.Series, pd.DataFrame], 
    X_test: pd.DataFrame, 
    X_train: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    y_train: Union[pd.Series, pd.DataFrame], 
    data_name: str, 
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Evaluates a Transformed Regression model applying a Log1p transformation 
    across three regimes: In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, and CV statistics.
    """
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    # Coerce to ndarray so get_qof receives Union[np.ndarray, pd.Series] — not DataFrame
    y_np = np.asarray(y, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    model_is = sm.OLS(np.log1p(y), X).fit()
    yp_is = np.expm1(model_is.predict(X))
    k = X.shape[1]
    
    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "Log1p Transformation", "Log1p", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(np.log1p(y_train), X_train).fit()
    yp_oos = np.expm1(model_oos.predict(X_test))
    k = X_train.shape[1]
    
    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "Log1p Transformation", "Log1p", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="log1p")

    return (qof_is, qof_oos, cv_stats)


def boxcox_reg(
    X: pd.DataFrame, 
    y: Union[pd.Series, pd.DataFrame], 
    X_test: pd.DataFrame, 
    X_train: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    y_train: Union[pd.Series, pd.DataFrame], 
    data_name: str, 
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]], float]:
    """
    Evaluates a Transformed Regression model utilizing an optimized Box-Cox transformation 
    across three regimes: In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, 
        CV statistics, and the best tuned lambda value.
    """
    (best_lambda, best_r_sq) = tune_box_cox_lambda(X, y)

    # Coerce to ndarray so get_qof receives Union[np.ndarray, pd.Series] — not DataFrame
    y_np = np.asarray(y, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(boxcox(y.to_numpy(), best_lambda), X).fit()
    yp_is = inv_boxcox(model_is.predict(X), best_lambda)
    k = X.shape[1]
    
    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "Box-Cox Transformation", "BoxCox", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(boxcox(y_train.to_numpy(), best_lambda), X_train).fit()
    yp_oos = inv_boxcox(model_oos.predict(X_test), best_lambda)
    k = X_train.shape[1]
    
    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "Box-Cox Transformation", "BoxCox", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="boxcox", lambda_=best_lambda)

    return (qof_is, qof_oos, cv_stats, best_lambda)


def order2_reg(
    X: pd.DataFrame, 
    y: Union[pd.Series, pd.DataFrame], 
    X_test: pd.DataFrame, 
    X_train: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    y_train: Union[pd.Series, pd.DataFrame], 
    data_name: str, 
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]], float]:
    """
    Evaluates an Order-2 (quadratic/polynomial) Ridge Regression model across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.
    
    Assumes X features passed are already the polynomial inputs. Applies 
    StandardScaler internally to properly normalize the higher-order terms.

    Args:
        X: The full input feature matrix (pre-transformed to order-2).
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, 
        CV statistics, and the best tuned alpha value.
    """
    (best_alpha, best_r_sq) = tune_ridge_lasso_alpha(X, y, method='ridge')
    
    y_np = np.asarray(y, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    X_scaler_is = StandardScaler()
    y_scaler_is = StandardScaler()
    
    X_scaled_is = X_scaler_is.fit_transform(X)
    y_scaled_is = y_scaler_is.fit_transform(y_np.reshape(-1, 1)).flatten()

    model_is = sm.OLS(y_scaled_is, X_scaled_is).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_is_scaled: np.ndarray = np.asarray(model_is.predict(X_scaled_is), dtype=float)
    
    yp_is = y_scaler_is.inverse_transform(yp_is_scaled.reshape(-1, 1)).flatten()
    k = X_scaled_is.shape[1]
    
    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "Order 2 Regression", "Order2Reg", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    X_scaler_oos = StandardScaler()
    y_scaler_oos = StandardScaler()
    
    X_train_scaled = X_scaler_oos.fit_transform(X_train)
    X_test_scaled = X_scaler_oos.transform(X_test)
    y_train_scaled = y_scaler_oos.fit_transform(y_train_np.reshape(-1, 1)).flatten()

    model_oos = sm.OLS(y_train_scaled, X_train_scaled).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_oos_scaled: np.ndarray = np.asarray(model_oos.predict(X_test_scaled), dtype=float)
    
    yp_oos = y_scaler_oos.inverse_transform(yp_oos_scaled.reshape(-1, 1)).flatten()
    k = X_train_scaled.shape[1]
    
    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "Order 2 Regression", "Order2Reg", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="ridge", alpha=best_alpha)

    return (qof_is, qof_oos, cv_stats, best_alpha)


def nn_2L(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_test: Union[pd.Series, pd.DataFrame],
    y_train: Union[pd.Series, pd.DataFrame],
    data_name: str,
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]], nn.Module, float]:
    """
    Evaluates a 2-Layer Neural Network model (no hidden layers) across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.

    Automatically tunes the output-layer activation function and learning rate via
    sequential cross-validated grid searches (hidden-size and hidden-activation stages
    are skipped for nn_2L because it has no hidden layers).

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, CV statistics,
        the best hidden activation (nn.Identity placeholder for nn_2L),
        the best output activation, and the best learning rate.
    """
    # ==========================================
    # --- Hyperparameter Tuning ---
    # ==========================================
    (_, _, best_output_activation_fn, _, _, best_lr) = tune_nn_hyperparams(X, y, method='nn_2L')

    y_np = np.asarray(y, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    X_scaler_is = StandardScaler()
    y_scaler_is = StandardScaler()

    X_scaled_is = X_scaler_is.fit_transform(X)
    y_scaled_is = y_scaler_is.fit_transform(y_np.reshape(-1, 1))

    input_features = X_scaled_is.shape[1]
    output_classes = 1 if len(y_scaled_is.shape) == 1 else y_scaled_is.shape[1]

    X_tr_tensor = torch.tensor(X_scaled_is, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_scaled_is, dtype=torch.float32)

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model_is = NoHiddenLayerNN(input_size=input_features, output_size=output_classes, activation_fn=best_output_activation_fn)
        model_is.lr = best_lr
        model_is.fit(X_tr_tensor, y_tr_tensor)
        yp_is_tensor, _ = model_is.predict(X_tr_tensor, y_tr_tensor)

    yp_is_scaled = yp_is_tensor.detach().cpu().numpy()
    yp_is = y_scaler_is.inverse_transform(yp_is_scaled).flatten()
    k = X_scaled_is.shape[1]

    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "2L Neural Network", "NN_2L", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    X_scaler_oos = StandardScaler()
    y_scaler_oos = StandardScaler()

    X_train_scaled = X_scaler_oos.fit_transform(X_train)
    X_test_scaled = X_scaler_oos.transform(X_test)
    y_train_scaled = y_scaler_oos.fit_transform(y_train_np.reshape(-1, 1))
    y_test_scaled = y_scaler_oos.transform(y_test_np.reshape(-1, 1))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model_oos = NoHiddenLayerNN(input_size=input_features, output_size=output_classes, activation_fn=best_output_activation_fn)
        model_oos.lr = best_lr
        model_oos.fit(X_train_tensor, y_train_tensor)
        yp_oos_tensor, _ = model_oos.predict(X_test_tensor, y_test_tensor)

    yp_oos_scaled = yp_oos_tensor.detach().cpu().numpy()
    yp_oos = y_scaler_oos.inverse_transform(yp_oos_scaled).flatten()
    k = X_train_scaled.shape[1]

    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "2L Neural Network", "NN_2L", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="nn_2L", output_activation_fn=best_output_activation_fn, lr=best_lr)

    return (qof_is, qof_oos, cv_stats, best_output_activation_fn, best_lr)


def nn_3L(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_test: Union[pd.Series, pd.DataFrame],
    y_train: Union[pd.Series, pd.DataFrame],
    data_name: str,
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]], nn.Module, nn.Module, int, float]:
    """
    Evaluates a 3-Layer Neural Network model (1 hidden layer) across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.

    Automatically tunes the hidden-layer activation function, output-layer activation
    function, hidden layer size, and learning rate via sequential cross-validated grid
    searches before fitting.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, CV statistics,
        the best hidden activation, the best output activation, the best hidden layer size,
        and the best learning rate.
    """
    # ==========================================
    # --- Hyperparameter Tuning ---
    # ==========================================
    (best_activation_fn, _, best_output_activation_fn, best_hidden_1, _, best_lr) = tune_nn_hyperparams(X, y, method='nn_3L')

    y_np = np.asarray(y, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    X_scaler_is = StandardScaler()
    y_scaler_is = StandardScaler()

    X_scaled_is = X_scaler_is.fit_transform(X)
    y_scaled_is = y_scaler_is.fit_transform(y_np.reshape(-1, 1))

    input_features = X_scaled_is.shape[1]
    output_classes = 1 if len(y_scaled_is.shape) == 1 else y_scaled_is.shape[1]

    X_tr_tensor = torch.tensor(X_scaled_is, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_scaled_is, dtype=torch.float32)

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model_is = OneHiddenLayerNN(
            input_size=input_features,
            hidden_size=best_hidden_1,
            output_size=output_classes,
            hidden_activation_fn=best_activation_fn,
            output_activation_fn=best_output_activation_fn
        )
        model_is.lr = best_lr
        model_is.fit(X_tr_tensor, y_tr_tensor)
        yp_is_tensor, _ = model_is.predict(X_tr_tensor, y_tr_tensor)

    yp_is_scaled = yp_is_tensor.detach().cpu().numpy()
    yp_is = y_scaler_is.inverse_transform(yp_is_scaled).flatten()
    k = X_scaled_is.shape[1]

    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "3L Neural Network", "NN_3L", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    X_scaler_oos = StandardScaler()
    y_scaler_oos = StandardScaler()

    X_train_scaled = X_scaler_oos.fit_transform(X_train)
    X_test_scaled = X_scaler_oos.transform(X_test)
    y_train_scaled = y_scaler_oos.fit_transform(y_train_np.reshape(-1, 1))
    y_test_scaled = y_scaler_oos.transform(y_test_np.reshape(-1, 1))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model_oos = OneHiddenLayerNN(
            input_size=input_features,
            hidden_size=best_hidden_1,
            output_size=output_classes,
            hidden_activation_fn=best_activation_fn,
            output_activation_fn=best_output_activation_fn
        )
        model_oos.lr = best_lr
        model_oos.fit(X_train_tensor, y_train_tensor)
        yp_oos_tensor, _ = model_oos.predict(X_test_tensor, y_test_tensor)

    yp_oos_scaled = yp_oos_tensor.detach().cpu().numpy()
    yp_oos = y_scaler_oos.inverse_transform(yp_oos_scaled).flatten()
    k = X_train_scaled.shape[1]

    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "3L Neural Network", "NN_3L", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(
        X, y, method="nn_3L",
        nn_hidden_1=best_hidden_1,
        activation_fn=best_activation_fn,
        output_activation_fn=best_output_activation_fn,
        lr=best_lr
    )

    return (qof_is, qof_oos, cv_stats, best_activation_fn, best_output_activation_fn, best_hidden_1, best_lr)


def nn_4L(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_test: Union[pd.Series, pd.DataFrame],
    y_train: Union[pd.Series, pd.DataFrame],
    data_name: str,
    folder_name: str
) -> Tuple[List[float], List[float], List[List[float]], nn.Module, nn.Module, nn.Module, int, int, float]:
    """
    Evaluates a 4-Layer Neural Network model (2 hidden layers) across three regimes:
    In-Sample, 80-20 Split (Out-of-Sample), and 5-Fold Cross-Validation.

    Automatically tunes the hidden-layer activation functions, output-layer activation
    function, hidden layer sizes, and learning rate via sequential cross-validated grid
    searches before fitting.

    Args:
        X: The full input feature matrix.
        y: The full target response vector.
        X_test: The testing set feature matrix.
        X_train: The training set feature matrix.
        y_test: The testing set target vector.
        y_train: The training set target vector.
        data_name (str): Identifier for the dataset (used for saving plots).
        folder_name (str): Directory where plots should be saved.

    Returns:
        Tuple containing In-Sample QoF metrics, Out-of-Sample QoF metrics, CV statistics,
        the best activation for hidden layer 1, the best activation for hidden layer 2,
        the best output activation, the best hidden size for layer 1,
        the best hidden size for layer 2, and the best learning rate.
    """
    # ==========================================
    # --- Hyperparameter Tuning ---
    # ==========================================
    (best_activation_fn, best_activation_fn_2, best_output_activation_fn, best_hidden_1, best_hidden_2, best_lr) = tune_nn_hyperparams(X, y, method='nn_4L')

    y_np = np.asarray(y, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    X_scaler_is = StandardScaler()
    y_scaler_is = StandardScaler()

    X_scaled_is = X_scaler_is.fit_transform(X)
    y_scaled_is = y_scaler_is.fit_transform(y_np.reshape(-1, 1))

    input_features = X_scaled_is.shape[1]
    output_classes = 1 if len(y_scaled_is.shape) == 1 else y_scaled_is.shape[1]

    X_tr_tensor = torch.tensor(X_scaled_is, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_scaled_is, dtype=torch.float32)

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model_is = TwoHiddenLayerNN(
            input_size=input_features,
            hidden_size_1=best_hidden_1,
            hidden_size_2=best_hidden_2,
            output_size=output_classes,
            hidden_activation_fn_1=best_activation_fn,
            hidden_activation_fn_2=best_activation_fn_2,
            output_activation_fn=best_output_activation_fn
        )
        model_is.lr = best_lr
        model_is.fit(X_tr_tensor, y_tr_tensor)
        yp_is_tensor, _ = model_is.predict(X_tr_tensor, y_tr_tensor)

    yp_is_scaled = yp_is_tensor.detach().cpu().numpy()
    yp_is = y_scaler_is.inverse_transform(yp_is_scaled).flatten()
    k = X_scaled_is.shape[1]

    qof_is = get_qof(y_np, yp_is, k)
    save_sorted_plot(y_np, yp_is, data_name, folder_name, "4L Neural Network", "NN_4L", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    X_scaler_oos = StandardScaler()
    y_scaler_oos = StandardScaler()

    X_train_scaled = X_scaler_oos.fit_transform(X_train)
    X_test_scaled = X_scaler_oos.transform(X_test)
    y_train_scaled = y_scaler_oos.fit_transform(y_train_np.reshape(-1, 1))
    y_test_scaled = y_scaler_oos.transform(y_test_np.reshape(-1, 1))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model_oos = TwoHiddenLayerNN(
            input_size=input_features,
            hidden_size_1=best_hidden_1,
            hidden_size_2=best_hidden_2,
            output_size=output_classes,
            hidden_activation_fn_1=best_activation_fn,
            hidden_activation_fn_2=best_activation_fn_2,
            output_activation_fn=best_output_activation_fn
        )
        model_oos.lr = best_lr
        model_oos.fit(X_train_tensor, y_train_tensor)
        yp_oos_tensor, _ = model_oos.predict(X_test_tensor, y_test_tensor)

    yp_oos_scaled = yp_oos_tensor.detach().cpu().numpy()
    yp_oos = y_scaler_oos.inverse_transform(yp_oos_scaled).flatten()
    k = X_train_scaled.shape[1]

    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, data_name, folder_name, "4L Neural Network", "NN_4L", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(
        X, y, method="nn_4L",
        nn_hidden_1=best_hidden_1,
        nn_hidden_2=best_hidden_2,
        activation_fn=best_activation_fn,
        activation_fn_2=best_activation_fn_2,
        output_activation_fn=best_output_activation_fn,
        lr=best_lr
    )

    return (qof_is, qof_oos, cv_stats, best_activation_fn, best_activation_fn_2, best_output_activation_fn, best_hidden_1, best_hidden_2, best_lr)