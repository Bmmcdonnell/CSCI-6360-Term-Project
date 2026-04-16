import os
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from typing import List, Optional, Tuple, Union
from get_qof import get_qof
from get_cv_qof import get_cv_qof
from neural_network_classes import NoHiddenLayerNN, OneHiddenLayerNN, TwoHiddenLayerNN


def get_qof2(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    method: str = 'ridge',
    alpha: float = 0.0,
    lambda_: float = 0.0,
    cv: bool = False,
    nn_hidden_1: int = 100,
    nn_hidden_2: int = 50,
    activation_fn: Optional[nn.Module] = None,
    activation_fn_2: Optional[nn.Module] = None,
    output_activation_fn: Optional[nn.Module] = None,
    lr: float = 0.01
) -> Tuple[List[float], Optional[List[List[float]]]]:
    """
    Fits a specified regression model (or neural network) on the entire dataset and 
    calculates Quality of Fit (QoF) metrics. Optionally performs k-fold cross-validation 
    to gather out-of-sample performance statistics.

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        method (str, optional): The regression method to evaluate. Valid options are:
            'linreg', 'ridge', 'lasso', 'sqrt', 'log1p', 'boxcox', 'nn_2L', 'nn_3L', 'nn_4L'. 
            Defaults to 'ridge'.
        alpha (float, optional): The regularization penalty strength used for Ridge 
            and Lasso models. Defaults to 0.0.
        lambda_ (float, optional): The transformation parameter used specifically 
            for the Box-Cox transformation. Defaults to 0.0.
        cv (bool, optional): Flag indicating whether to perform k-fold cross-validation. 
            Defaults to False.
        nn_hidden_1 (int, optional): Number of nodes in the first hidden layer 
            (used for 'nn_3L' and 'nn_4L'). Defaults to 100.
        nn_hidden_2 (int, optional): Number of nodes in the second hidden layer 
            (used for 'nn_4L'). Defaults to 50.
        activation_fn (nn.Module, optional): Hidden-layer activation function for
            layer 1 (used for 'nn_3L' and 'nn_4L'). Defaults to nn.Sigmoid().
        activation_fn_2 (nn.Module, optional): Hidden-layer activation function for
            layer 2 (used for 'nn_4L' only). Defaults to nn.Sigmoid().
        output_activation_fn (nn.Module, optional): Output-layer activation function
            applied to all NN variants. Defaults to nn.Identity().
        lr (float, optional): Learning rate for the SGD optimizer applied to all
            NN variants. Defaults to 0.01.

    Returns:
        Tuple[List[float], Optional[List[List[float]]]]: A tuple containing:
            - qof (List[float]): A list of 15 metrics evaluating the model on the full dataset.
            - cv_stats (Optional[List[List[float]]]): If `cv=True`, a matrix of 15 lists 
              containing metrics across each fold. Otherwise, returns None.
              
    Raises:
        ValueError: If an unsupported `method` string is provided.
    """
    # ==========================================
    # --- Cross-Validation ---
    # ==========================================
    if activation_fn is None:
        activation_fn = nn.Sigmoid()
    if activation_fn_2 is None:
        activation_fn_2 = nn.Sigmoid()
    if output_activation_fn is None:
        output_activation_fn = nn.Identity()

    if cv:
        # Delegate cross-validation logic to the external get_cv_qof pipeline
        cv_stats: Optional[List[List[float]]] = get_cv_qof(
            X, y,
            method=method,
            alpha=alpha,
            lambda_=lambda_,
            n_splits=5,
            nn_hidden_1=nn_hidden_1,
            nn_hidden_2=nn_hidden_2,
            activation_fn=activation_fn,
            activation_fn_2=activation_fn_2,
            output_activation_fn=output_activation_fn,
            lr=lr
        )
    else:
        cv_stats = None

    # Coerce the target variable to a contiguous NumPy array to ensure standardized evaluation
    y_numpy: np.ndarray = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y, dtype=float)

    # Declare y_pred so it is always defined before get_qof is called
    y_pred: np.ndarray

    # ==========================================
    # --- Model Training and Prediction ---
    # ==========================================

    # ------------------------------------------
    # 1. PyTorch Neural Network Models
    # ------------------------------------------
    if method in ['nn_2L', 'nn_3L', 'nn_4L']:
        # --- Target Dimension Handling & Scaling ---
        y_np = y_numpy.copy()

        # Scikit-learn's StandardScaler requires strictly 2D arrays (samples x features/targets)
        if len(y_np.shape) == 1:
            y_np = y_np.reshape(-1, 1)

        # Neural networks require standardized inputs to ensure stable and rapid gradient descent
        X_scaler = StandardScaler()
        X_scaled: np.ndarray = X_scaler.fit_transform(X)

        y_scaler = StandardScaler()
        y_scaled: np.ndarray = y_scaler.fit_transform(y_np)

        # --- Tensor Conversion ---
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        input_features = X.shape[1]
        output_classes = y_tensor.shape[1]

        # Suppress standard output to prevent the console from being flooded with epoch logs.
        # Use separate typed variables per NN class so Pylance can resolve .fit()/.predict() unambiguously.
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            if method == 'nn_2L':
                nn_model = NoHiddenLayerNN(input_size=input_features, output_size=output_classes, activation_fn=output_activation_fn)
                nn_model.lr = lr
                nn_model.fit(X_tensor, y_tensor)
                predictions, _ = nn_model.predict(X_tensor, y_tensor)
            elif method == 'nn_3L':
                nn_model = OneHiddenLayerNN(input_size=input_features, hidden_size=nn_hidden_1, output_size=output_classes, hidden_activation_fn=activation_fn, output_activation_fn=output_activation_fn)
                nn_model.lr = lr
                nn_model.fit(X_tensor, y_tensor)
                predictions, _ = nn_model.predict(X_tensor, y_tensor)
            else:  # nn_4L
                nn_model = TwoHiddenLayerNN(input_size=input_features, hidden_size_1=nn_hidden_1, hidden_size_2=nn_hidden_2, output_size=output_classes, hidden_activation_fn_1=activation_fn, hidden_activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn)
                nn_model.lr = lr
                nn_model.fit(X_tensor, y_tensor)
                predictions, _ = nn_model.predict(X_tensor, y_tensor)

        # Detach predictions from the computational graph and convert back to NumPy
        y_pred_scaled: np.ndarray = predictions.detach().cpu().numpy()

        # Inverse transform target predictions back to their original semantic scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        # Collapse predictions back to a 1D vector if the original true target was 1D
        if len(y_numpy.shape) == 1:
            y_pred = y_pred.flatten()

    # ------------------------------------------
    # 2. Statsmodels Regression Models
    # ------------------------------------------
    else:
        if method == 'linreg':
            ols_model = sm.OLS(y, X).fit()
            y_pred = np.asarray(ols_model.predict(X), dtype=float)

        elif method in ['ridge', 'lasso']:
            # --- Target Dimension Handling & Scaling ---
            y_np = y_numpy.copy()
            is_1d = False

            if len(y_np.shape) == 1:
                y_np = y_np.reshape(-1, 1)
                is_1d = True

            # Regularization (L1/L2) penalizes coefficient magnitude uniformly only if features are scaled
            X_scaler = StandardScaler()
            X_scaled = X_scaler.fit_transform(X)

            y_scaler = StandardScaler()
            y_scaled = y_scaler.fit_transform(y_np)

            # Fit scaled data using elastic net parameters (L1_wt=0 is pure Ridge, L1_wt=1 is pure Lasso)
            if method == 'ridge':
                reg_model = sm.OLS(y_scaled, X_scaled).fit_regularized(alpha=alpha, L1_wt=0.0)
            else:  # lasso
                reg_model = sm.OLS(y_scaled, X_scaled).fit_regularized(alpha=alpha, L1_wt=1.0)

            # Raw prediction exists in the scaled target space; coerce to ndarray to allow reshape
            y_pred_scaled = np.asarray(reg_model.predict(X_scaled), dtype=float)

            # Ensure 2D shape for the inverse transform operation
            if len(y_pred_scaled.shape) == 1:
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)

            # Revert to original domain for accurate QoF evaluation
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

            if is_1d:
                y_pred = y_pred.flatten()

        elif method == 'sqrt':
            # Fit on square-root transformed targets, predict, and square the result to revert
            sqrt_model = sm.OLS(np.sqrt(y), X).fit()
            y_pred = np.asarray(sqrt_model.predict(X), dtype=float) ** 2

        elif method == 'log1p':
            # Fit on log(1+y) transformed targets, revert using exponential minus 1
            log_model = sm.OLS(np.log1p(y), X).fit()
            y_pred = np.expm1(np.asarray(log_model.predict(X), dtype=float))

        elif method == 'boxcox':
            # Apply Box-Cox power transform to stabilize variance, revert using specific lambda
            bc_model = sm.OLS(boxcox(y, lambda_), X).fit()
            y_pred = inv_boxcox(np.asarray(bc_model.predict(X), dtype=float), lambda_)

        else:
            raise ValueError(f"Unknown method '{method}'. Valid methods are: 'linreg', 'ridge', 'lasso', 'sqrt', 'log1p', 'boxcox', 'nn_2L', 'nn_3L', 'nn_4L'")

    # ==========================================
    # --- Metrics Calculation & Return ---
    # ==========================================
    k = X.shape[1]

    # Calculate the 15 Quality of Fit metrics on the unscaled full dataset predictions
    qof = get_qof(y_numpy, y_pred, k)

    return (qof, cv_stats)
