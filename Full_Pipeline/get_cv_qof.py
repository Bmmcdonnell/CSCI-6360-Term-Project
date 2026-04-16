import os
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from typing import List, Optional, Union
from get_qof import get_qof
from neural_network_classes import NoHiddenLayerNN, OneHiddenLayerNN, TwoHiddenLayerNN


def get_cv_qof(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    method: str = "linreg",
    alpha: float = 0.0,
    lambda_: float = 0.0,
    n_splits: int = 5,
    nn_hidden_1: int = 100,
    nn_hidden_2: int = 50,
    activation_fn: Optional[nn.Module] = None,
    activation_fn_2: Optional[nn.Module] = None,
    output_activation_fn: Optional[nn.Module] = None,
    lr: float = 0.01
) -> List[List[float]]:
    """
    Performs k-fold cross-validation for a specified regression or neural network model 
    and calculates 15 Quality of Fit (QoF) metrics across each fold.
    
    This function acts as a unified pipeline handling standard linear regression, 
    regularized regression (Ridge, Lasso), target variable transformations 
    (Square Root, Log1p, Box-Cox), and PyTorch Neural Networks (2L, 3L, 4L).

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        method (str, optional): The regression method to evaluate. Valid options are:
            'linreg', 'ridge', 'lasso', 'sqrt', 'log1p', 'boxcox', 'nn_2L', 'nn_3L', 'nn_4L'. 
            Defaults to 'linreg'.
        alpha (float, optional): The regularization penalty strength used for Ridge 
            and Lasso models. Defaults to 0.0.
        lambda_ (float, optional): The transformation parameter used specifically 
            for the Box-Cox transformation. Defaults to 0.0.
        n_splits (int, optional): The number of cross-validation folds. Defaults to 5.
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
        List[List[float]]: A matrix of size (15 x n_splits) containing the QoF 
            metric values evaluated across all k-folds. Each of the 15 inner lists 
            represents a distinct metric.

    Raises:
        ValueError: If an unsupported `method` string is provided.
    """
    if activation_fn is None:
        activation_fn = nn.Sigmoid()
    if activation_fn_2 is None:
        activation_fn_2 = nn.Sigmoid()
    if output_activation_fn is None:
        output_activation_fn = nn.Identity()

    # ==========================================
    # --- Cross-Validation Setup ---
    # ==========================================
    # Initialize the KFold generator with a fixed random state for reproducibility
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Pre-allocate a list of 15 empty lists to store the aggregated metrics
    cv_stats: List[List[float]] = [[] for _ in range(15)]

    # Coerce target variable to a contiguous NumPy array for standardized slicing and evaluation
    y_numpy: np.ndarray = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.array(y)

    for train_idx, val_idx in kf.split(X):
        # --- Data Splitting ---
        # X and y are always pd.DataFrame / pd.Series so .iloc is safe and unambiguous
        X_tr: pd.DataFrame = X.iloc[train_idx]
        X_val: pd.DataFrame = X.iloc[val_idx]
        y_tr: Union[pd.Series, pd.DataFrame] = y.iloc[train_idx]
        y_val_np: np.ndarray = y_numpy[val_idx]

        # Declare y_pred so it is always defined before get_qof is called
        y_pred: np.ndarray

        # ==========================================
        # --- Model Training and Prediction ---
        # ==========================================

        # ------------------------------------------
        # 1. PyTorch Neural Network Models
        # ------------------------------------------
        if method in ['nn_2L', 'nn_3L', 'nn_4L']:
            # --- Target Dimension Handling for PyTorch & Scaling ---
            y_tr_np: np.ndarray = y_tr.to_numpy() if isinstance(y_tr, (pd.Series, pd.DataFrame)) else np.array(y_tr)
            y_val_tensor_np: np.ndarray = y_val_np.copy()

            # Scikit-learn's StandardScaler requires strictly 2D arrays (samples x features/targets)
            if len(y_tr_np.shape) == 1:
                y_tr_np = y_tr_np.reshape(-1, 1)
            if len(y_val_tensor_np.shape) == 1:
                y_val_tensor_np = y_val_tensor_np.reshape(-1, 1)

            # --- Data Scaling (Features & Targets) ---
            # Neural networks require standardized inputs to ensure stable gradient descent
            X_scaler = StandardScaler()
            X_tr_scaled: np.ndarray = X_scaler.fit_transform(X_tr)
            X_val_scaled: np.ndarray = X_scaler.transform(X_val)  # Prevent data leakage by transforming using training stats

            y_scaler = StandardScaler()
            y_tr_scaled: np.ndarray = y_scaler.fit_transform(y_tr_np)
            y_val_scaled: np.ndarray = y_scaler.transform(y_val_tensor_np)  # Scaled validation targets for internal test loss eval

            # --- Tensor Conversion ---
            X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_tr_tensor = torch.tensor(y_tr_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

            input_features = X_tr.shape[1]
            output_classes = y_tr_tensor.shape[1]

            # Suppress standard output to prevent the console from being flooded with epoch logs per fold
            # Use separate typed variables per NN class so Pylance can resolve .fit()/.predict() unambiguously
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                if method == 'nn_2L':
                    nn_model = NoHiddenLayerNN(input_size=input_features, output_size=output_classes, activation_fn=output_activation_fn)
                    nn_model.lr = lr
                    nn_model.fit(X_tr_tensor, y_tr_tensor)
                    predictions, _ = nn_model.predict(X_val_tensor, y_val_tensor)
                elif method == 'nn_3L':
                    nn_model = OneHiddenLayerNN(input_size=input_features, hidden_size=nn_hidden_1, output_size=output_classes, hidden_activation_fn=activation_fn, output_activation_fn=output_activation_fn)
                    nn_model.lr = lr
                    nn_model.fit(X_tr_tensor, y_tr_tensor)
                    predictions, _ = nn_model.predict(X_val_tensor, y_val_tensor)
                else:  # nn_4L
                    nn_model = TwoHiddenLayerNN(input_size=input_features, hidden_size_1=nn_hidden_1, hidden_size_2=nn_hidden_2, output_size=output_classes, hidden_activation_fn_1=activation_fn, hidden_activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn)
                    nn_model.lr = lr
                    nn_model.fit(X_tr_tensor, y_tr_tensor)
                    predictions, _ = nn_model.predict(X_val_tensor, y_val_tensor)

            # Detach predictions from the computational graph and convert back to NumPy
            y_pred_scaled: np.ndarray = predictions.detach().cpu().numpy()

            # Inverse transform target predictions back to their original semantic scale
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

            # Collapse predictions back to a 1D vector if the original true target was 1D
            if len(y_val_np.shape) == 1:
                y_pred = y_pred.flatten()

        # ------------------------------------------
        # 2. Statsmodels Regression Models
        # ------------------------------------------
        else:
            if method == 'linreg':
                ols_model = sm.OLS(y_tr, X_tr).fit()
                y_pred = np.asarray(ols_model.predict(X_val), dtype=float)

            elif method in ['ridge', 'lasso']:
                # --- Target Dimension Handling & Scaling ---
                y_tr_np = y_tr.to_numpy() if isinstance(y_tr, (pd.Series, pd.DataFrame)) else np.array(y_tr)

                is_1d = False
                if len(y_tr_np.shape) == 1:
                    y_tr_np = y_tr_np.reshape(-1, 1)
                    is_1d = True

                # Regularization (L1/L2) penalizes coefficient magnitude, so features must be uniformly scaled
                X_scaler = StandardScaler()
                X_tr_scaled = X_scaler.fit_transform(X_tr)
                X_val_scaled = X_scaler.transform(X_val)

                # Targets are also scaled to ensure the alpha hyperparameter behaves consistently
                y_scaler = StandardScaler()
                y_tr_scaled = y_scaler.fit_transform(y_tr_np)

                # Fit scaled data using elastic net parameters (L1_wt=0 is purely Ridge, L1_wt=1 is purely Lasso)
                if method == 'ridge':
                    reg_model = sm.OLS(y_tr_scaled, X_tr_scaled).fit_regularized(alpha=alpha, L1_wt=0.0)
                else:  # lasso
                    reg_model = sm.OLS(y_tr_scaled, X_tr_scaled).fit_regularized(alpha=alpha, L1_wt=1.0)

                # Raw prediction exists in the scaled target space; coerce to ndarray to allow reshape
                y_pred_scaled = np.asarray(reg_model.predict(X_val_scaled), dtype=float)

                # Ensure 2D shape for the inverse transform operation
                if len(y_pred_scaled.shape) == 1:
                    y_pred_scaled = y_pred_scaled.reshape(-1, 1)

                # Revert to original domain for accurate QoF evaluation
                y_pred = y_scaler.inverse_transform(y_pred_scaled)

                if is_1d:
                    y_pred = y_pred.flatten()

            elif method == 'sqrt':
                # Fit on square-root transformed targets, predict, and square the result to revert
                sqrt_model = sm.OLS(np.sqrt(y_tr), X_tr).fit()
                y_pred = np.asarray(sqrt_model.predict(X_val), dtype=float) ** 2

            elif method == 'log1p':
                # Fit on log(1+y) transformed targets, revert using exponential minus 1
                log_model = sm.OLS(np.log1p(y_tr), X_tr).fit()
                y_pred = np.expm1(np.asarray(log_model.predict(X_val), dtype=float))

            elif method == 'boxcox':
                # Apply Box-Cox power transform to stabilize variance, revert using specific lambda
                bc_model = sm.OLS(boxcox(y_tr, lambda_), X_tr).fit()
                y_pred = inv_boxcox(np.asarray(bc_model.predict(X_val), dtype=float), lambda_)

            else:
                raise ValueError(f"Unknown method '{method}'. Valid methods are: 'linreg', 'ridge', 'lasso', 'sqrt', 'log1p', 'boxcox', 'nn_2L', 'nn_3L', 'nn_4L'")

        # ==========================================
        # --- Model Evaluation ---
        # ==========================================
        k = X.shape[1]

        # Evaluate QoF on the unscaled/inversely-transformed original true values and final predictions
        temp_qof = get_qof(y_val_np, y_pred, k)

        # Append the calculated metrics to their respective lists
        for i in range(15):
            cv_stats[i].append(temp_qof[i])

    return cv_stats
