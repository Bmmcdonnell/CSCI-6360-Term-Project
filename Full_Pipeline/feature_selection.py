import numpy as np
import pandas as pd
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from save_plots import save_rsq_plot, save_aic_bic_plot
from feature_selection_methods import forward_select_all, backward_eliminate_all, stepwise_selection


def feature_selection(
    key: str,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    X_2: pd.DataFrame,
    data_name: str,
    folder_name: str,
    ridge_alpha: float,
    lasso_alpha: float,
    boxcox_lambda: float,
    order2reg_alpha: float,
    nn_3L_hidden_1: int = 100,
    nn_4L_hidden_1: int = 100,
    nn_4L_hidden_2: int = 50,
    nn_2L_output_activation_fn: Optional[nn.Module] = None,
    nn_2L_lr: float = 0.01,
    nn_3L_activation_fn: Optional[nn.Module] = None,
    nn_3L_output_activation_fn: Optional[nn.Module] = None,
    nn_3L_lr: float = 0.01,
    nn_4L_activation_fn: Optional[nn.Module] = None,
    nn_4L_activation_fn_2: Optional[nn.Module] = None,
    nn_4L_output_activation_fn: Optional[nn.Module] = None,
    nn_4L_lr: float = 0.01
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Performs feature selection (Forward, Backward, or Stepwise) across multiple regression models.
    
    Evaluates Linear, Ridge, Lasso, Square Root, Log1p, Box-Cox, Order-2 Polynomial models, 
    and Neural Networks (2L, 3L, 4L).
    Calculates and tracks Quality of Fit (QoF) metrics at each stage of the selection process.
    Automatically generates and saves high-resolution plots for R^2 vs. number of features, 
    as well as AIC/BIC vs. number of features.

    Args:
        key (str): The feature selection method to use ('Forward', 'Backward', or 'Stepwise').
        X (pd.DataFrame): The standard input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        X_2 (pd.DataFrame): The quadratic/interaction feature matrix (Order 2).
        data_name (str): The name of the dataset (used for plot titles and logging).
        folder_name (str): The directory path where output plots will be saved.
        ridge_alpha (float): The optimized regularization hyperparameter for Ridge Regression.
        lasso_alpha (float): The optimized regularization hyperparameter for Lasso Regression.
        boxcox_lambda (float): The optimized transformation hyperparameter for Box-Cox Regression.
        order2reg_alpha (float): The optimized regularization hyperparameter for Order 2 Regression.
        nn_3L_hidden_1 (int): Tuned hidden layer size for nn_3L. Defaults to 100.
        nn_4L_hidden_1 (int): Tuned first hidden layer size for nn_4L. Defaults to 100.
        nn_4L_hidden_2 (int): Tuned second hidden layer size for nn_4L. Defaults to 50.
        nn_2L_activation_fn (nn.Module): Tuned hidden activation for nn_2L. Defaults to nn.Sigmoid().
        nn_2L_output_activation_fn (nn.Module): Tuned output activation for nn_2L. Defaults to nn.Identity().
        nn_2L_lr (float): Tuned learning rate for nn_2L. Defaults to 0.01.
        nn_3L_activation_fn (nn.Module): Tuned hidden activation for nn_3L. Defaults to nn.Sigmoid().
        nn_3L_output_activation_fn (nn.Module): Tuned output activation for nn_3L. Defaults to nn.Identity().
        nn_3L_lr (float): Tuned learning rate for nn_3L. Defaults to 0.01.
        nn_4L_activation_fn (nn.Module): Tuned hidden activation for nn_4L layer 1. Defaults to nn.Sigmoid().
        nn_4L_activation_fn_2 (nn.Module): Tuned hidden activation for nn_4L layer 2. Defaults to nn.Sigmoid().
        nn_4L_output_activation_fn (nn.Module): Tuned output activation for nn_4L. Defaults to nn.Identity().
        nn_4L_lr (float): Tuned learning rate for nn_4L. Defaults to 0.01.

    Returns:
        Tuple of 10 Lists[str] containing the selected feature names for each model:
        (reg, ridge, lasso, sqrt, log1p, boxcox, order2reg, nn_2L, nn_3L, nn_4L).
    """
    if nn_2L_output_activation_fn is None:
        nn_2L_output_activation_fn = nn.Identity()
    if nn_3L_activation_fn is None:
        nn_3L_activation_fn = nn.Sigmoid()
    if nn_3L_output_activation_fn is None:
        nn_3L_output_activation_fn = nn.Identity()
    if nn_4L_activation_fn is None:
        nn_4L_activation_fn = nn.Sigmoid()
    if nn_4L_activation_fn_2 is None:
        nn_4L_activation_fn_2 = nn.Sigmoid()
    if nn_4L_output_activation_fn is None:
        nn_4L_output_activation_fn = nn.Identity()

    # Set the method name and metric value based on the key
    if key == 'Forward':
        method = 'Forward Selection'
        metric_val = 0
    elif key == 'Backward':
        method = 'Backward Elimination'
        metric_val = 0
    elif key == 'Stepwise':
        method = 'Stepwise Selection'
        metric_val = 1
    else:
        raise ValueError(f"key must be one of 'Forward', 'Backward', or 'Stepwise'. Received {key}")

    # Define the configurations for all 10 model types
    models_config = [
        {"name": "Linear Regression",     "prefix": "Reg",       "method": "linreg", "X_data": X,   "kwargs": {}},
        {"name": "Ridge Regression",      "prefix": "Ridge",     "method": "ridge",  "X_data": X,   "kwargs": {"alpha": ridge_alpha}},
        {"name": "Lasso Regression",      "prefix": "Lasso",     "method": "lasso",  "X_data": X,   "kwargs": {"alpha": lasso_alpha}},
        {"name": "Sqrt Transformation",   "prefix": "Sqrt",      "method": "sqrt",   "X_data": X,   "kwargs": {}},
        {"name": "Log1p Transformation",  "prefix": "Log1p",     "method": "log1p",  "X_data": X,   "kwargs": {}},
        {"name": "Box-Cox Transformation","prefix": "BoxCox",    "method": "boxcox", "X_data": X,   "kwargs": {"lambda_": boxcox_lambda}},
        {"name": "Order 2 Regression",    "prefix": "Order2Reg", "method": "ridge",  "X_data": X_2, "kwargs": {"alpha": order2reg_alpha}},
        {"name": "2L Neural Network",     "prefix": "NN_2L",     "method": "nn_2L",  "X_data": X,   "kwargs": {"output_activation_fn": nn_2L_output_activation_fn, "lr": nn_2L_lr}},
        {"name": "3L Neural Network",     "prefix": "NN_3L",     "method": "nn_3L",  "X_data": X,   "kwargs": {"nn_hidden_1": nn_3L_hidden_1, "activation_fn": nn_3L_activation_fn, "output_activation_fn": nn_3L_output_activation_fn, "lr": nn_3L_lr}},
        {"name": "4L Neural Network",     "prefix": "NN_4L",     "method": "nn_4L",  "X_data": X,   "kwargs": {"nn_hidden_1": nn_4L_hidden_1, "nn_hidden_2": nn_4L_hidden_2, "activation_fn": nn_4L_activation_fn, "activation_fn_2": nn_4L_activation_fn_2, "output_activation_fn": nn_4L_output_activation_fn, "lr": nn_4L_lr}}
    ]

    all_features: List[List[str]] = []

    for cfg in models_config:
        print("------------------------------------")
        print(f"{data_name} {method} for {cfg['name']}")
        print("------------------------------------")

        # Execute the appropriate feature selection function
        if key == 'Forward':
            features, qof_list, cv_stats_list = forward_select_all(cfg['X_data'], y, start_cols=None, method=cfg['method'], metric=metric_val, **cfg['kwargs'])
        elif key == 'Backward':
            features, qof_list, cv_stats_list = backward_eliminate_all(cfg['X_data'], y, start_cols=None, method=cfg['method'], metric=metric_val, **cfg['kwargs'])
        elif key == 'Stepwise':
            features, qof_list, cv_stats_list = stepwise_selection(cfg['X_data'], y, start_cols=None, method=cfg['method'], metric=metric_val, **cfg['kwargs'])

        # Extract tracking metrics — wrap each value in float() to narrow from floating[Any]
        x_vals: List[float] = [float(i) for i in range(len(features))]
        r_sq:     List[float] = [float(100 * qof_list[i][0])  for i in range(len(features))]
        adj_r_sq: List[float] = [float(100 * qof_list[i][1])  for i in range(len(features))]
        smape:    List[float] = [float(qof_list[i][8])         for i in range(len(features))]
        r_sq_cv:  List[float] = [float(100 * np.mean(cv_stats_list[i][0])) for i in range(len(features))]
        aic:      List[float] = [float(qof_list[i][13])        for i in range(len(features))]
        bic:      List[float] = [float(qof_list[i][14])        for i in range(len(features))]

        # Generate and save plots
        save_rsq_plot(key, x_vals, r_sq, adj_r_sq, smape, r_sq_cv, method, data_name, folder_name, cfg['name'], cfg['prefix'])
        save_aic_bic_plot(key, x_vals, aic, bic, method, data_name, folder_name, cfg['name'], cfg['prefix'])

        all_features.append(features)

    # Unpack exactly 10 lists to satisfy the return type
    return (
        all_features[0], all_features[1], all_features[2], all_features[3], all_features[4],
        all_features[5], all_features[6], all_features[7], all_features[8], all_features[9]
    )
