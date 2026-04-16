import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from get_qof2 import get_qof2


def _assert_cv_stats(cv_stats: Optional[List[List[float]]]) -> List[List[float]]:
    """Assert that cv_stats is not None (it won't be when cv=True) for Pylance narrowing."""
    assert cv_stats is not None, "cv_stats should not be None when cv=True"
    return cv_stats


def select_single_feature(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    in_cols: List[str],
    out_cols: List[str],
    method: str = 'linreg',
    alpha: float = 0.0,
    lambda_: float = 0.0,
    nn_hidden_1: int = 100,
    nn_hidden_2: int = 50,
    activation_fn: Optional[nn.Module] = None,
    activation_fn_2: Optional[nn.Module] = None,
    output_activation_fn: Optional[nn.Module] = None,
    lr: float = 0.01,
    metric: int = 0
) -> Tuple[List[str], List[str], str, List[float], List[List[float]]]:
    """
    Evaluates all candidate features and selects the single best feature to add to the model.
    
    Iterates through the features currently not in the model (`out_cols`), temporarily adds each 
    one, and calculates the Quality of Fit (QoF) without Cross-Validation to ensure computational 
    speed. It selects the feature that improves the specified target metric the most, then 
    re-evaluates the final updated model with Cross-Validation enabled.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        in_cols (List[str]): A list of column names currently included in the model.
        out_cols (List[str]): A list of candidate column names not yet in the model.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        nn_hidden_1 (int): Nodes in the first hidden layer for NN methods. Defaults to 100.
        nn_hidden_2 (int): Nodes in the second hidden layer for NN methods. Defaults to 50.
        activation_fn (nn.Module): Hidden-layer activation for layer 1. Defaults to nn.Sigmoid().
        activation_fn_2 (nn.Module): Hidden-layer activation for layer 2. Defaults to nn.Sigmoid().
        lr (float): Learning rate for NN optimizers. Defaults to 0.01.
        metric (int): The index of the QoF metric to optimize (e.g., 0 for R^2, 13 for AIC). 
                      Defaults to 0.

    Returns:
        Tuple containing updated included columns, updated excluded columns, the selected 
        feature added, the new model's QoF metrics, and the new model's CV statistics.
    """
    if activation_fn is None:
        activation_fn = nn.Sigmoid()
    if activation_fn_2 is None:
        activation_fn_2 = nn.Sigmoid()
    if output_activation_fn is None:
        output_activation_fn = nn.Identity()
    # ==========================================
    # --- Metric Initialization ---
    # ==========================================
    if metric in [0, 1, 12]:
        best_metric = -float('inf')
    elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
        best_metric = float('inf')
    else:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")

    feature_to_add = out_cols[0]

    # ==========================================
    # --- Feature Evaluation ---
    # ==========================================
    for col in out_cols:
        new_cols = in_cols + [col]
        temp_X = X[new_cols].copy()

        # Evaluate the temporary model (without Cross-Validation for speed)
        (temp_qof, _) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=False, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)

        cur_metric = temp_qof[metric]

        if metric in [0, 1, 12]:
            if cur_metric > best_metric:
                best_metric = cur_metric
                feature_to_add = col
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            if cur_metric < best_metric:
                best_metric = cur_metric
                feature_to_add = col

    # ==========================================
    # --- Finalize Selection ---
    # ==========================================
    new_in_cols = in_cols + [feature_to_add]
    new_out_cols = [col for col in out_cols if col != feature_to_add]

    temp_X = X[new_in_cols].copy()

    # Re-evaluate the best model, this time securing Cross-Validation statistics
    (best_qof, best_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
    best_cv_stats = _assert_cv_stats(best_cv_stats_raw)

    return (new_in_cols, new_out_cols, feature_to_add, best_qof, best_cv_stats)


def forward_select_all(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    start_cols: Optional[List[str]] = None,
    method: str = 'linreg',
    alpha: float = 0.0,
    lambda_: float = 0.0,
    nn_hidden_1: int = 100,
    nn_hidden_2: int = 50,
    activation_fn: Optional[nn.Module] = None,
    activation_fn_2: Optional[nn.Module] = None,
    output_activation_fn: Optional[nn.Module] = None,
    lr: float = 0.01,
    metric: int = 0
) -> Tuple[List[str], List[List[float]], List[List[List[float]]]]:
    """
    Performs Forward Selection feature engineering across the entire dataset.
    
    Iteratively adds the single best feature to the model using `select_single_feature` until 
    all available features have been included. Automatically handles Null/Intercept-only 
    baselines and tracks the progression of QoF metrics throughout the entire sequence.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        start_cols (Optional[List[str]]): A list of starting column names. Defaults to None.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        nn_hidden_1 (int): Nodes in the first hidden layer for NN methods. Defaults to 100.
        nn_hidden_2 (int): Nodes in the second hidden layer for NN methods. Defaults to 50.
        activation_fn (nn.Module): Hidden-layer activation for layer 1. Defaults to nn.Sigmoid().
        activation_fn_2 (nn.Module): Hidden-layer activation for layer 2. Defaults to nn.Sigmoid().
        lr (float): Learning rate for NN optimizers. Defaults to 0.01.
        metric (int): The index of the QoF metric to optimize. Defaults to 0 (R^2).

    Returns:
        Tuple containing a list of features in the order they were selected, a list of 
        historical QoF metrics, and a list of historical CV statistics.
    """
    if activation_fn is None:
        activation_fn = nn.Sigmoid()
    if activation_fn_2 is None:
        activation_fn_2 = nn.Sigmoid()
    if output_activation_fn is None:
        output_activation_fn = nn.Identity()
    
    # ==========================================
    # --- Initialization ---
    # ==========================================
    if start_cols is None:
        if 'intercept' in X.columns:
            start_cols_copy: List[str] = ['intercept']
            in_cols: List[str] = ['intercept']
            for_sel_features: List[str] = ['intercept']
        else:
            start_cols_copy = []
            in_cols = []
            for_sel_features = []
    else:
        start_cols_copy = start_cols.copy()
        in_cols = start_cols.copy()
        for_sel_features = start_cols.copy()

    qof_list: List[List[float]] = []
    cv_stats_list: List[List[List[float]]] = []

    # Handle Null/Intercept-only baseline if starting with empty predictors
    if len(in_cols) == 0:
        if 'intercept' not in X.columns:
            temp_X = X.copy()
            temp_X['intercept'] = 1
            X_int = temp_X[['intercept']].copy()
            (int_qof, int_cv_stats_raw) = get_qof2(X_int, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
            int_cv_stats = _assert_cv_stats(int_cv_stats_raw)
            for_sel_features.append('Null')
            qof_list.append(int_qof)
            cv_stats_list.append(int_cv_stats)
    else:
        temp_X = X[in_cols].copy()
        (qof, cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
        cv_stats = _assert_cv_stats(cv_stats_raw)
        qof_list = [qof]
        cv_stats_list = [cv_stats]

    out_cols = [col for col in X.columns if col not in start_cols_copy]

    # ==========================================
    # --- Forward Selection Loop ---
    # ==========================================
    while True:
        (new_in_cols, new_out_cols, feature_to_add, best_qof, best_cv_stats) = select_single_feature(
            X, y, in_cols, out_cols, method=method, alpha=alpha, lambda_=lambda_, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr, metric=metric
        )

        in_cols = new_in_cols.copy()
        out_cols = new_out_cols.copy()
        for_sel_features.append(feature_to_add)
        qof_list.append(best_qof)
        cv_stats_list.append(best_cv_stats)

        if len(out_cols) == 0:
            break

    return (for_sel_features, qof_list, cv_stats_list)


def eliminate_single_feature(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    in_cols: List[str],
    method: str = 'linreg',
    alpha: float = 0.0,
    lambda_: float = 0.0,
    nn_hidden_1: int = 100,
    nn_hidden_2: int = 50,
    activation_fn: Optional[nn.Module] = None,
    activation_fn_2: Optional[nn.Module] = None,
    output_activation_fn: Optional[nn.Module] = None,
    lr: float = 0.01,
    metric: int = 0
) -> Tuple[List[str], str, List[float], List[List[float]]]:
    """
    Evaluates all included features and selects the single worst feature to remove.
    
    Iterates through the features currently in the model (`in_cols`), temporarily removes each 
    one (excluding the intercept), and calculates the resulting QoF without Cross-Validation 
    for speed. It selects the feature whose removal yields the best metric (or hurts it the least) 
    and re-evaluates the pruned model with Cross-Validation enabled.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        in_cols (List[str]): A list of column names currently included in the model.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        nn_hidden_1 (int): Nodes in the first hidden layer for NN methods. Defaults to 100.
        nn_hidden_2 (int): Nodes in the second hidden layer for NN methods. Defaults to 50.
        activation_fn (nn.Module): Hidden-layer activation for layer 1. Defaults to nn.Sigmoid().
        activation_fn_2 (nn.Module): Hidden-layer activation for layer 2. Defaults to nn.Sigmoid().
        lr (float): Learning rate for NN optimizers. Defaults to 0.01.
        metric (int): The index of the QoF metric to optimize. Defaults to 0.

    Returns:
        Tuple containing the updated included columns, the selected feature removed, 
        the new model's QoF metrics, and the new model's CV statistics.
    """
    if activation_fn is None:
        activation_fn = nn.Sigmoid()
    if activation_fn_2 is None:
        activation_fn_2 = nn.Sigmoid()
    if output_activation_fn is None:
        output_activation_fn = nn.Identity()
    
    if metric in [0, 1, 12]:
        best_metric = -float('inf')
    elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
        best_metric = float('inf')
    else:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")

    feature_to_remove = in_cols[0]

    if 'intercept' in in_cols:
        in_cols_copy = [col for col in in_cols if col != 'intercept']
    else:
        in_cols_copy = in_cols.copy()

    for col in in_cols_copy:
        if 'intercept' in X.columns:
            new_cols = [col2 for col2 in in_cols_copy if col2 != col] + ['intercept']
        else:
            new_cols = [col2 for col2 in in_cols_copy if col2 != col]

        temp_X = X[new_cols].copy()
        (temp_qof, _) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=False, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
        cur_metric = temp_qof[metric]

        if metric in [0, 1, 12]:
            if cur_metric > best_metric:
                best_metric = cur_metric
                feature_to_remove = col
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            if cur_metric < best_metric:
                best_metric = cur_metric
                feature_to_remove = col

    new_in_cols = [col2 for col2 in in_cols if col2 != feature_to_remove]
    temp_X = X[new_in_cols].copy()

    (best_qof, best_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
    best_cv_stats = _assert_cv_stats(best_cv_stats_raw)

    return (new_in_cols, feature_to_remove, best_qof, best_cv_stats)


def backward_eliminate_all(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    start_cols: Optional[List[str]] = None,
    method: str = 'linreg',
    alpha: float = 0.0,
    lambda_: float = 0.0,
    nn_hidden_1: int = 100,
    nn_hidden_2: int = 50,
    activation_fn: Optional[nn.Module] = None,
    activation_fn_2: Optional[nn.Module] = None,
    output_activation_fn: Optional[nn.Module] = None,
    lr: float = 0.01,
    metric: int = 0
) -> Tuple[List[str], List[List[float]], List[List[List[float]]]]:
    """
    Performs Backward Elimination feature engineering across the entire dataset.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        start_cols (Optional[List[str]]): A list of starting column names. Defaults to None.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        nn_hidden_1 (int): Nodes in the first hidden layer for NN methods. Defaults to 100.
        nn_hidden_2 (int): Nodes in the second hidden layer for NN methods. Defaults to 50.
        activation_fn (nn.Module): Hidden-layer activation for layer 1. Defaults to nn.Sigmoid().
        activation_fn_2 (nn.Module): Hidden-layer activation for layer 2. Defaults to nn.Sigmoid().
        lr (float): Learning rate for NN optimizers. Defaults to 0.01.
        metric (int): The index of the QoF metric to optimize. Defaults to 0 (R^2).

    Returns:
        Tuple containing a list of features ordered by importance (reverse elimination order), 
        a list of historical QoF metrics, and a list of historical CV statistics.
    """
    if activation_fn is None:
        activation_fn = nn.Sigmoid()
    if activation_fn_2 is None:
        activation_fn_2 = nn.Sigmoid()
    if output_activation_fn is None:
        output_activation_fn = nn.Identity()

    if start_cols is None:
        in_cols: List[str] = list(X.columns)
    elif len(start_cols) == 0:
        raise ValueError("start_cols must be non-empty")
    else:
        in_cols = start_cols.copy()

    temp_X = X[in_cols].copy()
    (qof, cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
    cv_stats = _assert_cv_stats(cv_stats_raw)

    bac_eli_features: List[str] = []
    qof_list: List[List[float]] = [qof]
    cv_stats_list: List[List[List[float]]] = [cv_stats]

    while True:
        (new_in_cols, feature_to_remove, best_qof, best_cv_stats) = eliminate_single_feature(
            X, y, in_cols, method=method, alpha=alpha, lambda_=lambda_, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr, metric=metric
        )

        in_cols = new_in_cols.copy()
        bac_eli_features.append(feature_to_remove)
        qof_list.append(best_qof)
        cv_stats_list.append(best_cv_stats)

        if len(in_cols) == 1:
            break

    bac_eli_features.append(in_cols[0])

    if in_cols[0] != 'intercept':
        temp_X = X.copy()
        temp_X['intercept'] = 1
        X_int = temp_X[['intercept']].copy()
        (int_qof, int_cv_stats_raw) = get_qof2(X_int, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
        int_cv_stats = _assert_cv_stats(int_cv_stats_raw)
        bac_eli_features.append('Null')
        qof_list.append(int_qof)
        cv_stats_list.append(int_cv_stats)

    bac_eli_features.reverse()
    qof_list.reverse()
    cv_stats_list.reverse()

    return (bac_eli_features, qof_list, cv_stats_list)


def stepwise_selection(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    start_cols: Optional[List[str]] = None,
    method: str = 'linreg',
    alpha: float = 0.0,
    lambda_: float = 0.0,
    nn_hidden_1: int = 100,
    nn_hidden_2: int = 50,
    activation_fn: Optional[nn.Module] = None,
    activation_fn_2: Optional[nn.Module] = None,
    output_activation_fn: Optional[nn.Module] = None,
    lr: float = 0.01,
    metric: int = 1
) -> Tuple[List[str], List[List[float]], List[List[List[float]]]]:
    """
    Performs Stepwise Selection feature engineering to build an optimal statistical model.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        start_cols (Optional[List[str]]): A list of starting column names. Defaults to None.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        nn_hidden_1 (int): Nodes in the first hidden layer for NN methods. Defaults to 100.
        nn_hidden_2 (int): Nodes in the second hidden layer for NN methods. Defaults to 50.
        activation_fn (nn.Module): Hidden-layer activation for layer 1. Defaults to nn.Sigmoid().
        activation_fn_2 (nn.Module): Hidden-layer activation for layer 2. Defaults to nn.Sigmoid().
        lr (float): Learning rate for NN optimizers. Defaults to 0.01.
        metric (int): The index of the QoF metric to optimize. Defaults to 1 (Adjusted R^2).

    Returns:
        Tuple containing a list of the features in the final step-selected model, a list of 
        the historical progression of QoF metrics, and a list of historical CV statistics.
    """
    if activation_fn is None:
        activation_fn = nn.Sigmoid()
    if activation_fn_2 is None:
        activation_fn_2 = nn.Sigmoid()
    if output_activation_fn is None:
        output_activation_fn = nn.Identity()

    if metric not in [0, 1, 3, 4, 5, 6, 7, 8, 12, 13, 14]:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")

    qof_dict: Dict[str, List[float]] = {}
    cv_stats_dict: Dict[str, List[List[float]]] = {}
    cur_metric: float = 0.0

    # ==========================================
    # --- Initialization ---
    # ==========================================
    if start_cols is None:
        if 'intercept' in X.columns:
            start_cols_copy: List[str] = ['intercept']
            in_cols: List[str] = ['intercept']
            step_sel_features: List[str] = ['intercept']

            temp_X = X[in_cols].copy()
            (temp_qof, temp_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
            temp_cv_stats = _assert_cv_stats(temp_cv_stats_raw)
            qof_dict['intercept'] = temp_qof
            cv_stats_dict['intercept'] = temp_cv_stats
            cur_metric = float(temp_qof[metric])
        else:
            start_cols_copy = []
            in_cols = []
            step_sel_features = []
    else:
        start_cols_copy = start_cols.copy()
        in_cols = start_cols.copy()
        step_sel_features = start_cols.copy()

        # Initialize temp_qof so it is always bound even if in_cols is empty
        temp_qof: List[float] = [0.0] * 15
        for i in range(len(in_cols)):
            temp_in_cols = in_cols[:i+1]
            temp_X = X[temp_in_cols].copy()
            (temp_qof, temp_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
            temp_cv_stats = _assert_cv_stats(temp_cv_stats_raw)
            qof_dict[in_cols[i]] = temp_qof
            cv_stats_dict[in_cols[i]] = temp_cv_stats
        cur_metric = float(temp_qof[metric])

    # Handle Null/Intercept-only baseline
    if len(in_cols) == 0:
        if 'intercept' not in X.columns:
            temp_X = X.copy()
            temp_X['intercept'] = 1
            X_int = temp_X[['intercept']].copy()
            (int_qof, int_cv_stats_raw) = get_qof2(X_int, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
            int_cv_stats = _assert_cv_stats(int_cv_stats_raw)
            step_sel_features.append('Null')
            qof_dict['Null'] = int_qof
            cv_stats_dict['Null'] = int_cv_stats
            cur_metric = float(int_qof[metric])
        elif metric in [0, 1, 12]:
            cur_metric = -float('inf')
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            cur_metric = float('inf')

    out_cols = [col for col in X.columns if col not in start_cols_copy]
    num_cols = X.shape[1]

    # ==========================================
    # --- Stepwise Selection Loop ---
    # ==========================================
    while True:

        # Scenario A: model is empty; can only add
        if len(in_cols) <= 1:
            (sel_new_in_cols, sel_new_out_cols, feature_to_add, sel_best_qof, sel_best_cv_stats) = select_single_feature(
                X, y, in_cols, out_cols, method=method, alpha=alpha, lambda_=lambda_, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr, metric=metric
            )

            if metric in [0, 1, 12] and (sel_best_qof[metric] >= cur_metric):
                in_cols = sel_new_in_cols.copy()
                out_cols = sel_new_out_cols.copy()
                step_sel_features.append(feature_to_add)
                qof_dict[feature_to_add] = sel_best_qof
                cv_stats_dict[feature_to_add] = sel_best_cv_stats
                cur_metric = float(sel_best_qof[metric])
            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and (sel_best_qof[metric] <= cur_metric):
                in_cols = sel_new_in_cols.copy()
                out_cols = sel_new_out_cols.copy()
                step_sel_features.append(feature_to_add)
                qof_dict[feature_to_add] = sel_best_qof
                cv_stats_dict[feature_to_add] = sel_best_cv_stats
                cur_metric = float(sel_best_qof[metric])
            else:
                break

        # Scenario B: model is full; can only remove
        elif len(in_cols) == num_cols:
            (bac_new_in_cols, feature_to_remove, bac_best_qof, bac_best_cv_stats) = eliminate_single_feature(
                X, y, in_cols, method=method, alpha=alpha, lambda_=lambda_, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr, metric=metric
            )

            if metric in [0, 1, 12] and (bac_best_qof[metric] >= cur_metric):
                old_in_cols = in_cols.copy()
                in_cols = bac_new_in_cols.copy()
                temp_in_cols = bac_new_in_cols.copy()
                out_cols = [col for col in X.columns if col not in temp_in_cols]
                step_sel_features = [feat for feat in step_sel_features if feat != feature_to_remove]

                flag = False
                for i in range(len(old_in_cols)):
                    if old_in_cols[i] == feature_to_remove:
                        flag = True
                    if flag and (i < len(in_cols)):
                        temp_in_cols = in_cols[:i+1]
                        temp_X = X[temp_in_cols].copy()
                        (temp_qof, temp_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
                        temp_cv_stats = _assert_cv_stats(temp_cv_stats_raw)
                        qof_dict[in_cols[i]] = temp_qof
                        cv_stats_dict[in_cols[i]] = temp_cv_stats
                cur_metric = float(bac_best_qof[metric])

            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and (bac_best_qof[metric] <= cur_metric):
                old_in_cols = in_cols.copy()
                in_cols = bac_new_in_cols.copy()
                temp_in_cols = bac_new_in_cols.copy()
                out_cols = [col for col in X.columns if col not in temp_in_cols]
                step_sel_features = [feat for feat in step_sel_features if feat != feature_to_remove]

                flag = False
                for i in range(len(old_in_cols)):
                    if old_in_cols[i] == feature_to_remove:
                        flag = True
                    if flag and (i < len(in_cols)):
                        temp_in_cols = in_cols[:i+1]
                        temp_X = X[temp_in_cols].copy()
                        (temp_qof, temp_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
                        temp_cv_stats = _assert_cv_stats(temp_cv_stats_raw)
                        qof_dict[in_cols[i]] = temp_qof
                        cv_stats_dict[in_cols[i]] = temp_cv_stats
                cur_metric = float(bac_best_qof[metric])

            else:
                break

        # Scenario C: intermediate; evaluate both
        else:
            (sel_new_in_cols, sel_new_out_cols, feature_to_add, sel_best_qof, sel_best_cv_stats) = select_single_feature(
                X, y, in_cols, out_cols, method=method, alpha=alpha, lambda_=lambda_, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr, metric=metric
            )
            (bac_new_in_cols, feature_to_remove, bac_best_qof, bac_best_cv_stats) = eliminate_single_feature(
                X, y, in_cols, method=method, alpha=alpha, lambda_=lambda_, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr, metric=metric
            )

            if metric in [0, 1, 12] and ((sel_best_qof[metric] >= cur_metric) or (bac_best_qof[metric] >= cur_metric)):
                if sel_best_qof[metric] >= bac_best_qof[metric]:
                    in_cols = sel_new_in_cols.copy()
                    out_cols = sel_new_out_cols.copy()
                    step_sel_features.append(feature_to_add)
                    qof_dict[feature_to_add] = sel_best_qof
                    cv_stats_dict[feature_to_add] = sel_best_cv_stats
                    cur_metric = float(sel_best_qof[metric])
                else:
                    old_in_cols = in_cols.copy()
                    in_cols = bac_new_in_cols.copy()
                    temp_in_cols = bac_new_in_cols.copy()
                    out_cols = [col for col in X.columns if col not in temp_in_cols]
                    step_sel_features = [feat for feat in step_sel_features if feat != feature_to_remove]
                    flag = False
                    for i in range(len(old_in_cols)):
                        if old_in_cols[i] == feature_to_remove:
                            flag = True
                        if flag and (i < len(in_cols)):
                            temp_in_cols = in_cols[:i+1]
                            temp_X = X[temp_in_cols].copy()
                            (temp_qof, temp_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
                            temp_cv_stats = _assert_cv_stats(temp_cv_stats_raw)
                            qof_dict[in_cols[i]] = temp_qof
                            cv_stats_dict[in_cols[i]] = temp_cv_stats
                    cur_metric = float(bac_best_qof[metric])

            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and ((sel_best_qof[metric] <= cur_metric) or (bac_best_qof[metric] <= cur_metric)):
                if sel_best_qof[metric] <= bac_best_qof[metric]:
                    in_cols = sel_new_in_cols.copy()
                    out_cols = sel_new_out_cols.copy()
                    step_sel_features.append(feature_to_add)
                    qof_dict[feature_to_add] = sel_best_qof
                    cv_stats_dict[feature_to_add] = sel_best_cv_stats
                    cur_metric = float(sel_best_qof[metric])
                else:
                    old_in_cols = in_cols.copy()
                    in_cols = bac_new_in_cols.copy()
                    temp_in_cols = bac_new_in_cols.copy()
                    out_cols = [col for col in X.columns if col not in temp_in_cols]
                    step_sel_features = [feat for feat in step_sel_features if feat != feature_to_remove]
                    flag = False
                    for i in range(len(old_in_cols)):
                        if old_in_cols[i] == feature_to_remove:
                            flag = True
                        if flag and (i < len(in_cols)):
                            temp_in_cols = in_cols[:i+1]
                            temp_X = X[temp_in_cols].copy()
                            (temp_qof, temp_cv_stats_raw) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True, nn_hidden_1=nn_hidden_1, nn_hidden_2=nn_hidden_2, activation_fn=activation_fn, activation_fn_2=activation_fn_2, output_activation_fn=output_activation_fn, lr=lr)
                            temp_cv_stats = _assert_cv_stats(temp_cv_stats_raw)
                            qof_dict[in_cols[i]] = temp_qof
                            cv_stats_dict[in_cols[i]] = temp_cv_stats
                    cur_metric = float(bac_best_qof[metric])

            else:
                break

    # ==========================================
    # --- Format Results Return ---
    # ==========================================
    qof_list: List[List[float]] = []
    cv_stats_list: List[List[List[float]]] = []

    for col in step_sel_features:
        qof_list.append(qof_dict[col])
        cv_stats_list.append(cv_stats_dict[col])

    return (step_sel_features, qof_list, cv_stats_list)
