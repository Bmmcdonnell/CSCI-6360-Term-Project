import pandas as pd
from typing import cast
from model_evaluation import lin_reg, ridge_reg, lasso_reg, sqrt_reg, log1p_reg, boxcox_reg, order2_reg, nn_2L, nn_3L, nn_4L
from feature_selection import feature_selection
from latex_tables import is_oos_comparison, model_comparison, model_comparison_cv, cv_table


def get_tables(
    OX: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.DataFrame,
    OX_test: pd.DataFrame,
    OX_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train: pd.DataFrame,
    X_2: pd.DataFrame,
    X_2_test: pd.DataFrame,
    X_2_train: pd.DataFrame,
    data_name: str,
    folder_name: str
) -> None:
    """
    Orchestrator function to evaluate multiple regression models, perform feature selection,
    and generate formatted outputs (LaTeX tables and plain text lists) for a given dataset.

    Args:
        OX (pd.DataFrame): The full input feature matrix (with intercept).
        X (pd.DataFrame): The input feature matrix (without intercept).
        y (pd.DataFrame): The target response vector.
        OX_test (pd.DataFrame): The testing input feature matrix (with intercept).
        OX_train (pd.DataFrame): The training input feature matrix (with intercept).
        X_test (pd.DataFrame): The testing input feature matrix (without intercept).
        X_train (pd.DataFrame): The training input feature matrix (without intercept).
        y_test (pd.DataFrame): The testing target response vector.
        y_train (pd.DataFrame): The training target response vector.
        X_2 (pd.DataFrame): The full quadratic/interaction feature matrix (Order 2).
        X_2_test (pd.DataFrame): The testing Order 2 feature matrix.
        X_2_train (pd.DataFrame): The training Order 2 feature matrix.
        data_name (str): The descriptive name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        None: Executes the entire modeling pipeline, prints tables, and saves plots.
    """
    # ==========================================
    # --- Data Formatting ---
    # ==========================================
    # Squeeze single-column DataFrames to Series for Statsmodels compatibility.
    # Use cast() to narrow the return type of squeeze() — which Pylance infers as
    # Scalar | DataFrame | Series[Any] — down to a concrete pd.Series, without
    # calling any constructor that might reject the union type.
    y_s: pd.Series = cast(pd.Series, y.squeeze())
    y_test_s: pd.Series = cast(pd.Series, y_test.squeeze())
    y_train_s: pd.Series = cast(pd.Series, y_train.squeeze())

    # Define configurations to gracefully handle varying names, functions, and inputs
    model_configs = [
        {"id": "reg",    "eval_name": "Regression",            "cv_name": "Regression",          "latex": "Linear Regression",           "fs_name": "Regression",       "func": lin_reg,    "X": OX, "X_test": OX_test, "X_train": OX_train},
        {"id": "ridge",  "eval_name": "Ridge",                 "cv_name": "Ridge",               "latex": "Ridge Regression",            "fs_name": "Ridge",            "func": ridge_reg,  "X": X, "X_test": X_test, "X_train": X_train},
        {"id": "lasso",  "eval_name": "Lasso",                 "cv_name": "Lasso",               "latex": "Lasso Regression",            "fs_name": "Lasso",            "func": lasso_reg,  "X": X, "X_test": X_test, "X_train": X_train},
        {"id": "sqrt",   "eval_name": "Sqrt",                  "cv_name": "Sqrt",                "latex": "Sqrt Transformation",         "fs_name": "Sqrt",             "func": sqrt_reg,   "X": OX, "X_test": OX_test, "X_train": OX_train},
        {"id": "log1p",  "eval_name": "Log1p",                 "cv_name": "Log1p",               "latex": "Log1p Transformation",        "fs_name": "Log1p",            "func": log1p_reg,  "X": OX, "X_test": OX_test, "X_train": OX_train},
        {"id": "boxcox", "eval_name": "Box-Cox",               "cv_name": "Box-Cox",             "latex": "Box-Cox Transformation",      "fs_name": "Box-Cox",          "func": boxcox_reg, "X": OX, "X_test": OX_test, "X_train": OX_train},
        {"id": "order2", "eval_name": "Order 2 Polynomial",    "cv_name": "Order 2 Polynomial",  "latex": "Order 2 Polynomial Regression","fs_name": "Order 2 Polynomial","func": order2_reg, "X": X_2, "X_test": X_2_test, "X_train": X_2_train},
        {"id": "nn_2L",  "eval_name": "2-Layer Neural Network","cv_name": "2L Neural Network",   "latex": "2L Neural Network",           "fs_name": "NN 2L",            "func": nn_2L,      "X": X, "X_test": X_test, "X_train": X_train},
        {"id": "nn_3L",  "eval_name": "3-Layer Neural Network","cv_name": "3L Neural Network",   "latex": "3L Neural Network",           "fs_name": "NN 3L",            "func": nn_3L,      "X": X, "X_test": X_test, "X_train": X_train},
        {"id": "nn_4L",  "eval_name": "4-Layer Neural Network","cv_name": "4L Neural Network",   "latex": "4L Neural Network",           "fs_name": "NN 4L",            "func": nn_4L,      "X": X, "X_test": X_test, "X_train": X_train},
    ]

    # ==========================================
    # --- Model Evaluation ---
    # ==========================================
    results = {}
    for cfg in model_configs:
        print("------------------------------------")
        print(f"{data_name} {cfg['eval_name']}")
        print("------------------------------------")

        args = (cfg["X"], y_s, cfg["X_test"], cfg["X_train"], y_test_s, y_train_s, data_name, folder_name)

        res = cfg['func'](*args)
        results[cfg['id']] = {
            'qof_is':   res[0],
            'qof_oos':  res[1],
            'cv_stats': res[2],
            'extra':    res[3:]
        }

    ridge_alpha     = results['ridge']['extra'][0]
    lasso_alpha     = results['lasso']['extra'][0]
    lasso_features  = results['lasso']['extra'][1]
    boxcox_lambda   = results['boxcox']['extra'][0]
    order2reg_alpha = results['order2']['extra'][0]

    # nn_2L extra: (best_activation_fn, best_output_activation_fn, best_lr)
    nn_2L_output_activation_fn = results['nn_2L']['extra'][0]
    nn_2L_lr                   = results['nn_2L']['extra'][1]

    # nn_3L extra: (best_activation_fn, best_output_activation_fn, best_hidden_1, best_lr)
    nn_3L_activation_fn        = results['nn_3L']['extra'][0]
    nn_3L_output_activation_fn = results['nn_3L']['extra'][1]
    nn_3L_hidden_1             = results['nn_3L']['extra'][2]
    nn_3L_lr                   = results['nn_3L']['extra'][3]

    # nn_4L extra: (best_activation_fn, best_activation_fn_2, best_output_activation_fn, best_hidden_1, best_hidden_2, best_lr)
    nn_4L_activation_fn        = results['nn_4L']['extra'][0]
    nn_4L_activation_fn_2      = results['nn_4L']['extra'][1]
    nn_4L_output_activation_fn = results['nn_4L']['extra'][2]
    nn_4L_hidden_1             = results['nn_4L']['extra'][3]
    nn_4L_hidden_2             = results['nn_4L']['extra'][4]
    nn_4L_lr                   = results['nn_4L']['extra'][5]

    # ==========================================
    # --- Feature Selection ---
    # ==========================================
    fs_methods = [
        ("Forward",   "Forward Selection"),
        ("Backward",  "Backward Elimination"),
        ("Stepwise",  "Stepwise Selection")
    ]

    fs_results = {}
    for method_key, method_name in fs_methods:
        print("------------------------------------")
        print(f"{data_name} {method_name}")
        print("------------------------------------")
        fs_results[method_key] = feature_selection(
            method_key, X, y_s, X_2, data_name, folder_name,
            ridge_alpha, lasso_alpha, boxcox_lambda, order2reg_alpha,
            nn_3L_hidden_1=nn_3L_hidden_1,
            nn_4L_hidden_1=nn_4L_hidden_1,
            nn_4L_hidden_2=nn_4L_hidden_2,
            nn_2L_output_activation_fn=nn_2L_output_activation_fn,
            nn_2L_lr=nn_2L_lr,
            nn_3L_activation_fn=nn_3L_activation_fn,
            nn_3L_output_activation_fn=nn_3L_output_activation_fn,
            nn_3L_lr=nn_3L_lr,
            nn_4L_activation_fn=nn_4L_activation_fn,
            nn_4L_activation_fn_2=nn_4L_activation_fn_2,
            nn_4L_output_activation_fn=nn_4L_output_activation_fn,
            nn_4L_lr=nn_4L_lr
        )

    # ==========================================
    # --- Cross-Validation Tables ---
    # ==========================================
    for cfg in model_configs:
        print("------------------------------------")
        print(f"{data_name} {cfg['cv_name']} CV Table")
        print("------------------------------------")

        if cfg['id'] == 'ridge':
            print(f"{data_name} Ridge alpha Used: {ridge_alpha}")
        elif cfg['id'] == 'lasso':
            print(f"{data_name} Lasso alpha Used: {lasso_alpha}")
            print(f"{data_name} Lasso Selected Features: {lasso_features}")
        elif cfg['id'] == 'boxcox':
            print(f"{data_name} Box-Cox lambda Used: {boxcox_lambda}")
        elif cfg['id'] == 'order2':
            print(f"{data_name} Order 2 Polynomial alpha Used: {order2reg_alpha}")
        elif cfg['id'] == 'nn_2L':
            print(f"{data_name} NN 2L output_activation: {type(nn_2L_output_activation_fn).__name__}, lr: {nn_2L_lr:.0e}")
        elif cfg['id'] == 'nn_3L':
            print(f"{data_name} NN 3L hidden_activation: {type(nn_3L_activation_fn).__name__}, output_activation: {type(nn_3L_output_activation_fn).__name__}, hidden_1: {nn_3L_hidden_1}, lr: {nn_3L_lr:.0e}")
        elif cfg['id'] == 'nn_4L':
            print(f"{data_name} NN 4L hidden_activation_1: {type(nn_4L_activation_fn).__name__}, hidden_activation_2: {type(nn_4L_activation_fn_2).__name__}, output_activation: {type(nn_4L_output_activation_fn).__name__}, hidden_1: {nn_4L_hidden_1}, hidden_2: {nn_4L_hidden_2}, lr: {nn_4L_lr:.0e}")

        cv_table(results[cfg['id']]['cv_stats'], data_name, cfg['latex'])

    # ==========================================
    # --- LaTeX Comparison Tables ---
    # ==========================================
    for cfg in model_configs:
        print("------------------------------------")
        print(f"{data_name} {cfg['cv_name']} Comparison Table")
        print("------------------------------------")
        is_oos_comparison(results[cfg['id']]['qof_is'], results[cfg['id']]['qof_oos'], data_name, cfg['latex'])

    # Unpack exactly 10 QoF lists for model_comparison() which expects 10 positional args
    ids = [cfg['id'] for cfg in model_configs]
    qof_is_all  = [results[i]['qof_is']  for i in ids]
    qof_oos_all = [results[i]['qof_oos'] for i in ids]

    print("------------------------------------")
    print(f"{data_name} Model In-Sample Comparison Table")
    print("------------------------------------")
    model_comparison(
        qof_is_all[0], qof_is_all[1], qof_is_all[2], qof_is_all[3], qof_is_all[4],
        qof_is_all[5], qof_is_all[6], qof_is_all[7], qof_is_all[8], qof_is_all[9],
        data_name, False
    )

    print("------------------------------------")
    print(f"{data_name} Model Out-of-Sample Comparison Table")
    print("------------------------------------")
    model_comparison(
        qof_oos_all[0], qof_oos_all[1], qof_oos_all[2], qof_oos_all[3], qof_oos_all[4],
        qof_oos_all[5], qof_oos_all[6], qof_oos_all[7], qof_oos_all[8], qof_oos_all[9],
        data_name, True
    )

    print("------------------------------------")
    print(f"{data_name} Model Cross-Validation Comparison Table")
    print("------------------------------------")
    model_comparison_cv(
        results['reg']['cv_stats'], results['ridge']['cv_stats'], results['lasso']['cv_stats'],
        results['sqrt']['cv_stats'], results['log1p']['cv_stats']   , results['boxcox']['cv_stats'],
        results['order2']['cv_stats'],
        results['nn_2L']['cv_stats'], results['nn_3L']['cv_stats'], results['nn_4L']['cv_stats'],
        data_name
    )

    # ==========================================
    # --- Feature Selection Results Printing ---
    # ==========================================
    fs_print_suffixes = [
        ("Forward",   "Forward Selection Order"),
        ("Backward",  "Backward Elimination Reversed Order"),
        ("Stepwise",  "Stepwise Selection Order")
    ]

    for i, cfg in enumerate(model_configs):
        for fs_key, print_suffix in fs_print_suffixes:
            print("------------------------------------")
            print(f"{data_name} {cfg['fs_name']} {print_suffix}")
            print("------------------------------------")
            print(fs_results[fs_key][i])
