from typing import List, cast
import pandas as pd
import numpy as np
import statsmodels.api as sm
import torch
import torch.nn as nn
import os
import contextlib
from hyperparameter_tuning import tune_ridge_lasso_alpha
from neural_network_classes import OneHiddenLayerNN, TwoHiddenLayerNN
from latex_tables import cv_table, is_oos_comparison
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_selection_methods import stepwise_selection
from get_cv_qof import get_cv_qof
from get_qof import get_qof
from main import get_tables, get_tables_2
from model_evaluation import order2_reg, sqrt_reg
from save_plots import save_sorted_plot

def tp_insurance():
    """
    Main entry point for evaluating regression models on the Medical Insurance dataset.
    
    * Loads data.
    * Calls the `get_tables` orchestrator to generate metrics, plots, and LaTeX summaries.
    """
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    OXy = pd.read_csv("Insurance_Charges/cleaned_insurance_with_intercept.csv")
    OX = OXy.drop('charges', axis=1)
    y = OXy[['charges']]

    OXY_2 = pd.read_csv("Insurance_Charges/cleaned_order_2_insurance_with_intercept.csv")
    OX_2 = OXY_2.drop('charges', axis=1)

    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    OX_train, OX_test, OX_2_train, OX_2_test, y_train, y_test = train_test_split(OX, OX_2, y, test_size=0.2, random_state=0)

    X = OX.drop('intercept', axis=1)
    X_train = OX_train.drop('intercept', axis=1)
    X_test = OX_test.drop('intercept', axis=1)

    X_2 = OX_2.drop('intercept', axis=1)
    X_2_train = OX_2_train.drop('intercept', axis=1)
    X_2_test = OX_2_test.drop('intercept', axis=1)

    # ==========================================
    # --- Run Pipeline ---
    # ==========================================
    get_tables(OX, X, y, OX_test, OX_train, X_test, X_train, y_test, y_train, X_2, X_2_test, X_2_train, "Insurance Charges", "Insurance_Charges_Plots")

    print("------------------------------------")
    print("Finished")
    print("------------------------------------")








def tp_big_insurance():
    """
    Main entry point for evaluating regression models on the Big Medical Insurance dataset.
    
    * Loads data.
    * Calls the `get_tables` orchestrator to generate metrics, plots, and LaTeX summaries.
    """
    print("------------------------------------")
    print("Starting")
    print("------------------------------------")

    # ==========================================
    # --- Data Loading ---
    # ==========================================
    OXy = pd.read_csv("Insurance_Annual_Medical_Cost/cleaned_insurance_with_intercept.csv")
    OX = OXy.drop('annual_medical_cost', axis=1)
    y = OXy[['annual_medical_cost']]
 
    OXY_2 = pd.read_csv("Insurance_Annual_Medical_Cost/cleaned_order_2_insurance_with_intercept.csv")
    OX_2 = OXY_2.drop('annual_medical_cost', axis=1)
 
    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    OX_train, OX_test, OX_2_train, OX_2_test, y_train, y_test = train_test_split(
        OX, OX_2, y, test_size=0.2, random_state=0
    )
 
    X = OX.drop('intercept', axis=1)
    X_train = OX_train.drop('intercept', axis=1)
    X_test  = OX_test.drop('intercept', axis=1)
 
    X_2       = OX_2.drop('intercept', axis=1)
    X_2_train = OX_2_train.drop('intercept', axis=1)
    X_2_test  = OX_2_test.drop('intercept', axis=1)
 
    # ==========================================
    # --- Run Pipeline ---
    # ==========================================
    # get_tables_2(OX, X, y, OX_test, OX_train, X_test, X_train, y_test, y_train, X_2, X_2_test, X_2_train, "Big Insurance Charges", "Big_Insurance_Charges_Plots")

    # is_o2_qof, oos_o2_qof, cv_stats_o2, o2_alpha = order2_reg(X_2, y, X_2_test, X_2_train, y_test, y_train, "Insurance Annual Medical Cost", "Insurance_Annual_Medical_Cost_Plots")

    # o2_alpha, _ = tune_ridge_lasso_alpha(X_2, y, 'ridge')
    o2_alpha = 0.0015

    y_np = np.asarray(y, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float).ravel()

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    X_2_scaler_oos = StandardScaler()
    y_scaler_oos = StandardScaler()
    
    X_2_train_scaled = X_2_scaler_oos.fit_transform(X_2_train)
    X_2_test_scaled = X_2_scaler_oos.transform(X_2_test)
    y_train_scaled = y_scaler_oos.fit_transform(y_train_np.reshape(-1, 1)).flatten()

    o2_oos = sm.OLS(y_train_scaled, X_2_train_scaled).fit_regularized(alpha=o2_alpha, L1_wt=0.0)
    yp_oos_scaled_02: np.ndarray = np.asarray(o2_oos.predict(X_2_test_scaled), dtype=float)
    
    yp_oos_o2 = y_scaler_oos.inverse_transform(yp_oos_scaled_02.reshape(-1, 1)).flatten()
    k = X_2_train_scaled.shape[1]
    
    qof_oos = get_qof(y_test_np, yp_oos_o2, k)
    save_sorted_plot(y_test_np, yp_oos_o2, "Insurance Annual Medical Cost", "Insurance_Annual_Medical_Cost_Plots", "Order 2 Regression", "Order2Reg", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")

    cv_stats_o2 = get_cv_qof(X_2, y, 'ridge', alpha=o2_alpha)

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
            input_size=X_train_scaled.shape[1],
            hidden_size_1=128,
            hidden_size_2=64,
            output_size=1,
            hidden_activation_fn_1=nn.ELU(),
            hidden_activation_fn_2=nn.ELU(),
            output_activation_fn=nn.Identity()
        )
        model_oos.lr = 0.005
        model_oos.fit(X_train_tensor, y_train_tensor)
        yp_oos_tensor, _ = model_oos.predict(X_test_tensor, y_test_tensor)

    yp_oos_scaled = yp_oos_tensor.detach().cpu().numpy()
    yp_oos = y_scaler_oos.inverse_transform(yp_oos_scaled).flatten()
    k = X_train_scaled.shape[1]

    qof_oos = get_qof(y_test_np, yp_oos, k)
    save_sorted_plot(y_test_np, yp_oos, "Insurance Annual Medical Cost", "Insurance_Annual_Medical_Cost_Plots", "4L Neural Network", "NN_4L", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")

    cv_stats_nn_4L = get_cv_qof(X, y, 'nn_4L', nn_hidden_1=128, nn_hidden_2=64, activation_fn=nn.ELU(), activation_fn_2=nn.ELU(), output_activation_fn=nn.Identity(), lr=0.005)

    # is_oos_comparison(is_o2_qof, oos_o2_qof, "Insurance Annual Medical Cost", "Order 2 Regression")

    cv_table(cv_stats_o2, "Insurance Annual Medical Cost", "Order 2 Regression")

    cv_table(cv_stats_nn_4L, "Insurance Annual Medical Cost", "4 Layer Neural Network")
 
    print("------------------------------------")
    print("Finished")
    print("------------------------------------")


# tp_insurance()

tp_big_insurance()