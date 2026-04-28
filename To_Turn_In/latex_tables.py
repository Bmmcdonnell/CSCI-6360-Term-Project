import numpy as np
from typing import List

def is_oos_comparison(
    is_qof: List[float], 
    oos_qof: List[float], 
    data_name: str, 
    model_name: str
) -> None:
    """
    Generates a LaTeX-formatted table comparing In-Sample and Out-of-Sample 
    (80-20 Split) Quality of Fit (QoF) metrics for a single regression model.

    Args:
        is_qof (List[float]): A 15-element list of QoF metrics evaluated on the In-Sample dataset.
        oos_qof (List[float]): A 15-element list of QoF metrics evaluated on the Out-of-Sample dataset.
        data_name (str): The name of the dataset, used in the table caption and label.
        model_name (str): The name of the regression model, used in the table caption and label.

    Returns:
        None: Outputs the raw LaTeX table string directly to standard output (stdout).
    """
    # ==========================================
    # --- Table Header Configuration ---
    # ==========================================
    # Initialize a floating table environment with centered alignment
    print("\\begin{table}[H]")
    print("\\centering")
    print(f"\\caption{{{data_name} {model_name}}}")
    print(f"\\label{{tab:{data_name} {model_name}}}")
    
    # Define a 3-column structure with vertical borders
    print("\\begin{tabular}{|c|c|c|}\\hline")
    print("Metric & In-Sample & 80-20 Split \\\\ \\hline \\hline")
    
    # ==========================================
    # --- Table Data Rows ---
    # ==========================================
    # Hardcoded row prints for each specific metric index (0 through 14)
    # Values are formatted to 4 decimal places for uniform precision
    print(f"rSq & {is_qof[0]:.4g} & {oos_qof[0]:.4g} \\\\ \\hline")
    print(f"rSqBar & {is_qof[1]:.4g} & {oos_qof[1]:.4g} \\\\ \\hline")
    print(f"sst & {is_qof[2]:.4g} & {oos_qof[2]:.4g} \\\\ \\hline")
    print(f"sse & {is_qof[3]:.4g} & {oos_qof[3]:.4g} \\\\ \\hline")
    print(f"sde & {is_qof[4]:.4g} & {oos_qof[4]:.4g} \\\\ \\hline")
    print(f"mse0 & {is_qof[5]:.4g} & {oos_qof[5]:.4g} \\\\ \\hline")
    print(f"rmse & {is_qof[6]:.4g} & {oos_qof[6]:.4g} \\\\ \\hline")
    print(f"mae & {is_qof[7]:.4g} & {oos_qof[7]:.4g} \\\\ \\hline")
    print(f"smape & {is_qof[8]:.4g} & {oos_qof[8]:.4g} \\\\ \\hline")
    print(f"m & {is_qof[9]:.4g} & {oos_qof[9]:.4g} \\\\ \\hline")
    print(f"dfr & {is_qof[10]:.4g} & {oos_qof[10]:.4g} \\\\ \\hline")
    print(f"df & {is_qof[11]:.4g} & {oos_qof[11]:.4g} \\\\ \\hline")
    print(f"fStat & {is_qof[12]:.4g} & {oos_qof[12]:.4g} \\\\ \\hline")
    print(f"aic & {is_qof[13]:.4g} & {oos_qof[13]:.4g} \\\\ \\hline")
    print(f"bic & {is_qof[14]:.4g} & {oos_qof[14]:.4g} \\\\ \\hline")
    
    # Close the LaTeX environments
    print("\\end{tabular}")
    print("\\end{table}")


def model_comparison(
    reg_qof: List[float], 
    ridge_qof: List[float], 
    lasso_qof: List[float], 
    sqrt_qof: List[float], 
    log1p_qof: List[float], 
    boxcox_qof: List[float], 
    order2reg_qof: List[float], 
    nn_2L_qof: List[float], 
    nn_3L_qof: List[float], 
    nn_4L_qof: List[float], 
    data_name: str, 
    validate: bool
) -> None:
    """
    Generates a LaTeX-formatted table comparing the Quality of Fit (QoF) metrics 
    across ten evaluated regression models for a specific data split.

    Args:
        reg_qof (List[float]): QoF metrics for standard Linear Regression.
        ridge_qof (List[float]): QoF metrics for Ridge Regression.
        lasso_qof (List[float]): QoF metrics for Lasso Regression.
        sqrt_qof (List[float]): QoF metrics for Square Root Transformed Regression.
        log1p_qof (List[float]): QoF metrics for Log1p Transformed Regression.
        boxcox_qof (List[float]): QoF metrics for Box-Cox Transformed Regression.
        order2reg_qof (List[float]): QoF metrics for Order-2 Polynomial Regression.
        nn_2L_qof (List[float]): QoF metrics for 2-Layer Neural Network.
        nn_3L_qof (List[float]): QoF metrics for 3-Layer Neural Network.
        nn_4L_qof (List[float]): QoF metrics for 4-Layer Neural Network.
        data_name (str): The name of the dataset.
        validate (bool): Flag indicating if the metrics evaluate Out-of-Sample (True) 
            or In-Sample (False) performance.

    Returns:
        None: Outputs the raw LaTeX table string directly to standard output (stdout).
    """
    # Determine the context string for the table caption based on validation state
    method = "Out-of-Sample" if validate else "In-Sample"

    # ==========================================
    # --- Table Header Configuration ---
    # ==========================================
    print("\\begin{table}[H]")
    print("\\centering")
    
    # Scale down the table to fit the text width, preventing the 11 columns from overflowing page margins
    print(f"\\caption{{{data_name} {method} QoF Comparison}}")
    print(f"\\label{{tab:{data_name} {method} QoF Comparison}}")
    print("\\resizebox{\\textwidth}{!}{")
    
    # Define an 11-column structure with vertical borders
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}\\hline")
    print("Metric & Reg & Ridge & Lasso & Sqrt & Log1p & Box-Cox & Order 2 & NN 2L & NN 3L & NN 4L \\\\ \\hline \\hline")
    
    # ==========================================
    # --- Table Data Rows ---
    # ==========================================
    # Sequentially print each metric across all 10 models side-by-side
    print(f"rSq & {reg_qof[0]:.4g} & {ridge_qof[0]:.4g} & {lasso_qof[0]:.4g} & {sqrt_qof[0]:.4g} & {log1p_qof[0]:.4g} & {boxcox_qof[0]:.4g} & {order2reg_qof[0]:.4g} & {nn_2L_qof[0]:.4g} & {nn_3L_qof[0]:.4g} & {nn_4L_qof[0]:.4g} \\\\ \\hline")
    print(f"rSqBar & {reg_qof[1]:.4g} & {ridge_qof[1]:.4g} & {lasso_qof[1]:.4g} & {sqrt_qof[1]:.4g} & {log1p_qof[1]:.4g} & {boxcox_qof[1]:.4g} & {order2reg_qof[1]:.4g} & {nn_2L_qof[1]:.4g} & {nn_3L_qof[1]:.4g} & {nn_4L_qof[1]:.4g} \\\\ \\hline")
    print(f"sst & {reg_qof[2]:.4g} & {ridge_qof[2]:.4g} & {lasso_qof[2]:.4g} & {sqrt_qof[2]:.4g} & {log1p_qof[2]:.4g} & {boxcox_qof[2]:.4g} & {order2reg_qof[2]:.4g} & {nn_2L_qof[2]:.4g} & {nn_3L_qof[2]:.4g} & {nn_4L_qof[2]:.4g} \\\\ \\hline")
    print(f"sse & {reg_qof[3]:.4g} & {ridge_qof[3]:.4g} & {lasso_qof[3]:.4g} & {sqrt_qof[3]:.4g} & {log1p_qof[3]:.4g} & {boxcox_qof[3]:.4g} & {order2reg_qof[3]:.4g} & {nn_2L_qof[3]:.4g} & {nn_3L_qof[3]:.4g} & {nn_4L_qof[3]:.4g} \\\\ \\hline")
    print(f"sde & {reg_qof[4]:.4g} & {ridge_qof[4]:.4g} & {lasso_qof[4]:.4g} & {sqrt_qof[4]:.4g} & {log1p_qof[4]:.4g} & {boxcox_qof[4]:.4g} & {order2reg_qof[4]:.4g} & {nn_2L_qof[4]:.4g} & {nn_3L_qof[4]:.4g} & {nn_4L_qof[4]:.4g} \\\\ \\hline")
    print(f"mse0 & {reg_qof[5]:.4g} & {ridge_qof[5]:.4g} & {lasso_qof[5]:.4g} & {sqrt_qof[5]:.4g} & {log1p_qof[5]:.4g} & {boxcox_qof[5]:.4g} & {order2reg_qof[5]:.4g} & {nn_2L_qof[5]:.4g} & {nn_3L_qof[5]:.4g} & {nn_4L_qof[5]:.4g} \\\\ \\hline")
    print(f"rmse & {reg_qof[6]:.4g} & {ridge_qof[6]:.4g} & {lasso_qof[6]:.4g} & {sqrt_qof[6]:.4g} & {log1p_qof[6]:.4g} & {boxcox_qof[6]:.4g} & {order2reg_qof[6]:.4g} & {nn_2L_qof[6]:.4g} & {nn_3L_qof[6]:.4g} & {nn_4L_qof[6]:.4g} \\\\ \\hline")
    print(f"mae & {reg_qof[7]:.4g} & {ridge_qof[7]:.4g} & {lasso_qof[7]:.4g} & {sqrt_qof[7]:.4g} & {log1p_qof[7]:.4g} & {boxcox_qof[7]:.4g} & {order2reg_qof[7]:.4g} & {nn_2L_qof[7]:.4g} & {nn_3L_qof[7]:.4g} & {nn_4L_qof[7]:.4g} \\\\ \\hline")
    print(f"smape & {reg_qof[8]:.4g} & {ridge_qof[8]:.4g} & {lasso_qof[8]:.4g} & {sqrt_qof[8]:.4g} & {log1p_qof[8]:.4g} & {boxcox_qof[8]:.4g} & {order2reg_qof[8]:.4g} & {nn_2L_qof[8]:.4g} & {nn_3L_qof[8]:.4g} & {nn_4L_qof[8]:.4g} \\\\ \\hline")
    print(f"m & {reg_qof[9]:.4g} & {ridge_qof[9]:.4g} & {lasso_qof[9]:.4g} & {sqrt_qof[9]:.4g} & {log1p_qof[9]:.4g} & {boxcox_qof[9]:.4g} & {order2reg_qof[9]:.4g} & {nn_2L_qof[9]:.4g} & {nn_3L_qof[9]:.4g} & {nn_4L_qof[9]:.4g} \\\\ \\hline")
    print(f"dfr & {reg_qof[10]:.4g} & {ridge_qof[10]:.4g} & {lasso_qof[10]:.4g} & {sqrt_qof[10]:.4g} & {log1p_qof[10]:.4g} & {boxcox_qof[10]:.4g} & {order2reg_qof[10]:.4g} & {nn_2L_qof[10]:.4g} & {nn_3L_qof[10]:.4g} & {nn_4L_qof[10]:.4g} \\\\ \\hline")
    print(f"df & {reg_qof[11]:.4g} & {ridge_qof[11]:.4g} & {lasso_qof[11]:.4g} & {sqrt_qof[11]:.4g} & {log1p_qof[11]:.4g} & {boxcox_qof[11]:.4g} & {order2reg_qof[11]:.4g} & {nn_2L_qof[11]:.4g} & {nn_3L_qof[11]:.4g} & {nn_4L_qof[11]:.4g} \\\\ \\hline")
    print(f"fStat & {reg_qof[12]:.4g} & {ridge_qof[12]:.4g} & {lasso_qof[12]:.4g} & {sqrt_qof[12]:.4g} & {log1p_qof[12]:.4g} & {boxcox_qof[12]:.4g} & {order2reg_qof[12]:.4g} & {nn_2L_qof[12]:.4g} & {nn_3L_qof[12]:.4g} & {nn_4L_qof[12]:.4g} \\\\ \\hline")
    print(f"aic & {reg_qof[13]:.4g} & {ridge_qof[13]:.4g} & {lasso_qof[13]:.4g} & {sqrt_qof[13]:.4g} & {log1p_qof[13]:.4g} & {boxcox_qof[13]:.4g} & {order2reg_qof[13]:.4g} & {nn_2L_qof[13]:.4g} & {nn_3L_qof[13]:.4g} & {nn_4L_qof[13]:.4g} \\\\ \\hline")
    print(f"bic & {reg_qof[14]:.4g} & {ridge_qof[14]:.4g} & {lasso_qof[14]:.4g} & {sqrt_qof[14]:.4g} & {log1p_qof[14]:.4g} & {boxcox_qof[14]:.4g} & {order2reg_qof[14]:.4g} & {nn_2L_qof[14]:.4g} & {nn_3L_qof[14]:.4g} & {nn_4L_qof[14]:.4g} \\\\ \\hline")
    
    # Close LaTeX environments
    print("\\end{tabular}")
    print("}") # End resizebox scope
    print("\\end{table}")


def model_comparison_cv(
    reg_cv_stats: List[List[float]],
    ridge_cv_stats: List[List[float]],
    lasso_cv_stats: List[List[float]],
    sqrt_cv_stats: List[List[float]],
    log1p_cv_stats: List[List[float]],
    boxcox_cv_stats: List[List[float]],
    order2reg_cv_stats: List[List[float]],
    nn_2L_cv_stats: List[List[float]],
    nn_3L_cv_stats: List[List[float]],
    nn_4L_cv_stats: List[List[float]],
    data_name: str
) -> None:
    """
    Generates a LaTeX-formatted table comparing the Cross Validation Quality of Fit (QoF) metrics 
    across ten evaluated regression models for a specific data split.

    Args:
        reg_qof (List[List[float]]): CV QoF metrics for standard Linear Regression.
        ridge_qof (List[List[float]]): CV QoF metrics for Ridge Regression.
        lasso_qof (List[List[float]]): CV QoF metrics for Lasso Regression.
        sqrt_qof (List[List[float]]): CV QoF metrics for Square Root Transformed Regression.
        log1p_qof (List[List[float]]): CV QoF metrics for Log1p Transformed Regression.
        boxcox_qof (List[List[float]]): CV QoF metrics for Box-Cox Transformed Regression.
        order2reg_qof (List[List[float]]): CV QoF metrics for Order-2 Polynomial Regression.
        nn_2L_qof (List[List[float]]): CV QoF metrics for 2-Layer Neural Network.
        nn_3L_qof (List[List[float]]): CV QoF metrics for 3-Layer Neural Network.
        nn_4L_qof (List[List[float]]): CV QoF metrics for 4-Layer Neural Network.
        data_name (str): The name of the dataset.
        validate (bool): Flag indicating if the metrics evaluate Out-of-Sample (True) 
            or In-Sample (False) performance.

    Returns:
        None: Outputs the raw LaTeX table string directly to standard output (stdout).
    """
    # Determine the context string for the table caption based on validation state
    method = "Cross-Validation"

    # ==========================================
    # --- Table Header Configuration ---
    # ==========================================
    print("\\begin{table}[H]")
    print("\\centering")
    
    # Scale down the table to fit the text width, preventing the 11 columns from overflowing page margins
    print(f"\\caption{{{data_name} {method} QoF Comparison}}")
    print(f"\\label{{tab:{data_name} {method} QoF Comparison}}")
    print("\\resizebox{\\textwidth}{!}{")
    
    # Define an 11-column structure with vertical borders
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}\\hline")
    print("Metric & Reg & Ridge & Lasso & Sqrt & Log1p & Box-Cox & Order 2 & NN 2L & NN 3L & NN 4L \\\\ \\hline \\hline")
    
    # ==========================================
    # --- Table Data Rows ---
    # ==========================================
    # Sequentially print each metric across all 10 models side-by-side
    print(f"rSq & {np.mean(reg_cv_stats[0]):.4g} & {np.mean(ridge_cv_stats[0]):.4g} & {np.mean(lasso_cv_stats[0]):.4g} & {np.mean(sqrt_cv_stats[0]):.4g} & {np.mean(log1p_cv_stats[0]):.4g} & {np.mean(boxcox_cv_stats[0]):.4g} & {np.mean(order2reg_cv_stats[0]):.4g} & {np.mean(nn_2L_cv_stats[0]):.4g} & {np.mean(nn_3L_cv_stats[0]):.4g} & {np.mean(nn_4L_cv_stats[0]):.4g} \\\\ \\hline")
    print(f"rSqBar & {np.mean(reg_cv_stats[1]):.4g} & {np.mean(ridge_cv_stats[1]):.4g} & {np.mean(lasso_cv_stats[1]):.4g} & {np.mean(sqrt_cv_stats[1]):.4g} & {np.mean(log1p_cv_stats[1]):.4g} & {np.mean(boxcox_cv_stats[1]):.4g} & {np.mean(order2reg_cv_stats[1]):.4g} & {np.mean(nn_2L_cv_stats[1]):.4g} & {np.mean(nn_3L_cv_stats[1]):.4g} & {np.mean(nn_4L_cv_stats[1]):.4g} \\\\ \\hline")
    print(f"sst & {np.mean(reg_cv_stats[2]):.4g} & {np.mean(ridge_cv_stats[2]):.4g} & {np.mean(lasso_cv_stats[2]):.4g} & {np.mean(sqrt_cv_stats[2]):.4g} & {np.mean(log1p_cv_stats[2]):.4g} & {np.mean(boxcox_cv_stats[2]):.4g} & {np.mean(order2reg_cv_stats[2]):.4g} & {np.mean(nn_2L_cv_stats[2]):.4g} & {np.mean(nn_3L_cv_stats[2]):.4g} & {np.mean(nn_4L_cv_stats[2]):.4g} \\\\ \\hline")
    print(f"sse & {np.mean(reg_cv_stats[3]):.4g} & {np.mean(ridge_cv_stats[3]):.4g} & {np.mean(lasso_cv_stats[3]):.4g} & {np.mean(sqrt_cv_stats[3]):.4g} & {np.mean(log1p_cv_stats[3]):.4g} & {np.mean(boxcox_cv_stats[3]):.4g} & {np.mean(order2reg_cv_stats[3]):.4g} & {np.mean(nn_2L_cv_stats[3]):.4g} & {np.mean(nn_3L_cv_stats[3]):.4g} & {np.mean(nn_4L_cv_stats[3]):.4g} \\\\ \\hline")
    print(f"sde & {np.mean(reg_cv_stats[4]):.4g} & {np.mean(ridge_cv_stats[4]):.4g} & {np.mean(lasso_cv_stats[4]):.4g} & {np.mean(sqrt_cv_stats[4]):.4g} & {np.mean(log1p_cv_stats[4]):.4g} & {np.mean(boxcox_cv_stats[4]):.4g} & {np.mean(order2reg_cv_stats[4]):.4g} & {np.mean(nn_2L_cv_stats[4]):.4g} & {np.mean(nn_3L_cv_stats[4]):.4g} & {np.mean(nn_4L_cv_stats[4]):.4g} \\\\ \\hline")
    print(f"mse0 & {np.mean(reg_cv_stats[5]):.4g} & {np.mean(ridge_cv_stats[5]):.4g} & {np.mean(lasso_cv_stats[5]):.4g} & {np.mean(sqrt_cv_stats[5]):.4g} & {np.mean(log1p_cv_stats[5]):.4g} & {np.mean(boxcox_cv_stats[5]):.4g} & {np.mean(order2reg_cv_stats[5]):.4g} & {np.mean(nn_2L_cv_stats[5]):.4g} & {np.mean(nn_3L_cv_stats[5]):.4g} & {np.mean(nn_4L_cv_stats[5]):.4g} \\\\ \\hline")
    print(f"rmse & {np.mean(reg_cv_stats[6]):.4g} & {np.mean(ridge_cv_stats[6]):.4g} & {np.mean(lasso_cv_stats[6]):.4g} & {np.mean(sqrt_cv_stats[6]):.4g} & {np.mean(log1p_cv_stats[6]):.4g} & {np.mean(boxcox_cv_stats[6]):.4g} & {np.mean(order2reg_cv_stats[6]):.4g} & {np.mean(nn_2L_cv_stats[6]):.4g} & {np.mean(nn_3L_cv_stats[6]):.4g} & {np.mean(nn_4L_cv_stats[6]):.4g} \\\\ \\hline")
    print(f"mae & {np.mean(reg_cv_stats[7]):.4g} & {np.mean(ridge_cv_stats[7]):.4g} & {np.mean(lasso_cv_stats[7]):.4g} & {np.mean(sqrt_cv_stats[7]):.4g} & {np.mean(log1p_cv_stats[7]):.4g} & {np.mean(boxcox_cv_stats[7]):.4g} & {np.mean(order2reg_cv_stats[7]):.4g} & {np.mean(nn_2L_cv_stats[7]):.4g} & {np.mean(nn_3L_cv_stats[7]):.4g} & {np.mean(nn_4L_cv_stats[7]):.4g} \\\\ \\hline")
    print(f"smape & {np.mean(reg_cv_stats[8]):.4g} & {np.mean(ridge_cv_stats[8]):.4g} & {np.mean(lasso_cv_stats[8]):.4g} & {np.mean(sqrt_cv_stats[8]):.4g} & {np.mean(log1p_cv_stats[8]):.4g} & {np.mean(boxcox_cv_stats[8]):.4g} & {np.mean(order2reg_cv_stats[8]):.4g} & {np.mean(nn_2L_cv_stats[8]):.4g} & {np.mean(nn_3L_cv_stats[8]):.4g} & {np.mean(nn_4L_cv_stats[8]):.4g} \\\\ \\hline")
    print(f"m & {np.mean(reg_cv_stats[9]):.4g} & {np.mean(ridge_cv_stats[9]):.4g} & {np.mean(lasso_cv_stats[9]):.4g} & {np.mean(sqrt_cv_stats[9]):.4g} & {np.mean(log1p_cv_stats[9]):.4g} & {np.mean(boxcox_cv_stats[9]):.4g} & {np.mean(order2reg_cv_stats[9]):.4g} & {np.mean(nn_2L_cv_stats[9]):.4g} & {np.mean(nn_3L_cv_stats[9]):.4g} & {np.mean(nn_4L_cv_stats[9]):.4g} \\\\ \\hline")
    print(f"dfr & {np.mean(reg_cv_stats[10]):.4g} & {np.mean(ridge_cv_stats[10]):.4g} & {np.mean(lasso_cv_stats[10]):.4g} & {np.mean(sqrt_cv_stats[10]):.4g} & {np.mean(log1p_cv_stats[10]):.4g} & {np.mean(boxcox_cv_stats[10]):.4g} & {np.mean(order2reg_cv_stats[10]):.4g} & {np.mean(nn_2L_cv_stats[10]):.4g} & {np.mean(nn_3L_cv_stats[10]):.4g} & {np.mean(nn_4L_cv_stats[10]):.4g} \\\\ \\hline")
    print(f"df & {np.mean(reg_cv_stats[11]):.4g} & {np.mean(ridge_cv_stats[11]):.4g} & {np.mean(lasso_cv_stats[11]):.4g} & {np.mean(sqrt_cv_stats[11]):.4g} & {np.mean(log1p_cv_stats[11]):.4g} & {np.mean(boxcox_cv_stats[11]):.4g} & {np.mean(order2reg_cv_stats[11]):.4g} & {np.mean(nn_2L_cv_stats[11]):.4g} & {np.mean(nn_3L_cv_stats[11]):.4g} & {np.mean(nn_4L_cv_stats[11]):.4g} \\\\ \\hline")
    print(f"fStat & {np.mean(reg_cv_stats[12]):.4g} & {np.mean(ridge_cv_stats[12]):.4g} & {np.mean(lasso_cv_stats[12]):.4g} & {np.mean(sqrt_cv_stats[12]):.4g} & {np.mean(log1p_cv_stats[12]):.4g} & {np.mean(boxcox_cv_stats[12]):.4g} & {np.mean(order2reg_cv_stats[12]):.4g} & {np.mean(nn_2L_cv_stats[12]):.4g} & {np.mean(nn_3L_cv_stats[12]):.4g} & {np.mean(nn_4L_cv_stats[12]):.4g} \\\\ \\hline")
    print(f"aic & {np.mean(reg_cv_stats[13]):.4g} & {np.mean(ridge_cv_stats[13]):.4g} & {np.mean(lasso_cv_stats[13]):.4g} & {np.mean(sqrt_cv_stats[13]):.4g} & {np.mean(log1p_cv_stats[13]):.4g} & {np.mean(boxcox_cv_stats[13]):.4g} & {np.mean(order2reg_cv_stats[13]):.4g} & {np.mean(nn_2L_cv_stats[13]):.4g} & {np.mean(nn_3L_cv_stats[13]):.4g} & {np.mean(nn_4L_cv_stats[13]):.4g} \\\\ \\hline")
    print(f"bic & {np.mean(reg_cv_stats[14]):.4g} & {np.mean(ridge_cv_stats[14]):.4g} & {np.mean(lasso_cv_stats[14]):.4g} & {np.mean(sqrt_cv_stats[14]):.4g} & {np.mean(log1p_cv_stats[14]):.4g} & {np.mean(boxcox_cv_stats[14]):.4g} & {np.mean(order2reg_cv_stats[14]):.4g} & {np.mean(nn_2L_cv_stats[14]):.4g} & {np.mean(nn_3L_cv_stats[14]):.4g} & {np.mean(nn_4L_cv_stats[14]):.4g} \\\\ \\hline")
    
    # Close LaTeX environments
    print("\\end{tabular}")
    print("}") # End resizebox scope
    print("\\end{table}")


def cv_table(
    cv_stats: List[List[float]],
    data_name: str, 
    model_name: str
) -> None:
    """
    Generates a LaTeX-formatted table summarizing Cross-Validation (CV) statistics.
    
    Calculates and displays the minimum, maximum, mean, and standard deviation 
    for each of the 15 Quality of Fit (QoF) metrics across all evaluated CV folds.

    Args:
        cv_stats (List[List[float]]): A list containing 15 sub-lists, where each 
            sub-list holds the values of a specific QoF metric across all CV folds.
        data_name (str): The name of the dataset.
        model_name (str): The name of the regression model.

    Returns:
        None: Outputs the raw LaTeX table string directly to standard output (stdout).
    """
    # ==========================================
    # --- Table Header Configuration ---
    # ==========================================
    print("\\begin{table}[H]")
    print("\\centering")
    print(f"\\caption{{{data_name} {model_name} CV}}")
    print(f"\\label{{tab:{data_name} {model_name} CV}}")
    
    # Define a 6-column structure with vertical borders
    print("\\begin{tabular}{|c|c|c|c|c|c|}\\hline")
    print("Metric & num folds & min & max & mean & stdev \\\\ \\hline \\hline")
    
    # ==========================================
    # --- Table Data Rows ---
    # ==========================================
    # len(cv_stats[X]) calculates the number of folds dynamically
    # np.mean and np.std are computed on the fly for the aggregate columns
    print(f"rSq & {len(cv_stats[0])} & {min(cv_stats[0]):.4g} & {max(cv_stats[0]):.4g} & {np.mean(cv_stats[0]):.4g} & {np.std(cv_stats[0]):.4g} \\\\ \\hline")
    print(f"rSqBar & {len(cv_stats[1])} & {min(cv_stats[1]):.4g} & {max(cv_stats[1]):.4g} & {np.mean(cv_stats[1]):.4g} & {np.std(cv_stats[1]):.4g} \\\\ \\hline")
    print(f"sst & {len(cv_stats[2])} & {min(cv_stats[2]):.4g} & {max(cv_stats[2]):.4g} & {np.mean(cv_stats[2]):.4g} & {np.std(cv_stats[2]):.4g} \\\\ \\hline")
    print(f"sse & {len(cv_stats[3])} & {min(cv_stats[3]):.4g} & {max(cv_stats[3]):.4g} & {np.mean(cv_stats[3]):.4g} & {np.std(cv_stats[3]):.4g} \\\\ \\hline")
    print(f"sde & {len(cv_stats[4])} & {min(cv_stats[4]):.4g} & {max(cv_stats[4]):.4g} & {np.mean(cv_stats[4]):.4g} & {np.std(cv_stats[4]):.4g} \\\\ \\hline")
    print(f"mse0 & {len(cv_stats[5])} & {min(cv_stats[5]):.4g} & {max(cv_stats[5]):.4g} & {np.mean(cv_stats[5]):.4g} & {np.std(cv_stats[5]):.4g} \\\\ \\hline")
    print(f"rmse & {len(cv_stats[6])} & {min(cv_stats[6]):.4g} & {max(cv_stats[6]):.4g} & {np.mean(cv_stats[6]):.4g} & {np.std(cv_stats[6]):.4g} \\\\ \\hline")
    print(f"mae & {len(cv_stats[7])} & {min(cv_stats[7]):.4g} & {max(cv_stats[7]):.4g} & {np.mean(cv_stats[7]):.4g} & {np.std(cv_stats[7]):.4g} \\\\ \\hline")
    print(f"smape & {len(cv_stats[8])} & {min(cv_stats[8]):.4g} & {max(cv_stats[8]):.4g} & {np.mean(cv_stats[8]):.4g} & {np.std(cv_stats[8]):.4g} \\\\ \\hline")
    print(f"m & {len(cv_stats[9])} & {min(cv_stats[9]):.4g} & {max(cv_stats[9]):.4g} & {np.mean(cv_stats[9]):.4g} & {np.std(cv_stats[9]):.4g} \\\\ \\hline")
    print(f"dfr & {len(cv_stats[10])} & {min(cv_stats[10]):.4g} & {max(cv_stats[10]):.4g} & {np.mean(cv_stats[10]):.4g} & {np.std(cv_stats[10]):.4g} \\\\ \\hline")
    print(f"df & {len(cv_stats[11])} & {min(cv_stats[11]):.4g} & {max(cv_stats[11]):.4g} & {np.mean(cv_stats[11]):.4g} & {np.std(cv_stats[11]):.4g} \\\\ \\hline")
    print(f"fStat & {len(cv_stats[12])} & {min(cv_stats[12]):.4g} & {max(cv_stats[12]):.4g} & {np.mean(cv_stats[12]):.4g} & {np.std(cv_stats[12]):.4g} \\\\ \\hline")
    print(f"aic & {len(cv_stats[13])} & {min(cv_stats[13]):.4g} & {max(cv_stats[13]):.4g} & {np.mean(cv_stats[13]):.4g} & {np.std(cv_stats[13]):.4g} \\\\ \\hline")
    print(f"bic & {len(cv_stats[14])} & {min(cv_stats[14]):.4g} & {max(cv_stats[14]):.4g} & {np.mean(cv_stats[14]):.4g} & {np.std(cv_stats[14]):.4g} \\\\ \\hline")
    
    # Close LaTeX environments
    print("\\end{tabular}")
    print("\\end{table}")