import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Tuple, Union
from get_cv_qof import get_cv_qof


def tune_ridge_lasso_alpha(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    method: str = 'ridge'
) -> Tuple[float, float]:
    """
    Tunes the regularization hyperparameter alpha for Ridge or Lasso regression.
    
    This function performs a multi-stage grid search to find the optimal alpha value
    that maximizes the mean R-squared (R^2) score via 5-fold cross-validation. The search
    progresses to narrow down the optimal value efficiently:
    1. A wide-range logarithmic coarse search to find the order of magnitude.
    2. An intermediate-range search relative to the first-stage winner.
    3. A fine-grained linear search tailored to the magnitude of the current best alpha.

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        method (str, optional): The regression method to tune. Must be either 
            'ridge' or 'lasso'. Defaults to 'ridge'.

    Returns:
        Tuple[float, float]: A tuple containing:
            - best_alpha (float): The optimal regularization strength found.
            - best_r_sq (float): The mean cross-validated R-squared score for the best alpha.
            
    Raises:
        ValueError: If `method` is not 'ridge' or 'lasso'.
    """
    print("-----------------------------------")
    print(f"--- Tuning alpha for {method} ---")
    print("-----------------------------------")

    if method not in ['ridge', 'lasso']:
        raise ValueError(f"method must be one of 'ridge' or 'lasso'. Received {method}")

    best_alpha: float = 0.0
    best_r_sq: float = -float('inf')

    # ==========================================
    # --- STAGE 1: Coarse Logarithmic Search ---
    # ==========================================
    # Evaluate orders of magnitude from very weak (10^-7) to very strong (10^6) regularization
    alpha_list_0 = [10**i for i in range(-7, 7)]

    for alpha in alpha_list_0:
        qof = get_cv_qof(X, y, method=method, alpha=alpha)
        # qof[0] is the R-squared metric list across the 5 folds; wrap mean in float() to
        # narrow from floating[Any] to float so the return type is satisfied
        cur_r_sq_mean: float = float(np.mean(qof[0]))

        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_alpha = float(alpha)

    # ==========================================
    # --- STAGE 2: Intermediate Search ---
    # ==========================================
    left_1 = best_alpha / 10
    alpha_list_1 = [left_1, 2*left_1, 4*left_1, 6*left_1, 8*left_1, 10*left_1,
                    20*left_1, 40*left_1, 60*left_1, 80*left_1, 100*left_1]

    for alpha in alpha_list_1:
        qof = get_cv_qof(X, y, method=method, alpha=alpha)
        cur_r_sq_mean = float(np.mean(qof[0]))

        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_alpha = float(alpha)

    # ==========================================
    # --- STAGE 3: Targeted Fine-Grained Search ---
    # ==========================================
    if best_alpha == left_1:
        alpha_0 = left_1 / 10
        alpha_list_2 = [alpha_0, 2*alpha_0, 4*alpha_0, 6*alpha_0, 8*alpha_0, 10*alpha_0,
                        20*alpha_0, 40*alpha_0, 60*alpha_0, 80*alpha_0, 100*alpha_0]

    elif best_alpha == 100 * left_1:
        alpha_0 = left_1 * 10
        alpha_list_2 = [alpha_0, 2*alpha_0, 4*alpha_0, 6*alpha_0, 8*alpha_0, 10*alpha_0,
                        20*alpha_0, 40*alpha_0, 60*alpha_0, 80*alpha_0, 100*alpha_0]

    elif best_alpha == 2 * left_1:
        dif_1 = left_1 / 4
        lam_0 = left_1
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        dif_2 = (2 * left_1) / 4
        lam_5 = lam_4 + dif_2; lam_6 = lam_5 + dif_2; lam_7 = lam_6 + dif_2; lam_8 = lam_7 + dif_2
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]

    elif best_alpha == 10 * left_1:
        dif_1 = (2 * left_1) / 4
        lam_0 = best_alpha - (2 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        dif_2 = (10 * left_1) / 4
        lam_5 = lam_4 + dif_2; lam_6 = lam_5 + dif_2; lam_7 = lam_6 + dif_2; lam_8 = lam_7 + dif_2
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]

    elif best_alpha == 20 * left_1:
        dif_1 = (10 * left_1) / 4
        lam_0 = best_alpha - (10 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        dif_2 = (20 * left_1) / 4
        lam_5 = lam_4 + dif_2; lam_6 = lam_5 + dif_2; lam_7 = lam_6 + dif_2; lam_8 = lam_7 + dif_2
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]

    elif best_alpha < 10 * left_1:
        dif_1 = (2 * left_1) / 4
        lam_0 = best_alpha - (2 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        lam_5 = lam_4 + dif_1; lam_6 = lam_5 + dif_1; lam_7 = lam_6 + dif_1; lam_8 = lam_7 + dif_1
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]

    else:  # best_alpha > 10 * left_1
        dif_1 = (20 * left_1) / 4
        lam_0 = best_alpha - (20 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        lam_5 = lam_4 + dif_1; lam_6 = lam_5 + dif_1; lam_7 = lam_6 + dif_1; lam_8 = lam_7 + dif_1
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]

    for alpha in alpha_list_2:
        qof = get_cv_qof(X, y, method=method, alpha=alpha)
        cur_r_sq_mean = float(np.mean(qof[0]))

        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_alpha = float(alpha)

    print(f"Best Alpha for {method}: {best_alpha} (CV R2: {best_r_sq:.4g})")
    return (best_alpha, best_r_sq)


def tune_box_cox_lambda(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame]
) -> Tuple[float, float]:
    """
    Tunes the power transformation lambda for a Box-Cox model.
    
    Iterates through a predefined grid of standard power transformations to determine 
    which lambda yields the highest mean R-squared via 5-fold cross-validation. 
    Standard semantic mappings apply (e.g., 0 for log transform, 0.5 for square root, 
    -1 for inverse).

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector to be transformed.

    Returns:
        Tuple[float, float]: A tuple containing:
            - best_lambda (float): The optimal Box-Cox transformation parameter.
            - best_r_sq (float): The mean cross-validated R-squared score for the best lambda.
    """
    print("-----------------------------------")
    print(f"--- Tuning lambda for Box-Cox ---")
    print("-----------------------------------")

    best_lambda: float = 0.0
    best_r_sq: float = -float('inf')

    lambdas = [-4, -3, -2, -1, -1.0/2, -1.0/3, -1.0/4, -1.0/5, -1.0/6, -1.0/7, -1.0/8,
               -1.0/9, -1.0/10, -1.0/11, -1.0/12, 0, 1.0/12, 1.0/11, 1.0/10, 1.0/9,
               1.0/8, 1.0/7, 1.0/6, 1.0/5, 1.0/4, 1.0/3, 1.0/2, 1, 2, 3, 4]

    for lambda_ in lambdas:
        qof = get_cv_qof(X, y, method='boxcox', lambda_=lambda_)
        cur_r_sq_mean: float = float(np.mean(qof[0]))

        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_lambda = float(lambda_)

    print(f"Best Lambda for Box-Cox: {best_lambda} (CV R2: {best_r_sq:.4g})")
    return (best_lambda, best_r_sq)


def tune_nn_hyperparams(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    method: str = 'nn_3L'
) -> Tuple[nn.Module, nn.Module, nn.Module, int, int, float]:
    """
    Tunes the activation functions, hidden layer sizes, and learning rate for a
    PyTorch neural network via sequential 5-fold cross-validation grid searches.

    The three stages run in dependency order so each subsequent stage builds on
    the winner from the previous one:

        Stage 1 — Activation function(s):
            Searches over [Identity, Sigmoid, Tanh, ReLU, LeakyReLU, ELU] for both
            the hidden layer(s) AND the output layer for all NN variants.

            - ``nn_2L``: Only the output activation is tuned (no hidden layers).
              ``best_activation_fn`` and ``best_activation_fn_2`` are returned as
              ``nn.Identity()`` placeholders.
            - ``nn_3L``: The single hidden activation and the output activation are
              searched jointly across all (hidden × output) pairs.
            - ``nn_4L``: Both hidden activations and the output activation are
              searched jointly across all (h1 × h2 × output) triples.

        Stage 2 — Hidden layer size(s):
            Searches [8, 16, 32, 64, 100, 128, 200, 256] nodes for each hidden
            layer using the Stage 1 winners.
            ``nn_2L`` has no hidden layers so both sizes are returned as defaults
            (``nn_hidden_1=100``, ``nn_hidden_2=50``).

        Stage 3 — Learning rate:
            Searches [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1] using the
            activation functions and hidden sizes selected in previous stages.

    All stages optimise the mean cross-validated R² (``qof[0]``).

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (Union[pd.Series, pd.DataFrame]): The target response vector.
        method (str, optional): Which NN variant to tune.  Must be one of
            ``'nn_2L'``, ``'nn_3L'``, or ``'nn_4L'``.  Defaults to ``'nn_3L'``.

    Returns:
        Tuple[nn.Module, nn.Module, nn.Module, int, int, float]: A 6-element tuple:
            - ``best_activation_fn``     (nn.Module): Best hidden-layer activation
              for layer 1 (``nn.Identity()`` placeholder for ``nn_2L``).
            - ``best_activation_fn_2``   (nn.Module): Best hidden-layer activation
              for layer 2 (``nn.Identity()`` placeholder for ``nn_2L``/``nn_3L``).
            - ``best_output_activation_fn`` (nn.Module): Best output-layer activation
              for all NN variants.
            - ``best_hidden_1`` (int): Best node count for hidden layer 1
              (default 100 for ``nn_2L``).
            - ``best_hidden_2`` (int): Best node count for hidden layer 2
              (default 50 for ``nn_2L``/``nn_3L``).
            - ``best_lr`` (float): Best learning rate found in Stage 3.

    Raises:
        ValueError: If ``method`` is not one of the three supported NN strings.
    """
    if method not in ['nn_2L', 'nn_3L', 'nn_4L']:
        raise ValueError(
            f"method must be one of 'nn_2L', 'nn_3L', or 'nn_4L'. Received {method}"
        )

    print("-----------------------------------")
    print(f"--- Tuning hyperparameters for {method} ---")
    print("-----------------------------------")

    # ==========================================
    # --- Candidate Grids ---
    # ==========================================
    activation_candidates = [
        ("Identity",  nn.Identity()),
        ("Sigmoid",   nn.Sigmoid()),
        ("Tanh",      nn.Tanh()),
        ("ReLU",      nn.ReLU()),
        ("LeakyReLU", nn.LeakyReLU()),
        ("ELU",       nn.ELU()),
    ]

    hidden_size_candidates = [8, 16, 32, 64, 100, 128, 200, 256]

    lr_candidates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

    # ==========================================
    # --- Stage 1: Activation Function(s) ---
    # ==========================================
    # Output activation is tuned for ALL NN variants.
    # Hidden activations are tuned only for nn_3L (1 hidden) and nn_4L (2 hidden).
    print(f"[{method}] Stage 1: Tuning activation function(s)...")

    best_act_r_sq: float = -float('inf')
    best_activation_fn:        nn.Module = nn.Identity()   # hidden layer 1 placeholder
    best_activation_fn_2:      nn.Module = nn.Identity()   # hidden layer 2 placeholder
    best_output_activation_fn: nn.Module = nn.Identity()   # output layer

    if method == 'nn_2L':
        # No hidden layers — only search the output activation dimension.
        for out_name, out_fn in activation_candidates:
            qof = get_cv_qof(X, y, method=method, output_activation_fn=out_fn)
            cur_r_sq: float = float(np.mean(qof[0]))
            print(f"  output_activation={out_name:10s}  CV R²={cur_r_sq:.4g}")
            if cur_r_sq > best_act_r_sq:
                best_act_r_sq          = cur_r_sq
                best_output_activation_fn = out_fn

    elif method == 'nn_3L':
        # One hidden layer — search the hidden activation functions.
        for act_name, act_fn in activation_candidates:
            qof = get_cv_qof(
                X, y, method=method,
                activation_fn=act_fn,
                output_activation_fn=nn.Identity()
            )
            cur_r_sq = float(np.mean(qof[0]))
            print(
                f"  hidden={act_name:10s}  output={str(nn.Identity()):10s}"
                f"  CV R²={cur_r_sq:.4g}"
            )
            if cur_r_sq > best_act_r_sq:
                best_act_r_sq             = cur_r_sq
                best_activation_fn        = act_fn

        # One hidden layer — search the output activation functions.
        for out_name, out_fn in activation_candidates:
            qof = get_cv_qof(
                X, y, method=method,
                activation_fn=best_activation_fn,
                output_activation_fn=out_fn
            )
            cur_r_sq = float(np.mean(qof[0]))
            print(
                f"  hidden={str(best_activation_fn):10s}  output={out_name:10s}"
                f"  CV R²={cur_r_sq:.4g}"
            )
            if cur_r_sq > best_act_r_sq:
                best_act_r_sq             = cur_r_sq
                best_output_activation_fn = out_fn

    else:  # nn_4L — two hidden layers + output; search three dimensions sequentially (h1, then h2, then output).
        for act_name_1, act_fn_1 in activation_candidates:
            qof = get_cv_qof(
                X, y, method=method,
                activation_fn=act_fn_1,
                activation_fn_2=nn.Identity(),
                output_activation_fn=nn.Identity()
            )
            cur_r_sq = float(np.mean(qof[0]))
            print(
                f"  h1={act_name_1:10s}  h2={str(nn.Identity()):10s}"
                f"  out={str(nn.Identity()):10s}  CV R²={cur_r_sq:.4g}"
            )
            if cur_r_sq > best_act_r_sq:
                best_act_r_sq             = cur_r_sq
                best_activation_fn        = act_fn_1

        for act_name_2, act_fn_2 in activation_candidates:
            qof = get_cv_qof(
                X, y, method=method,
                activation_fn=best_activation_fn,
                activation_fn_2=act_fn_2,
                output_activation_fn=nn.Identity()
            )
            cur_r_sq = float(np.mean(qof[0]))
            print(
                f"  h1={str(best_activation_fn):10s}  h2={act_name_2:10s}"
                f"  out={str(nn.Identity()):10s}  CV R²={cur_r_sq:.4g}"
            )
            if cur_r_sq > best_act_r_sq:
                best_act_r_sq             = cur_r_sq
                best_activation_fn_2      = act_fn_2
        
        for out_name, out_fn in activation_candidates:
            qof = get_cv_qof(
                X, y, method=method,
                activation_fn=best_activation_fn,
                activation_fn_2=best_activation_fn_2,
                output_activation_fn=out_fn
            )
            cur_r_sq = float(np.mean(qof[0]))
            print(
                f"  h1={str(best_activation_fn):10s}  h2={str(best_activation_fn_2):10s}"
                f"  out={out_name:10s}  CV R²={cur_r_sq:.4g}"
            )
            if cur_r_sq > best_act_r_sq:
                best_act_r_sq             = cur_r_sq
                best_output_activation_fn = out_fn

    print(f"[{method}] Stage 1 complete. Best CV R²={best_act_r_sq:.4g}")

    # ==========================================
    # --- Stage 2: Hidden Layer Size(s) ---
    # ==========================================
    # nn_2L has no hidden layers; return defaults.
    if method == 'nn_2L':
        best_hidden_1: int = 100
        best_hidden_2: int = 50
        print(f"[{method}] Stage 2 skipped (no hidden layers). Using defaults h1=100, h2=50.")
    else:
        print(f"[{method}] Stage 2: Tuning hidden layer size(s)...")

        best_size_r_sq: float = -float('inf')
        best_hidden_1 = 100   # will be overwritten
        best_hidden_2 = 50    # placeholder for nn_3L

        if method == 'nn_3L':
            # Single hidden layer — search one size dimension.
            for h1 in hidden_size_candidates:
                qof = get_cv_qof(
                    X, y, method=method,
                    activation_fn=best_activation_fn,
                    output_activation_fn=best_output_activation_fn,
                    nn_hidden_1=h1
                )
                cur_r_sq = float(np.mean(qof[0]))
                print(f"  hidden_1={h1:4d}  CV R²={cur_r_sq:.4g}")
                if cur_r_sq > best_size_r_sq:
                    best_size_r_sq = cur_r_sq
                    best_hidden_1  = h1

        else:  # nn_4L — two hidden layers; search jointly.
            for h1 in hidden_size_candidates:
                for h2 in hidden_size_candidates:
                    if h2 <= h1:  # enforce h2 ≤ h1 to reduce search space and promote funnel-shaped architecture
                        qof = get_cv_qof(
                            X, y, method=method,
                            activation_fn=best_activation_fn,
                            activation_fn_2=best_activation_fn_2,
                            output_activation_fn=best_output_activation_fn,
                            nn_hidden_1=h1,
                            nn_hidden_2=h2
                        )
                        cur_r_sq = float(np.mean(qof[0]))
                        print(f"  hidden_1={h1:4d}  hidden_2={h2:4d}  CV R²={cur_r_sq:.4g}")
                        if cur_r_sq > best_size_r_sq:
                            best_size_r_sq = cur_r_sq
                            best_hidden_1  = h1
                            best_hidden_2  = h2

        print(
            f"[{method}] Stage 2 complete. "
            f"Best h1={best_hidden_1}, h2={best_hidden_2}, CV R²={best_size_r_sq:.4g}"
        )

    # ==========================================
    # --- Stage 3: Learning Rate ---
    # ==========================================
    print(f"[{method}] Stage 3: Tuning learning rate...")

    best_lr_r_sq: float = -float('inf')
    best_lr: float = 0.01

    for lr in lr_candidates:
        qof = get_cv_qof(
            X, y, method=method,
            activation_fn=best_activation_fn,
            activation_fn_2=best_activation_fn_2,
            output_activation_fn=best_output_activation_fn,
            nn_hidden_1=best_hidden_1,
            nn_hidden_2=best_hidden_2,
            lr=lr
        )
        cur_r_sq = float(np.mean(qof[0]))
        print(f"  lr={lr:.0e}  CV R²={cur_r_sq:.4g}")
        if cur_r_sq > best_lr_r_sq:
            best_lr_r_sq = cur_r_sq
            best_lr      = lr

    print(
        f"[{method}] Stage 3 complete. Best lr={best_lr:.0e}, CV R²={best_lr_r_sq:.4g}"
    )
    print(
        f"[{method}] Final tuned hyperparams — "
        f"activation_fn={str(best_activation_fn)}, activation_fn_2={str(best_activation_fn_2)}, output_activation_fn={str(best_output_activation_fn)}, "
        f"h1_size={best_hidden_1}, h2_size={best_hidden_2}, lr={best_lr:.0e}"
    )

    return (best_activation_fn, best_activation_fn_2, best_output_activation_fn, best_hidden_1, best_hidden_2, best_lr)
