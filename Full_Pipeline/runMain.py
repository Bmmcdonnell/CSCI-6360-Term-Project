import pandas as pd
from sklearn.model_selection import train_test_split
from main import get_tables

def tp_insurance():
    """
    Main entry point for evaluating regression models on the Medical Insurance dataset.
    
    * Loads data.
    * Calls the `get_tables` orchestrator to generate metrics, plots, and LaTeX summaries.
    """
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    OXy = pd.read_csv("Insurance_Charges_1/cleaned_insurance_with_intercept.csv")
    OX = OXy.drop('charges', axis=1)
    y = OXy[['charges']]

    OXY_2 = pd.read_csv("Insurance_Charges_1/cleaned_order_2_insurance_with_intercept.csv")
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
    get_tables(OX, X, y, OX_test, OX_train, X_test, X_train, y_test, y_train, X_2, X_2_test, X_2_train, "Insurance Charges", "Insurance_Charges_1_Plots")

    print("------------------------------------")
    print("Finished")
    print("------------------------------------")

tp_insurance()