
import numpy as np
import pandas as pd


def create_features(df):
   

    df = df.copy()

    # Zero Balance Flag
    df["Is_Zero_Balance"] = np.where(
        df["Balance"].isna(),
        np.nan,
        (df["Balance"] == 0).astype(int)
    )

    #  Age Group
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, np.inf],
        labels=["Under 25", "25-34", "35-44", "45-54", "55+"],
        right=False
    )

    # Balance to Salary Ratio
    df["Balance_To_Salary"] = (
        df["Balance"] /
        df["EstimatedSalary"].replace(0, np.nan)
    )

    # Product Risk Flag
    df["Product_Risk_Flag"] = np.where(
        df["NumOfProducts"].isna(),
        np.nan,
        (df["NumOfProducts"] >= 3).astype(int)
    )

    return df
