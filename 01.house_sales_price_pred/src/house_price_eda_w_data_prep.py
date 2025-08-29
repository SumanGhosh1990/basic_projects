import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from typing import Tuple, Dict, Any

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer   


# ------------------------------
# 1) Data loading pipeline (independent)
# ------------------------------
def load_data(train_path: Path, test_path: Path, drop_id: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if drop_id:
        for df in (train_df, test_df):
            if "Id" in df.columns:
                df.drop(columns=["Id"], inplace=True)
    return train_df, test_df


# ------------------------------
# 2) EDA pipeline (independent)
#    prints info(), missing %, saves missing report
# ------------------------------
def eda_info_missing(df: pd.DataFrame, name: str, artifacts_dir: Path):
    print(f"\n=== INFO: {name} ===")
    print(df.info())
    print(f"\n=== Describe (top) {name} ===")
    print(df.describe(include='all').T.head(20))
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_perc = (missing / len(df)).round(4)
    missing_df = pd.concat([missing, missing_perc], axis=1, keys=["TotalMissing", "MissingPerc"])
    os.makedirs(artifacts_dir, exist_ok=True)
    missing_df.to_csv(artifacts_dir / f"missing_{name}.csv")
    print(f"\nSaved missing report -> {artifacts_dir / f'missing_{name}.csv'}")
    return missing_df



# ------------------------------
# 3) Train test split pipeline (independent)
# ------------------------------
def train_test_split_df(df: pd.DataFrame, 
                        target: str, 
                        test_size: float = 0.2, 
                        random_state: int = 42, 
                        stratify: bool = True):
    """
    Split a DataFrame into train and test sets, keeping independent and dependent variables together.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing both features and target.
    target : str
        Name of the target column.
    test_size : float, default=0.2
        Proportion of dataset to include in the test split.
    random_state : int, default=42
        Controls the shuffling applied before the split.
    stratify : bool, default=True
        If True, stratify based on the target column.

    Returns
    -------
    train_df : pd.DataFrame
        Training set with features and target.
    test_df : pd.DataFrame
        Test set with features and target.
    """
    stratify_col = df[target] if stratify else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    return train_df, test_df



# ------------------------------
# 4) Plot pipeline (dev only, independent)
# ------------------------------
def plot_features_vs_target(df: pd.DataFrame, features: list, target: str, show_corr_heatmap: bool = True):
    """
    Plots distribution/relationship of given features with target variable.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    features (list): List of column names to visualize
    target (str): Target column name (e.g. 'SalePrice')
    show_corr_heatmap (bool): If True, shows correlation heatmap for numeric features
    """
    numeric_feats = []
    
    for col in features:
        plt.figure(figsize=(8, 5))
        
        if df[col].dtype == "category" or df[col].dtype == "object":
            # Categorical / Object variables → boxplot
            sns.boxplot(x=df[col], y=df[target])
            plt.xticks(rotation=45)
            plt.title(f"{col} vs {target}")
        
        else:
            # Numeric variables → scatterplot + correlation
            corr = df[[col, target]].corr().iloc[0, 1]
            numeric_feats.append(col)
            
            sns.scatterplot(x=df[col], y=df[target])
            plt.title(f"{col} vs {target} (corr = {corr:.2f})")
        
        plt.tight_layout()
        plt.show()
    
    # Show correlation heatmap if requested
    if show_corr_heatmap and numeric_feats:
        corr_matrix = df[numeric_feats + [target]].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Correlation Heatmap with Target")
        plt.show()


# ------------------------------
# 5) Feature engineering + impute + encoding + save artifacts (fit on dev)
# ------------------------------

#Impute missing values pipeline (fit on dev, save imputers)
def impute_missing(
    df: pd.DataFrame,
    object_cols: list[str],
    numeric_cols: list[str],
    category_cols: list[str],
    bool_cols: list[str],
    artifacts_dir: Path = Path("artifacts")
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Fit imputers and apply them on training data.
    Saves imputers + column lists as artifacts for reuse.

    Returns
    -------
    df_copy : pd.DataFrame
        Imputed dataframe
    imputers : dict
        Dictionary of imputers/strategies for later use
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df_copy = df.copy()
    imputers = {}
    print(len(numeric_cols))
    print(numeric_cols)

    # 1. Object columns -> fillna("Not available")
    if object_cols:
        df_copy[object_cols] = df_copy[object_cols].fillna("Not available")
        imputers["object"] = {"strategy": "fillna", "value": "Not available"}

    # 2. Category columns -> add "Not available" category then fill
    if category_cols:
        for col in category_cols:
            if "Not available" not in df_copy[col].cat.categories:
                df_copy[col] = df_copy[col].cat.add_categories("Not available")
            df_copy[col] = df_copy[col].fillna("Not available")
        imputers["category"] = {"strategy": "add_category+fillna", "value": "Not available"}

    # 3. Boolean columns -> fillna(False)
    if bool_cols:
        df_copy[bool_cols] = df_copy[bool_cols].fillna(False)
        imputers["bool"] = {"strategy": "fillna", "value": False}

    # 4. Numeric columns -> median imputer
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df_copy[numeric_cols] = num_imputer.fit_transform(df_copy[numeric_cols])
        imputers["numeric"] = num_imputer

    # ---- Save artifacts ----
    joblib.dump(imputers, artifacts_dir / "imputers.pkl")
    pd.Series(object_cols).to_csv(artifacts_dir / "object_cols.csv", index=False)
    pd.Series(numeric_cols).to_csv(artifacts_dir / "numeric_cols.csv", index=False)
    pd.Series(category_cols).to_csv(artifacts_dir / "category_cols.csv", index=False)
    pd.Series(bool_cols).to_csv(artifacts_dir / "bool_cols.csv", index=False)

    return df_copy, imputers


#Apply imputers (itv/test)
def apply_imputers(
    df: pd.DataFrame,
    imputers: dict,
    object_cols: list,
    numeric_cols: list,
    category_cols: list,
    bool_cols: list,
    artifacts_dir: Path = Path("artifacts")
) -> pd.DataFrame:
    """
    Apply previously fitted imputers to a new dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (e.g., validation/test set).
    imputers : dict
        Dictionary of imputers/values for different data types.
    numeric_cols, object_cols, category_cols, bool_cols : list
        Column lists from training.
    artifacts_dir : Path
        Where imputers and column lists are saved.

    Returns
    -------
    df_imputed : pd.DataFrame
        New dataframe with imputations applied.
    """
    df_imputed = df.copy()
    print(numeric_cols)
    len(numeric_cols)

    # Numeric
    if numeric_cols and "numeric" in imputers:
        df_imputed[numeric_cols] = imputers["numeric"].transform(df_imputed[numeric_cols])

    # Object
    if object_cols and "object" in imputers:
        df_imputed[object_cols] = df_imputed[object_cols].fillna(imputers["object"]["value"])

    # Category
    for col in category_cols:
        if col in df_imputed.columns:
            if "Not available" not in df_imputed[col].cat.categories:
                df_imputed[col] = df_imputed[col].cat.add_categories("Not available")
            df_imputed[col] = df_imputed[col].fillna("Not available")

    # Boolean
    if bool_cols and "bool" in imputers:
        df_imputed[bool_cols] = df_imputed[bool_cols].fillna(imputers["bool"]["value"])

    return df_imputed

# def apply_imputers(
#     df: pd.DataFrame,
#     imputers: dict,
#     numeric_cols: list,
#     object_cols: list,
#     category_cols: list,
#     bool_cols: list
# ) -> pd.DataFrame:
#     """
#     Apply imputers to a dataframe and return an imputed copy.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataset to apply imputations on (e.g., test set).
#     imputers : dict
#         Dictionary of imputers/values for different data types.
#     numeric_cols : list
#         List of numeric columns.
#     object_cols : list
#         List of object/text columns.
#     category_cols : list
#         List of categorical columns.
#     bool_cols : list
#         List of boolean columns.

#     Returns
#     -------
#     df_imputed : pd.DataFrame
#         New dataframe with imputations applied.
#     """
#     df_imputed = df.copy()

#     # Numeric
#     if numeric_cols:
#         df_imputed[numeric_cols] = imputers["numeric"].transform(df_imputed[numeric_cols])

#     # Object
#     if object_cols:
#         df_imputed[object_cols] = df_imputed[object_cols].fillna(imputers["object"]["value"])

#     # Category
#     for col in category_cols:
#         if "Not available" not in df_imputed[col].cat.categories:
#             df_imputed[col] = df_imputed[col].cat.add_categories("Not available")
#         df_imputed[col] = df_imputed[col].fillna("Not available")

#     # Boolean
#     if bool_cols:
#         df_imputed[bool_cols] = df_imputed[bool_cols].fillna(imputers["bool"]["value"])

#     return df_imputed

# Encoding pipeline (fit on dev, save encodings)
def fit_encoding_and_encode(
    df: pd.DataFrame,
    target: str,
    cutoff: int = 10,
    artifacts_dir: Path = Path("artifacts")
) -> tuple[pd.DataFrame, dict[str,float | list[str]]]:
    df = df.copy()

    # ---------- Ordinal numeric ----------
    ordinal_numeric = ["OverallQual", "OverallCond"]

    # ---------- Object but ordinal category ----------
    ordinal_maps = {
        "ExterQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "ExterCond": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "BsmtQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "BsmtCond": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "BsmtExposure": {"Gd":4,"Av":3,"Mn":2,"No":1,"NA":0},
        "BsmtFinType1": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"NA":0},
        "BsmtFinType2": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"NA":0},
        "HeatingQC": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "KitchenQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "FireplaceQu": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "GarageQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "GarageCond": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "GarageFinish": {"Fin":3,"RFn":2,"Unf":1,"NA":0},
        "PoolQC": {"Ex":4,"Gd":3,"TA":2,"Fa":1,"NA":0},
        "Fence": {"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"NA":0},
        "Functional": {"Typ":7,"Min1":6,"Min2":5,"Mod":4,"Maj1":3,"Maj2":2,"Sev":1,"Sal":0},
        "PavedDrive": {"Y":2,"P":1,"N":0}
    }

    var_type_records = []  # store variable classification

    # Apply ordinal mappings
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            var_type_records.append({"variable": col, "type": "ordinal_categorical"})

    for col in ordinal_numeric:
        if col in df.columns:
            var_type_records.append({"variable": col, "type": "ordinal_numeric"})

    # ---------- Case 1: Numeric but categorical ----------
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype("object")
        var_type_records.append({"variable": "MSSubClass", "type": "categorical_numeric"})

    # ---------- Case 2: Numeric with low variation ----------
    low_var_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in [target] + ordinal_numeric and df[col].nunique() < 3
    ]
    for col in low_var_cols:
        df[col] = df[col].astype("object")
        var_type_records.append({"variable": col, "type": "categorical_numeric"})

    # ---------- Case 3: Object but no order ----------
    nominal_cols = df.select_dtypes(include="object").columns

    encoding_records = []
    onehot_cols = []
    target_encoded_cols = []
    global_mean = df[target].mean()

    for col in nominal_cols:
        if df[col].nunique() <= cutoff:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            onehot_cols.extend(dummies.columns.tolist())
                
            # tag the newly created dummy columns, not the dropped one
            for dcol in dummies.columns:
                var_type_records.append({"variable": dcol, "type": "nominal_categorical"})
        else:
            target_mean = df.groupby(col)[target].mean()
            temp = target_mean.reset_index()
            temp.columns = ["Category", "AverageSalePrice"]
            temp.insert(0, "Variable", col)
            encoding_records.append(temp)
            df[col] = df[col].map(target_mean).fillna(global_mean)
            target_encoded_cols.append(col)
            var_type_records.append({"variable": col, "type": "target_encoded"})

    # ---------- Final sweep: assign pure numeric ----------
    tagged_vars = {rec["variable"] for rec in var_type_records}
    for col in df.columns:
        if col not in tagged_vars and col != target:
            if pd.api.types.is_numeric_dtype(df[col]):
                var_type_records.append({"variable": col, "type": "continuous_numeric"})

    # ---------- Save artifacts ----------
    os.makedirs(artifacts_dir, exist_ok=True)

    if encoding_records:
        encoding_df = pd.concat(encoding_records, ignore_index=True)
        encoding_df.to_csv(artifacts_dir / "target_mean_encoding.csv", index=False)

    pd.Series(onehot_cols).to_csv(artifacts_dir / "onehot_columns.csv", index=False)
    pd.Series([global_mean]).to_csv(artifacts_dir / "global_mean.csv", index=False)
    pd.Series(df.columns).to_csv(artifacts_dir / "feature_columns.csv", index=False)

    var_type_df = pd.DataFrame(var_type_records)
    var_type_df.to_csv(artifacts_dir / "variable_types.csv", index=False)

    return df, {"onehot_cols": onehot_cols, "global_mean": global_mean, "var_types": var_type_records}


# Apply encoding pipeline (itv/test)
def apply_encoders(
    df: pd.DataFrame, 
    artifacts_dir: Path = Path("artifacts")
) -> pd.DataFrame:
    
    df_copy = df.copy()

    # load encodings
    encoding_df = pd.read_csv(artifacts_dir / "target_mean_encoding.csv")
    onehot_cols = pd.read_csv(artifacts_dir / "onehot_columns.csv")
    global_mean = pd.read_csv(artifacts_dir  / "global_mean.csv")
    final_features = pd.read_csv(artifacts_dir / "feature_columns.csv")

    onehot_cols = onehot_cols.iloc[:,0].tolist()
    global_mean = global_mean.iloc[0,0]
    final_features = final_features.iloc[:,0].tolist()

    # ---------- Object but ordinal category ----------
    ordinal_maps = {
        "ExterQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "ExterCond": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "BsmtQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "BsmtCond": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "BsmtExposure": {"Gd":4,"Av":3,"Mn":2,"No":1,"NA":0},
        "BsmtFinType1": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"NA":0},
        "BsmtFinType2": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"NA":0},
        "HeatingQC": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "KitchenQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "FireplaceQu": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "GarageQual": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "GarageCond": {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0},
        "GarageFinish": {"Fin":3,"RFn":2,"Unf":1,"NA":0},
        "PoolQC": {"Ex":4,"Gd":3,"TA":2,"Fa":1,"NA":0},
        "Fence": {"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"NA":0},
        "Functional": {"Typ":7,"Min1":6,"Min2":5,"Mod":4,"Maj1":3,"Maj2":2,"Sev":1,"Sal":0},
        "PavedDrive": {"Y":2,"P":1,"N":0}
    }
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype("object")

    # target encodings
    for variable in encoding_df["Variable"].unique():
        mapping = encoding_df[encoding_df["Variable"] == variable] \
                    .set_index("Category")["AverageSalePrice"].to_dict()
        if variable in df.columns:
            df[variable] = df[variable].map(mapping).fillna(global_mean)

    # one-hot encoding alignment
    current_nominals = df.select_dtypes(include="object").columns
    for col in current_nominals:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    for col in onehot_cols:
        if col not in df.columns:
            df[col] = 0  # add missing dummy column

    # align final feature order
    df = df.reindex(columns=final_features, fill_value=0)

    return df


# ------------------------------
# 6) Numeric summary & flooring/capping
# ------------------------------
def describe_numeric_summary(dev: pd.DataFrame, 
                             var_types_path: Path = Path("artifacts/variable_types.csv"),
                             save_path: Path = Path("artifacts/numeric_summary.csv")) -> pd.DataFrame:
    
    var_types = pd.read_csv(var_types_path)
    cont_vars = var_types.loc[var_types["type"] == "continuous_numeric", "variable"].tolist()
    summary = []
    for col in cont_vars:
        if col in dev.columns:
            arr = dev[col].dropna().values
            stats = {
                "variable": col,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "p1": float(np.percentile(arr,1)),
                "p10": float(np.percentile(arr,10)),
                "p25": float(np.percentile(arr,25)),
                "p50": float(np.percentile(arr,50)),
                "p75": float(np.percentile(arr,75)),
                "p90": float(np.percentile(arr,90)),
                "p95": float(np.percentile(arr,95)),
                "p99": float(np.percentile(arr,99)),
            }
            summary.append(stats)

    summary_df = pd.DataFrame(summary)

    # Ensure artifacts directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)

    return summary_df

def clip_with_summary(dev: pd.DataFrame, 
                      itv: pd.DataFrame, 
                      summary_df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    
    dev_clipped = dev.copy()
    itv_clipped = itv.copy()
    for _, row in summary_df.iterrows():
        col = row["variable"]
        p1, p99 = row["p1"], row["p99"]
        if col in dev_clipped.columns:
            dev_clipped[col] = dev_clipped[col].clip(lower=p1, upper=p99)
        if col in itv_clipped.columns:
            itv_clipped[col] = itv_clipped[col].clip(lower=p1, upper=p99)

        # Verification print (optional)
        print(f"{col} -> Dev [{dev_clipped[col].min()}, {dev_clipped[col].max()}], "
              f"ITV [{itv_clipped[col].min()}, {itv_clipped[col].max()}], "
              f"Expected [{p1}, {p99}]")
    
    return dev_clipped, itv_clipped
