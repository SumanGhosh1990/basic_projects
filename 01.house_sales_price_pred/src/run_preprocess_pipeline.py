from pathlib import Path
import numpy as np
import joblib
from src.house_price_eda_w_data_prep import *
from src.utils import add_time_features, add_numeric_features

project_root = Path.cwd().parent
train_path = project_root / "01.house_sales_price_pred" / "data" / "train.csv"
test_path = project_root / "01.house_sales_price_pred" / "data" / "test.csv"
artifacts_dir = project_root / "01.house_sales_price_pred" / "artifacts"

# 1) Load
train_df, test_df = load_data(train_path, test_path)
print(f"[1] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 2) EDA (independent)
missing_train = eda_info_missing(train_df, "train", artifacts_dir)

# 3) Train/test split (independent)
dev, itv = train_test_split_df(train_df, target="SalePrice", test_size=0.2,
                               random_state=42, stratify=False)
print(f"[3] Dev shape: {dev.shape}, Itv shape: {itv.shape}")

# # 4) Plot pipeline (dev only)
# numeric_feats = dev.select_dtypes(include=["int64","float64"]).columns.tolist()[:10]
# plot_features_vs_target(dev, features=numeric_feats, target="SalePrice")

# 5) Feature engineering on dev and itv
dev_fe = add_time_features(dev)
dev_fe = add_numeric_features(dev_fe)
print(f"[5] Dev (after FE) shape: {dev_fe.shape}")

itv_fe = add_time_features(itv)
itv_fe = add_numeric_features(itv_fe)
print(f"[5] Iev (after FE) shape: {itv_fe.shape}")

# 6) Impute (fit on dev)
object_cols = dev_fe.select_dtypes(include="object").columns.tolist()
numeric_cols = dev_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
category_cols = dev_fe.select_dtypes(include="category").columns.tolist()
bool_cols = dev_fe.select_dtypes(include="bool").columns.tolist()


dev_imputed, imputers = impute_missing(dev_fe, 
                                       object_cols, 
                                       list(set(numeric_cols) - set(["YearBuilt","YearRemodAdd","YrSold"])),
                                       category_cols, 
                                       bool_cols)
print(f"[6] Dev (after imputation) shape: {dev_imputed.shape}")

# Apply on itv
imputers = joblib.load("artifacts/imputers.pkl")
itv_imputed = apply_imputers(itv_fe, 
                             imputers, 
                             object_cols,
                             list(set(numeric_cols) - set(["YearBuilt","YearRemodAdd","YrSold"])),
                             category_cols, 
                             bool_cols)
print(f"[6] Iev (after imputation) shape: {itv_imputed.shape}")



# 7) Encoding (fit on dev) and save artifacts
dev_encoded, enc_artifacts = fit_encoding_and_encode(dev_imputed, 
                                                     target="SalePrice",
                                                     cutoff=5,
                                                     artifacts_dir=artifacts_dir)
print(f"[7] Dev (after encoding) shape: {dev_encoded.shape}")

# Apply on itv
itv_encoded = apply_encoders(itv_imputed, artifacts_dir)
print(f"[7] ITV (after encoding) shape: {itv_encoded.shape}")

# 8) Numeric summary + clipping
numeric_summary = describe_numeric_summary(dev_encoded)

dev_clipped, itv_clipped = clip_with_summary(dev_encoded, itv_encoded, numeric_summary)
print(f"[8] Dev (clipped) shape: {dev_clipped.shape}, ITV (clipped) shape: {itv_clipped.shape}")
