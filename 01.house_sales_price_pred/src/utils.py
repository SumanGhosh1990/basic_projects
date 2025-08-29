import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Age-related features ----
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # ---- Was remodeled or not ----
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    # ---- Time since remodel vs build ----
    df["SinceRemodelVsBuild"] = df["YearRemodAdd"] - df["YearBuilt"]

    # ---- House age bins ----
    df["HouseAgeBin"] = pd.cut(
        df["HouseAge"],
        bins=[-1, 10, 30, 60, 100, 200],  # -1 handles 0 properly
        labels=["New", "Recent", "Mid", "Old", "VeryOld"]
    )

    # ---- Seasonality of sale ----
    df["SaleSeason"] = df["MoSold"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })

    # ---- Interaction features ----
    if "OverallQual" in df.columns:
        df["AgeQualInteraction"] = df["HouseAge"] * df["OverallQual"]

    if "OverallCond" in df.columns:
        df["RemodCondInteraction"] = df["RemodAge"] * df["OverallCond"]

    # if "OverallQual" in df.columns:
    #     df["RemodQualBoost"] = df["IsRemodeled"] * df["OverallQual"]

    # ---- Drop raw year variables ----
    df = df.drop(columns=["YearBuilt", "YearRemodAdd", "YrSold"], errors="ignore")

    return df

def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- 1. Total Bathrooms ----
    df["TotalBathrooms"] = (
        df["FullBath"] + (0.5 * df["HalfBath"]) +
        df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"])
    )

    # ---- 2. Total Porch SF ----
    df["TotalPorchSF"] = (
        df["WoodDeckSF"] + df["OpenPorchSF"] +
        df["EnclosedPorch"] + df["3SsnPorch"] +
        df["ScreenPorch"]
    )

    # ---- 3. Total House SF (basement + 1st + 2nd floor) ----
    df["TotalHouseSF"] = (
        df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
    )

    # ---- 4. Total Finished SF (above grade + finished basement) ----
    df["TotalFinishedSF"] = (
        df["GrLivArea"] + df["BsmtFinSF1"] + df["BsmtFinSF2"]
    )

    # ---- 5. Total Rooms (above grade + bedrooms + kitchens) ----
    df["TotalRooms"] = (
        df["TotRmsAbvGrd"] + df["BedroomAbvGr"] + df["KitchenAbvGr"]
    )

    # ---- 6. Baths per Bedroom ----
    df["BathsPerBedroom"] = df["TotalBathrooms"] / (df["BedroomAbvGr"] + 1)

    # ---- 7. Living Area per Room ----
    df["LivingAreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)

    # ---- 8. Garage Score (cars * area) ----
    df["GarageScore"] = df["GarageCars"] * df["GarageArea"]

    # ---- 9. Lot Area per House SF ----
    df["LotAreaPerSF"] = df["LotArea"] / (df["TotalHouseSF"] + 1)

    # ---- 10. Basement Ratio (finished / total) ----
    df["BasementRatio"] = (
        (df["BsmtFinSF1"] + df["BsmtFinSF2"]) / (df["TotalBsmtSF"] + 1)
    )

    return df