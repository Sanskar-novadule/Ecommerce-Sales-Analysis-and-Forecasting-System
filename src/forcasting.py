import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

PROCESSED_DATA_PATH = "data/sales_data_processed.csv"

df = pd.read_csv(PROCESSED_DATA_PATH)

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. CONVERT MONTH TO DATETIME
# ============================================================

df["Month"] = pd.to_datetime(df["Month"])


# ============================================================
# 3. SORT DATA BY MONTH
# ============================================================

df = df.sort_values("Month").reset_index(drop=True)


# ============================================================
# 4. CREATE INPUT FEATURES (X)
# ============================================================

X = df[
    [
        "Month",
        "Region",
        "Category",
        "Quantity",
        "Price"
    ]
].copy()


# ============================================================
# 5. CREATE TARGET VARIABLE (y)
# ============================================================

y = df["TotalSales"]


# ============================================================
# 6. CREATE YEAR AND MONTH NUMBER
# ============================================================

X["Year"] = X["Month"].dt.year
X["Month_Number"] = X["Month"].dt.month

# Remove original Month
X = X.drop(columns=["Month"])


# ============================================================
# 7. ONE-HOT ENCODE REGION AND CATEGORY
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded = encoder.fit_transform(
    X[["Region", "Category"]]
)

encoded_columns = encoder.get_feature_names_out(
    ["Region", "Category"]
)

encoded_df = pd.DataFrame(
    encoded,
    columns=encoded_columns,
    index=X.index
)


# ============================================================
# 8. COMBINE NUMERICAL + ENCODED FEATURES
# ============================================================

X = pd.concat(
    [
        X[
            [
                "Quantity",
                "Price",
                "Year",
                "Month_Number"
            ]
        ],
        encoded_df
    ],
    axis=1
)


# ============================================================
# 9. DISPLAY FINAL FEATURES
# ============================================================

print("\nFinal X:")
print(X.head())

print("\nFinal X Columns:")
print(X.columns.tolist())

print("\nFinal X Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)


# ============================================================
# 10. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n================ TRAIN / TEST SPLIT ================")

print("Training X:", X_train.shape)
print("Testing X :", X_test.shape)

print("Training y:", y_train.shape)
print("Testing y :", y_test.shape)

print("\nLast Training Month:")
print(df.iloc[split_index - 1]["Month"])

print("\nFirst Testing Month:")
print(df.iloc[split_index]["Month"])


# ============================================================
# 11. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

print("\nLinear Regression model trained successfully!")


# ============================================================
# 12. RANDOM FOREST REGRESSOR
# ============================================================

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest model trained successfully!")


# ============================================================
# 13. MAKE TEST PREDICTIONS
# ============================================================

linear_predictions = linear_model.predict(X_test)

random_forest_predictions = random_forest_model.predict(X_test)


# ============================================================
# 14. SHOW FIRST 5 TEST PREDICTIONS
# ============================================================

print("\n================ PREDICTIONS ================")

print("\nLinear Regression Predictions:")
print(linear_predictions[:5])

print("\nRandom Forest Predictions:")
print(random_forest_predictions[:5])

print("\nActual TotalSales:")
print(y_test.head().values)


# ============================================================
# 15. LINEAR REGRESSION METRICS
# ============================================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# ============================================================
# 16. RANDOM FOREST METRICS
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    random_forest_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    random_forest_predictions
)


# ============================================================
# 17. MODEL PERFORMANCE
# ============================================================

print("\n================ MODEL PERFORMANCE ================")

print("\nLinear Regression")
print("-------------------------")
print(f"MAE  : {linear_mae:.2f}")
print(f"RMSE : {linear_rmse:.2f}")
print(f"R²   : {linear_r2:.4f}")


print("\nRandom Forest")
print("-------------------------")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.4f}")


# ============================================================
# 18. R² AS PERCENTAGE
# ============================================================

linear_percentage = linear_r2 * 100
rf_percentage = rf_r2 * 100


print("\n================ MODEL PERFORMANCE % ================")

print(
    f"Linear Regression Performance: "
    f"{linear_percentage:.2f}%"
)

print(
    f"Random Forest Performance: "
    f"{rf_percentage:.2f}%"
)


# ============================================================
# 19. SELECT BEST MODEL
# ============================================================

if rf_r2 > linear_r2:

    print("\nBest Model: Random Forest Regressor")

else:

    print("\nBest Model: Linear Regression")