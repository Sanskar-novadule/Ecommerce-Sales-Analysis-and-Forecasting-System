import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "data/sales_data_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 4. CONVERT MONTH TO DATETIME
# ==========================================

df["Month"] = pd.to_datetime(df["Month"])

print("\nMonth datatype:")
print(df["Month"].dtype)


# ==========================================
# 5. CHECK DATA TYPES
# ==========================================

print("\nData types:")
print(df.dtypes)


# ==========================================
# 6. FINAL DATA
# ==========================================

print("\nFinal data:")
print(df.head())

# ==========================================
# 7. FEATURE ENGINEERING
# ==========================================

df["Year"] = df["Month"].dt.year
df["Month_Number"] = df["Month"].dt.month

print("\nNew features:")
print(df[["Month", "Year", "Month_Number"]].head())

# ==========================================
# 8. VERIFY TOTAL SALES
# ==========================================

df["CalculatedSales"] = df["Quantity"] * df["Price"]

df["SalesDifference"] = (
    df["TotalSales"] - df["CalculatedSales"]
)

print("\nSales verification:")
print(
    df[
        ["Quantity", "Price", "TotalSales",
         "CalculatedSales", "SalesDifference"]
    ].head(10)
)

print("\nMaximum sales difference:")
print(df["SalesDifference"].abs().max())

# ==========================================
# 9. SAVE PROCESSED DATA
# ==========================================
df.drop(columns=["CalculatedSales", "SalesDifference"], inplace=True)
OUTPUT_PATH = "data/sales_data_processed.csv"

print("\nFinal columns before saving:")
print(df.columns.tolist())


df.to_csv(OUTPUT_PATH, index=False)

print(f"\nProcessed data saved to: {OUTPUT_PATH}")