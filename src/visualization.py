import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROCESSED_DATA_PATH = "data/sales_data_processed.csv"
df = pd.read_csv(PROCESSED_DATA_PATH)

region_sales = df.groupby("Region")["TotalSales"].sum()

print("Total sales by region:")
print(region_sales)

region_sales.plot(kind="bar", color="skyblue")
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")   
plt.tight_layout()
plt.savefig("output/sales_by_region.png")
plt.close()

Monthly_sales = df.groupby("Month")["TotalSales"].sum()
print("\nTotal sales by month:")
print(Monthly_sales)

Monthly_sales.plot(kind="line", marker="o", color="orange")
plt.title("Total Sales by Month")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/sales_by_month.png")
plt.close()

# ==========================================
# CATEGORY SALES - PIE CHART
# ==========================================

category_sales = df.groupby("Category")["TotalSales"].sum()

category_sales.plot(kind="pie", autopct="%1.1f%%")

plt.ylabel("Sales")
plt.title("Sales Distribution by Category")
plt.savefig("output/category_sales_pie.png")
plt.close()

# ==========================================
# 5. CORRELATION HEATMAP
# ==========================================

numeric_columns = [
    "Quantity",
    "Price",
    "TotalSales"
    
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("output/correlation_heatmap.png")

plt.show()
plt.close()