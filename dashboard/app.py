import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=" sanskar ajmera E-Commerce Sales Dashboard",
    
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/sales_data_processed.csv"

df = pd.read_csv(DATA_PATH)

df["Month"] = pd.to_datetime(df["Month"])


# ============================================================
# TITLE
# ============================================================

st.title("🛒 E-Commerce Sales Analysis & Forecasting Dashboard")

st.markdown("---")


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_revenue = df["TotalSales"].sum()

total_orders = len(df)

top_region = (
    df.groupby("Region")["TotalSales"]
    .sum()
    .idxmax()
)

top_category = (
    df.groupby("Category")["TotalSales"]
    .sum()
    .idxmax()
)


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Total Revenue",
        f"${total_revenue:,.2f}"
    )

with col2:
    st.metric(
        "📦 Total Orders",
        total_orders
    )

with col3:
    st.metric(
        "🌍 Top Region",
        top_region
    )

with col4:
    st.metric(
        "🏆 Top Category",
        top_category
    )


st.markdown("---")


# ============================================================
# ROW 1
# SALES BY REGION + MONTHLY SALES
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# SALES BY REGION
# ============================================================

with col1:

    st.subheader("Sales by Region")

    region_sales = (
        df.groupby("Region")["TotalSales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(5, 3))

    region_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales")

    plt.xticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# MONTHLY SALES
# ============================================================

with col2:

    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        df.groupby("Month")["TotalSales"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(
        monthly_sales.index,
        monthly_sales.values,
        marker="o"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# ROW 2
# CATEGORY PIE + CORRELATION HEATMAP
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# CATEGORY PIE
# ============================================================

with col1:

    st.subheader("Sales by Category")

    category_sales = (
        df.groupby("Category")["TotalSales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.pie(
        category_sales.values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
        startangle=90,
        pctdistance=0.70
    )

    ax.set_title("Category Sales Distribution")

    ax.legend(
        category_sales.index,
        title="Category",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

with col2:

    st.subheader("Correlation Heatmap")

    numeric_columns = [
        "Quantity",
        "Price",
        "TotalSales"
    ]

    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(5, 3.5))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Sales Correlation")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# SALES FORECAST SECTION
# ============================================================

st.markdown("---")

st.subheader("📈 Sales Forecast")

st.info(
    "Sales forecasting model will be connected here "
    "after completing the Machine Learning model."
)


# ============================================================
# DATA TABLE
# ============================================================

st.markdown("---")

st.subheader("📋 Sales Data")

st.dataframe(
    df,
    use_container_width=True
)