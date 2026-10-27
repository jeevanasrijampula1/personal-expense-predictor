import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Personal Expense Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Personal Expense & Spending Predictor")
st.write("Analyze your expenses and predict future spending using Machine Learning.")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_expense_data.csv")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df = df.dropna(subset=["Date", "Amount", "Category"])

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("cleaned_expense_data.csv not found. Keep it in the same folder as app.py.")
    st.stop()

expenses = df[
    df["Income/Expense"].astype(str).str.strip().str.lower() == "expense"
].copy()

if expenses.empty:
    st.warning("No expense records found in the dataset.")
    st.stop()

st.sidebar.header("Settings")

categories = sorted(expenses["Category"].unique())

selected_categories = st.sidebar.multiselect(
    "Select Categories",
    categories,
    default=categories
)

budget = st.sidebar.number_input(
    "Monthly Budget (₹)",
    min_value=0.0,
    value=10000.0,
    step=500.0
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=7,
    max_value=30,
    value=7
)

expenses = expenses[
    expenses["Category"].isin(selected_categories)
].copy()

if expenses.empty:
    st.warning("Please select at least one category.")
    st.stop()

st.header("📊 Expense Dashboard")

total_expense = expenses["Amount"].sum()
total_transactions = len(expenses)

daily_expenses = (
    expenses.groupby(expenses["Date"].dt.normalize())["Amount"]
    .sum()
    .sort_index()
)

average_daily = daily_expenses.mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Expenses", f"₹{total_expense:,.2f}")
col2.metric("Total Transactions", total_transactions)
col3.metric("Average Daily Expense", f"₹{average_daily:,.2f}")

st.header("📌 Expenses by Category")

category_expenses = expenses.groupby("Category")["Amount"].sum()
category_expenses = category_expenses.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
category_expenses.plot(kind="bar", ax=ax)

ax.set_xlabel("Category")
ax.set_ylabel("Amount (₹)")
ax.set_title("Total Expenses by Category")

plt.xticks(rotation=35)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

if not category_expenses.empty:
    st.success(
        f"Highest Spending Category: {category_expenses.idxmax()} "
        f"(₹{category_expenses.max():,.2f})"
    )

st.header("📅 Monthly Expense Analysis")

monthly_expenses = (
    expenses.groupby(expenses["Date"].dt.to_period("M"))["Amount"].sum()
)

monthly_expenses.index = monthly_expenses.index.astype(str)

fig, ax = plt.subplots(figsize=(9, 4))
monthly_expenses.plot(kind="line", marker="o", ax=ax)

ax.set_title("Monthly Expense Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Amount (₹)")

plt.xticks(rotation=35)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.header("🤖 Machine Learning Expense Prediction")

daily_expenses = daily_expenses.asfreq("D", fill_value=0)

if len(daily_expenses) < 2:
    st.warning("At least two days of data are required for prediction.")
else:
    X = list(range(len(daily_expenses)))
    X = pd.DataFrame({"Day": X})

    y = daily_expenses.values

    model = LinearRegression()
    model.fit(X, y)

    future_X = pd.DataFrame({
        "Day": range(len(daily_expenses), len(daily_expenses) + forecast_days)
    })

    predictions = model.predict(future_X)
    predictions = predictions.clip(min=0)

    future_dates = pd.date_range(
        start=daily_expenses.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted Expense (₹)": predictions.round(2)
    })

    st.subheader(f"Predicted Expenses for the Next {forecast_days} Days")

    st.dataframe(forecast, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        daily_expenses.index,
        daily_expenses.values,
        label="Historical Expenses"
    )

    ax.plot(
        future_dates,
        predictions,
        marker="o",
        label="Predicted Expenses"
    )

    ax.set_title("Historical vs Predicted Expenses")
    ax.set_xlabel("Date")
    ax.set_ylabel("Expense (₹)")
    ax.legend()
    ax.grid(True)

    plt.xticks(rotation=35)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.metric(
        "Total Predicted Expenses",
        f"₹{predictions.sum():,.2f}"
    )

    st.download_button(
        "Download Forecast CSV",
        forecast.to_csv(index=False).encode("utf-8"),
        "future_expense_forecast.csv",
        "text/csv"
    )

st.header("💳 Monthly Budget Planner")

current_month = pd.Timestamp.today().to_period("M")

current_month_expenses = expenses[
    expenses["Date"].dt.to_period("M") == current_month
]["Amount"].sum()

col1, col2 = st.columns(2)

col1.metric(
    "Expenses This Month",
    f"₹{current_month_expenses:,.2f}"
)

remaining = budget - current_month_expenses

col2.metric(
    "Remaining Budget",
    f"₹{remaining:,.2f}"
)

if remaining < 0:
    st.error("Your expenses have exceeded your budget!")
else:
    st.success("Your expenses are within your budget.")

if budget > 0:
    st.progress(min(current_month_expenses / budget, 1.0))
else:
    st.progress(0.0)

st.header("🧾 Expense Transaction Details")

st.dataframe(
    expenses.sort_values("Date", ascending=False),
    use_container_width=True
)

st.download_button(
    "Download Expense Data",
    expenses.to_csv(index=False).encode("utf-8"),
    "expense_data.csv",
    "text/csv"
)

st.divider()

st.caption("IML Mini Project | Personal Expense & Spending Predictor")

