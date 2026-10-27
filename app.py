import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path


st.set_page_config(
    page_title="Personal Expense Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Personal Expense & Spending Predictor")

st.write(
    "Track your expenses, analyze spending habits, "
    "manage your budget, and predict future expenses using Machine Learning."
)


DATA_FILE = Path("cleaned_expense_data.csv")
CURRENT_FILE = Path("user_expenses.csv")



@st.cache_data
def load_historical_data():
    if not DATA_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "Date",
        "Amount",
        "Category",
        "Income/Expense"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in cleaned_expense_data.csv: "
            + ", ".join(missing_columns)
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["Amount"] = pd.to_numeric(
        data["Amount"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date", "Amount", "Category"]
    )

    data["Category"] = data["Category"].astype(str)

    data["Income/Expense"] = (
        data["Income/Expense"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = data[
        data["Income/Expense"] == "expense"
    ].copy()

    data["Data Source"] = "Historical Data"

    return data



def load_current_expenses():
    columns = [
        "Date",
        "Category",
        "Amount",
        "Income/Expense",
        "Note"
    ]

    if not CURRENT_FILE.exists():
        return pd.DataFrame(columns=columns)

    try:
        data = pd.read_csv(CURRENT_FILE)

        for column in columns:
            if column not in data.columns:
                data[column] = None

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        data["Amount"] = pd.to_numeric(
            data["Amount"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["Date", "Amount", "Category"]
        )

        data["Category"] = data["Category"].astype(str)

        return data[columns].copy()

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)


try:
    historical_expenses = load_historical_data()

except (ValueError, pd.errors.ParserError) as error:
    st.error(f"Dataset error: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load historical data: {error}")
    st.stop()


if historical_expenses.empty and not CURRENT_FILE.exists():
    st.error(
        "Historical expense data not found. "
        "Please check cleaned_expense_data.csv."
    )
    st.stop()


current_expenses = load_current_expenses()


st.header("➕ Add Present Expense")

if not historical_expenses.empty:
    available_categories = sorted(
        historical_expenses["Category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
else:
    available_categories = []

if "Other" not in available_categories:
    available_categories.append("Other")

with st.form("expense_entry_form", clear_on_submit=True):

    expense_date = st.date_input(
        "Expense Date",
        value=pd.Timestamp.today().date()
    )

    expense_category = st.selectbox(
        "Select Category",
        available_categories
    )

    if expense_category == "Other":
        custom_category = st.text_input(
            "Enter Category Name"
        )
    else:
        custom_category = ""

    expense_amount = st.number_input(
        "Expense Amount (₹)",
        min_value=1.0,
        value=100.0,
        step=50.0
    )

    expense_note = st.text_input(
        "Expense Note",
        placeholder="Example: Lunch, Bus ticket, Shopping"
    )

    add_expense_button = st.form_submit_button(
        "Add Expense"
    )


if add_expense_button:

    final_category = (
        custom_category.strip()
        if expense_category == "Other"
        else expense_category
    )

    if not final_category:
        st.error("Please enter a category name.")

    elif pd.Timestamp(expense_date) > pd.Timestamp.today().normalize():
        st.error("Please select today or an earlier date.")

    else:
        new_expense = pd.DataFrame([{
            "Date": pd.Timestamp(expense_date),
            "Category": final_category,
            "Amount": float(expense_amount),
            "Income/Expense": "Expense",
            "Note": expense_note
        }])

        updated_expenses = pd.concat(
            [
                current_expenses,
                new_expense
            ],
            ignore_index=True
        )

        try:
            updated_expenses.to_csv(
                CURRENT_FILE,
                index=False
            )

            st.success(
                f"Expense of ₹{expense_amount:,.2f} saved successfully!"
            )

            st.rerun()

        except OSError as error:
            st.error(
                f"Unable to save expense: {error}"
            )



current_expenses = load_current_expenses()

if not historical_expenses.empty:
    historical_expenses["Data Source"] = "Historical Data"

if not current_expenses.empty:
    current_expenses["Data Source"] = "Current Entries"

all_expenses = pd.concat(
    [
        historical_expenses,
        current_expenses
    ],
    ignore_index=True
)

if all_expenses.empty:
    st.warning("No expense records available yet.")
    st.stop()

st.sidebar.header("⚙️ Settings")

source_options = [
    "All Expenses",
    "Historical Data",
    "Current Entries"
]

selected_source = st.sidebar.selectbox(
    "Choose Data Source",
    source_options
)

if selected_source == "Historical Data":
    expenses = all_expenses[
        all_expenses["Data Source"] == "Historical Data"
    ].copy()

elif selected_source == "Current Entries":
    expenses = all_expenses[
        all_expenses["Data Source"] == "Current Entries"
    ].copy()

else:
    expenses = all_expenses.copy()


categories = sorted(
    expenses["Category"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_categories = st.sidebar.multiselect(
    "Filter Categories",
    categories,
    default=categories
)

budget = st.sidebar.number_input(
    "Monthly Budget (₹)",
    min_value=0.0,
    value=10000.0,
    step=500.0
)

forecast_days = st.sidebar.select_slider(
    "Forecast Days",
    options=[7, 30, 90],
    value=7
)

expenses = expenses[
    expenses["Category"].isin(selected_categories)
].copy()


if expenses.empty:
    st.warning(
        "No expenses match your selected filters. "
        "Please select at least one category."
    )
    st.stop()


expenses["Date"] = pd.to_datetime(
    expenses["Date"],
    errors="coerce"
)

expenses["Amount"] = pd.to_numeric(
    expenses["Amount"],
    errors="coerce"
)
expenses["Date"] = pd.to_datetime(expenses["Date"], errors="coerce")
expenses["Amount"] = pd.to_numeric(expenses["Amount"], errors="coerce")
expenses = expenses.dropna(subset=["Date", "Amount"]).copy()
expenses = expenses.dropna(
    subset=["Date", "Amount", "Category"]
).copy()
st.header("📊 Expense Dashboard")

total_expense = expenses["Amount"].sum()

total_transactions = len(expenses)

daily_expenses = (
    expenses.groupby(
        expenses["Date"].dt.normalize()
    )["Amount"]
    .sum()
    .sort_index()
)

average_daily = daily_expenses.mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Expenses",
    f"₹{total_expense:,.2f}"
)

col2.metric(
    "Total Transactions",
    f"{total_transactions}"
)

col3.metric(
    "Average Expense per Recorded Day",
    f"₹{average_daily:,.2f}"
)

st.caption(
    "Historical totals include old records. "
    "They should not be interpreted as current-month spending."
)



st.header("📌 Expenses by Category")

category_expenses = (
    expenses.groupby("Category")["Amount"]
    .sum()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

category_expenses.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Category")
ax.set_ylabel("Expense Amount (₹)")
ax.set_title("Total Expenses by Category")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

if not category_expenses.empty:

    highest_category = category_expenses.idxmax()

    highest_amount = category_expenses.max()

    st.success(
        f"Highest Spending Category: {highest_category} "
        f"(₹{highest_amount:,.2f})"
    )


st.header("📅 Monthly Expense Analysis")

monthly_expenses = (
    expenses.groupby(
        expenses["Date"].dt.to_period("M")
    )["Amount"]
    .sum()
)

monthly_expenses.index = monthly_expenses.index.astype(str)

fig, ax = plt.subplots(figsize=(10, 5))

monthly_expenses.plot(
    kind="line",
    marker="o",
    ax=ax
)

ax.set_title("Monthly Expense Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Expense Amount (₹)")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    monthly_expenses.rename(
        "Total Expense (₹)"
    ).reset_index(names="Month"),
    use_container_width=True
)


st.header("💳 Monthly Budget Planner")

today = pd.Timestamp.today().normalize()

current_month = today.to_period("M")

# Budget uses only expenses entered by the user,
# not old historical records.
user_expenses = load_current_expenses()

if not user_expenses.empty:

    user_expenses["Date"] = pd.to_datetime(
        user_expenses["Date"],
        errors="coerce"
    )

    user_expenses["Amount"] = pd.to_numeric(
        user_expenses["Amount"],
        errors="coerce"
    )

    user_expenses = user_expenses.dropna(
        subset=["Date", "Amount"]
    )

    user_expenses_this_month = user_expenses[
        user_expenses["Date"].dt.to_period("M") == current_month
    ]

    current_month_expenses = user_expenses_this_month[
        user_expenses_this_month["Category"].isin(
            selected_categories
        )
    ]["Amount"].sum()

else:
    current_month_expenses = 0.0


col1, col2, col3 = st.columns(3)

col1.metric(
    "Your Expenses This Month",
    f"₹{current_month_expenses:,.2f}"
)

remaining_budget = budget - current_month_expenses

col2.metric(
    "Remaining Budget",
    f"₹{remaining_budget:,.2f}"
)

if budget > 0:
    budget_used = (
        current_month_expenses / budget
    ) * 100
else:
    budget_used = 0.0

col3.metric(
    "Budget Used",
    f"{budget_used:.1f}%"
)

if remaining_budget < 0:

    st.error(
        f"⚠️ Budget exceeded by ₹{abs(remaining_budget):,.2f}!"
    )

elif budget > 0:

    st.success(
        "Your expenses are within your monthly budget."
    )

else:

    st.info(
        "Set a budget greater than ₹0 to track your spending limit."
    )

if budget > 0:

    st.progress(
        min(
            max(current_month_expenses / budget, 0.0),
            1.0
        )
    )



st.header("🤖 Future Expense Prediction")

st.write(
    "The model uses your recently entered expenses "
    "to estimate future spending."
)

if user_expenses.empty:

    st.warning(
        "You have not entered any current expenses yet. "
        "Add your expenses above to enable current-spending predictions. "
        "The historical 2022 dataset is not used as current spending."
    )

else:

    # Use recent user-entered expense records only.
    recent_start = today - pd.Timedelta(days=89)

    recent_expenses = user_expenses[
        (user_expenses["Date"] >= recent_start)
        & (user_expenses["Date"] <= today)
    ].copy()

    if recent_expenses.empty:

        st.warning(
            "No current expense records were found in the last 90 days. "
            "Add recent expenses before generating a current forecast."
        )

    else:

        # Aggregate expenses by date.
        recent_daily = (
            recent_expenses.groupby(
                recent_expenses["Date"].dt.normalize()
            )["Amount"]
            .sum()
            .sort_index()
        )

        # Include dates between the first recent entry and today.
        # Missing dates are treated as zero; if you did not record
        # all expenses, the prediction may be less accurate.
        full_dates = pd.date_range(
            start=recent_daily.index.min(),
            end=today,
            freq="D"
        )

        recent_daily = recent_daily.reindex(
            full_dates,
            fill_value=0
        )

        recent_daily.index.name = "Date"

        number_of_days = len(recent_daily)

        if number_of_days < 2:

            st.warning(
                "At least two days of expense history are required "
                "to train the prediction model."
            )

        else:

            X = pd.DataFrame({
                "Day": range(number_of_days)
            })

            y = recent_daily.values

            model = LinearRegression()

            model.fit(X, y)

            future_X = pd.DataFrame({
                "Day": range(
                    number_of_days,
                    number_of_days + forecast_days
                )
            })

            predictions = model.predict(future_X)

            # Expenses cannot be negative.
            predictions = predictions.clip(min=0)

            future_dates = pd.date_range(
                start=today + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="D"
            )

            forecast = pd.DataFrame({
                "Date": future_dates,
                "Predicted Expense (₹)": predictions.round(2)
            })

            st.subheader(
                f"Predicted Expenses for the Next {forecast_days} Days"
            )

            st.dataframe(
                forecast,
                use_container_width=True
            )

            total_prediction = predictions.sum()

            st.metric(
                f"Total Predicted Spending ({forecast_days} Days)",
                f"₹{total_prediction:,.2f}"
            )

            st.caption(
                "This is an estimate from a simple Linear Regression model, "
                "not a guarantee. Predictions may be unreliable when there "
                "are few records or missing expense entries."
            )

            # Historical vs forecast graph
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                recent_daily.index,
                recent_daily.values,
                label="Recently Recorded Expenses"
            )

            ax.plot(
                future_dates,
                predictions,
                marker="o",
                label="Predicted Expenses"
            )

            ax.set_title(
                "Recent Expenses vs Future Prediction"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Expense Amount (₹)")

            ax.legend()
            ax.grid(True)

            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # Download forecast
            st.download_button(
                label="📥 Download Forecast CSV",
                data=forecast.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="future_expense_forecast.csv",
                mime="text/csv"
            )



st.header("🧾 Expense Transaction Details")

display_expenses = expenses.sort_values(
    "Date",
    ascending=False
).copy()

st.dataframe(
    display_expenses,
    use_container_width=True
)

st.download_button(
    label="📥 Download Expense Data",
    data=display_expenses.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="expense_data.csv",
    mime="text/csv"
)



st.header("📂 Your Current Expense Records")

if not user_expenses.empty:

    st.dataframe(
        user_expenses.sort_values(
            "Date",
            ascending=False
        ),
        use_container_width=True
    )

    st.download_button(
        label="📥 Download My Current Expenses",
        data=user_expenses.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="my_current_expenses.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No current expenses entered yet. "
        "Use the form at the top to add your first expense."
    )



st.divider()

st.caption(
    "IML Mini Project | Personal Expense & Spending Predictor"
)
