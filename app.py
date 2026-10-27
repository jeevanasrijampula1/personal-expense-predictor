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
        return pd.DataFrame(
            columns=[
                "Date",
                "Amount",
                "Category",
                "Income/Expense",
                "Note",
                "Data Source"
            ]
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "Date",
        "Amount",
        "Category",
        "Income/Expense"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in cleaned_expense_data.csv: "
            + ", ".join(missing_columns)
        )

    if "Note" not in data.columns:
        data["Note"] = ""

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
    ).copy()

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
        "Expense ID",
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

        if data.empty:
            return pd.DataFrame(columns=columns)

        for column in columns:

            if column not in data.columns:

                if column == "Expense ID":

                    data[column] = range(
                        1,
                        len(data) + 1
                    )

                elif column == "Income/Expense":

                    data[column] = "Expense"

                else:

                    data[column] = ""

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        data["Amount"] = pd.to_numeric(
            data["Amount"],
            errors="coerce"
        )

        data["Expense ID"] = pd.to_numeric(
            data["Expense ID"],
            errors="coerce"
        )

        data = data.dropna(
            subset=[
                "Expense ID",
                "Date",
                "Amount",
                "Category"
            ]
        ).copy()

        data["Expense ID"] = (
            data["Expense ID"].astype(int)
        )

        data["Category"] = data["Category"].astype(str)

        data = data[
            data["Income/Expense"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "expense"
        ].copy()

        return data[columns].copy()

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError
    ):

        return pd.DataFrame(columns=columns)



def save_current_expenses(data):

    columns = [
        "Expense ID",
        "Date",
        "Category",
        "Amount",
        "Income/Expense",
        "Note"
    ]

    data = data.copy()

    for column in columns:

        if column not in data.columns:

            data[column] = ""

    data = data[columns]

    data.to_csv(
        CURRENT_FILE,
        index=False
    )


try:

    historical_expenses = load_historical_data()

except Exception as error:

    st.error(
        f"Unable to load historical data: {error}"
    )

    st.stop()


current_expenses = load_current_expenses()


st.header("➕ Add Present Expense")

historical_categories = []

if not historical_expenses.empty:

    historical_categories = (
        historical_expenses["Category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

current_categories = []

if not current_expenses.empty:

    current_categories = (
        current_expenses["Category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

available_categories = sorted(
    set(historical_categories + current_categories)
)

if "Other" not in available_categories:

    available_categories.append("Other")


with st.form(
    "expense_entry_form",
    clear_on_submit=True
):

    expense_date = st.date_input(
        "Expense Date",
        value=pd.Timestamp.today().date()
    )

    expense_category = st.selectbox(
        "Select Category",
        available_categories
    )

    custom_category = ""

    if expense_category == "Other":

        custom_category = st.text_input(
            "Enter Category Name"
        )

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

        st.error(
            "Please enter a category name."
        )

    elif pd.Timestamp(expense_date) > pd.Timestamp.today().normalize():

        st.error(
            "Please select today or an earlier date."
        )

    else:

        current_expenses = load_current_expenses()

        if current_expenses.empty:

            next_id = 1

        else:

            next_id = (
                int(current_expenses["Expense ID"].max()) + 1
            )

        new_expense = pd.DataFrame([{
            "Expense ID": next_id,
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

            save_current_expenses(updated_expenses)

            st.success(
                f"Expense of ₹{expense_amount:,.2f} saved successfully!"
            )

            st.rerun()

        except OSError as error:

            st.error(
                f"Unable to save expense: {error}"
            )



current_expenses = load_current_expenses()

historical_expenses = historical_expenses.copy()

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

    st.info(
        "No expense records available. Add your first expense above."
    )

    st.stop()


all_expenses["Date"] = pd.to_datetime(
    all_expenses["Date"],
    errors="coerce"
)

all_expenses["Amount"] = pd.to_numeric(
    all_expenses["Amount"],
    errors="coerce"
)

all_expenses = all_expenses.dropna(
    subset=["Date", "Amount", "Category"]
).copy()


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

if expenses.empty:

    st.warning(
        "No records available for the selected data source."
    )

    st.stop()


min_date = expenses["Date"].min().date()
max_date = expenses["Date"].max().date()

date_range = st.sidebar.date_input(
    "Filter Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

budget = st.sidebar.number_input(
    "Monthly Budget (₹)",
    min_value=0.0,
    value=25000.0,
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

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    start_date = pd.Timestamp(date_range[0])

    end_date = (
        pd.Timestamp(date_range[1])
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )

    expenses = expenses[
        (expenses["Date"] >= start_date)
        & (expenses["Date"] <= end_date)
    ].copy()


if expenses.empty:

    st.warning(
        "No expenses match the selected filters. "
        "Change the category or date range."
    )

    st.stop()



st.header("📊 Expense Dashboard")

total_expense = expenses["Amount"].sum()

total_transactions = len(expenses)

daily_expenses = (
    expenses.groupby(
        expenses["Date"].dt.normalize()
    )["Amount"]
    .sum()
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
    "Dashboard totals depend on the selected data source and filters. "
    "Historical totals include old records and are not current-month totals."
)


st.header("📌 Expenses by Category")

category_expenses = (
    expenses.groupby("Category")["Amount"]
    .sum()
    .sort_values(ascending=False)
)

if not category_expenses.empty:

    fig, ax = plt.subplots(figsize=(10, 5))

    category_expenses.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("Expense Amount (₹)")
    ax.set_title("Total Expenses by Category")

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

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

monthly_table = monthly_expenses.rename(
    "Total Expense (₹)"
).reset_index()

monthly_table.columns = [
    "Month",
    "Total Expense (₹)"
]

monthly_table["Month"] = (
    monthly_table["Month"].astype(str)
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    monthly_table["Month"],
    monthly_table["Total Expense (₹)"],
    marker="o"
)

ax.set_title("Monthly Expense Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Expense Amount (₹)")

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

st.dataframe(
    monthly_table,
    use_container_width=True
)


st.header("📊 Monthly Spending Comparison")

today = pd.Timestamp.today().normalize()

this_month = today.to_period("M")

previous_month = this_month - 1

# Compare current user-entered records only.
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
    ).copy()

    user_expenses = user_expenses[
        user_expenses["Category"].isin(selected_categories)
    ].copy()

    this_month_total = user_expenses[
        user_expenses["Date"].dt.to_period("M") == this_month
    ]["Amount"].sum()

    previous_month_total = user_expenses[
        user_expenses["Date"].dt.to_period("M") == previous_month
    ]["Amount"].sum()

else:

    this_month_total = 0.0
    previous_month_total = 0.0


col1, col2 = st.columns(2)

col1.metric(
    "Current Month Spending",
    f"₹{this_month_total:,.2f}"
)

col2.metric(
    "Previous Month Spending",
    f"₹{previous_month_total:,.2f}"
)

difference = this_month_total - previous_month_total

if difference > 0:

    st.warning(
        f"Spending is ₹{difference:,.2f} higher than last month."
    )

elif difference < 0:

    st.success(
        f"Spending is ₹{abs(difference):,.2f} lower than last month."
    )

else:

    st.info(
        "Spending is equal to the previous month."
    )



st.header("💳 Monthly Budget Planner")

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
    ).copy()

    current_month_expenses = user_expenses[
        (
            user_expenses["Date"].dt.to_period("M")
            == this_month
        )
        & (
            user_expenses["Category"].isin(
                selected_categories
            )
        )
    ]["Amount"].sum()

else:

    current_month_expenses = 0.0


remaining_budget = budget - current_month_expenses

if budget > 0:

    budget_used = (
        current_month_expenses / budget
    ) * 100

else:

    budget_used = 0.0


col1, col2, col3 = st.columns(3)

col1.metric(
    "Your Expenses This Month",
    f"₹{current_month_expenses:,.2f}"
)

col2.metric(
    "Remaining Budget",
    f"₹{remaining_budget:,.2f}"
)

col3.metric(
    "Budget Used",
    f"{budget_used:.1f}%"
)


if budget > 0:

    st.progress(
        min(
            max(current_month_expenses / budget, 0.0),
            1.0
        )
    )

    if remaining_budget < 0:

        st.error(
            f"⚠️ Budget exceeded by ₹{abs(remaining_budget):,.2f}!"
        )

    elif budget_used >= 80:

        st.warning(
            "⚠️ You have used at least 80% of your monthly budget."
        )

    else:

        st.success(
            "Your expenses are within your monthly budget."
        )

else:

    st.info(
        "Set a monthly budget greater than ₹0 to track your spending."
    )

st.header("🤖 Future Expense Prediction")

st.write(
    "Linear Regression estimates future daily spending "
    "using recent user-entered expense records."
)

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
    ).copy()

    user_expenses = user_expenses[
        (user_expenses["Amount"] >= 0)
        & (
            user_expenses["Date"]
            <= today
        )
        & (
            user_expenses["Category"].isin(
                selected_categories
            )
        )
    ].copy()


if user_expenses.empty:

    st.warning(
        "Add current expenses in your selected categories "
        "to enable prediction."
    )

else:

    recent_start = today - pd.Timedelta(days=29)

    recent_expenses = user_expenses[
        (user_expenses["Date"] >= recent_start)
        & (user_expenses["Date"] <= today)
    ].copy()

    if recent_expenses.empty:

        st.warning(
            "No expenses were recorded in the last 30 days. "
            "Add recent records before predicting future expenses."
        )

    else:

        # Sum recorded expenses by day.
        recent_daily = (
            recent_expenses.groupby(
                recent_expenses["Date"].dt.normalize()
            )["Amount"]
            .sum()
            .sort_index()
        )

        # A date with no entry is unknown, not necessarily zero.
        # Therefore, use recorded days for the baseline.
        recorded_dates = recent_daily.index

        if len(recorded_dates) < 2:

            st.warning(
                "At least two different dates with expense entries "
                "are required for Linear Regression."
            )

        else:

            # Use elapsed days from the first recorded date.
            X = (
                recorded_dates - recorded_dates.min()
            ).days.to_numpy().reshape(-1, 1)

            y = recent_daily.to_numpy()

            model = LinearRegression()

            model.fit(X, y)

            future_dates = pd.date_range(
                start=today + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="D"
            )

            future_X = (
                future_dates - recorded_dates.min()
            ).days.to_numpy().reshape(-1, 1)

            predictions = model.predict(future_X)

            predictions = predictions.clip(min=0)

            # Guard against extreme extrapolation.
            # This cap is a practical safeguard, not a guarantee
            # of prediction accuracy.
            recent_average = float(recent_daily.mean())

            if recent_average > 0:

                daily_cap = recent_average * 3

                predictions = predictions.clip(
                    max=daily_cap
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

            total_prediction = float(predictions.sum())

            st.metric(
                f"Estimated Spending ({forecast_days} Days)",
                f"₹{total_prediction:,.2f}"
            )

            st.caption(
                "This is a Linear Regression estimate based on "
                "recent recorded spending. Predictions are limited "
                "to three times the recent average per recorded day "
                "to reduce extreme extrapolation. The limit does not "
                "guarantee accuracy."
            )

            if len(recorded_dates) < 7:

                st.warning(
                    "Prediction may be unreliable because fewer than "
                    "7 different dates contain expense records."
                )

            # Plot recorded expenses and predicted expenses.
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                recent_daily.index,
                recent_daily.values,
                marker="o",
                label="Recorded Daily Expenses"
            )

            ax.plot(
                future_dates,
                predictions,
                marker="o",
                label="Predicted Expenses"
            )

            ax.set_title(
                "Recent Recorded Expenses vs Future Prediction"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Expense Amount (₹)")

            ax.legend()
            ax.grid(True)

            plt.xticks(
                rotation=35,
                ha="right"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.download_button(
                label="📥 Download Forecast CSV",
                data=forecast.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="future_expense_forecast.csv",
                mime="text/csv"
            )



st.header("✏️ Manage Current Expenses")

user_expenses = load_current_expenses()

if user_expenses.empty:

    st.info(
        "No current expenses available to edit or delete."
    )

else:

    search_text = st.text_input(
        "🔎 Search by category or note"
    )

    manageable_expenses = user_expenses.copy()

    if search_text.strip():

        search_text = search_text.strip()

        matching_rows = (
            manageable_expenses["Category"]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False
            )
            |
            manageable_expenses["Note"]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        )

        manageable_expenses = manageable_expenses[
            matching_rows
        ].copy()

    st.dataframe(
        manageable_expenses.sort_values(
            "Date",
            ascending=False
        ),
        use_container_width=True
    )

    if not manageable_expenses.empty:

        selected_id = st.selectbox(
            "Select Expense ID to edit or delete",
            manageable_expenses["Expense ID"].tolist()
        )

        selected_record = user_expenses[
            user_expenses["Expense ID"] == selected_id
        ].iloc[0]

        with st.expander("✏️ Edit Selected Expense"):

            with st.form("edit_expense_form"):

                edit_date = st.date_input(
                    "Expense Date",
                    value=selected_record["Date"].date()
                )

                edit_category = st.text_input(
                    "Category",
                    value=str(selected_record["Category"])
                )

                edit_amount = st.number_input(
                    "Amount (₹)",
                    min_value=0.01,
                    value=max(
                        float(selected_record["Amount"]),
                        0.01
                    ),
                    step=10.0
                )

                edit_note = st.text_input(
                    "Note",
                    value=str(selected_record["Note"])
                )

                update_button = st.form_submit_button(
                    "Update Expense"
                )

            if update_button:

                if not edit_category.strip():

                    st.error(
                        "Category cannot be empty."
                    )

                elif pd.Timestamp(edit_date) > today:

                    st.error(
                        "Expense date cannot be in the future."
                    )

                else:

                    updated_records = user_expenses.copy()

                    row_mask = (
                        updated_records["Expense ID"]
                        == selected_id
                    )

                    updated_records.loc[
                        row_mask,
                        "Date"
                    ] = pd.Timestamp(edit_date)

                    updated_records.loc[
                        row_mask,
                        "Category"
                    ] = edit_category.strip()

                    updated_records.loc[
                        row_mask,
                        "Amount"
                    ] = float(edit_amount)

                    updated_records.loc[
                        row_mask,
                        "Note"
                    ] = edit_note

                    try:

                        save_current_expenses(
                            updated_records
                        )

                        st.success(
                            "Expense updated successfully!"
                        )

                        st.rerun()

                    except OSError as error:

                        st.error(
                            f"Unable to update expense: {error}"
                        )

        with st.expander("🗑️ Delete Selected Expense"):

            st.write(
                f"Selected amount: ₹{selected_record['Amount']:,.2f}"
            )

            confirm_delete = st.checkbox(
                "I confirm that I want to delete this expense.",
                key=f"confirm_delete_{selected_id}"
            )

            if st.button(
                "Delete Expense",
                key=f"delete_button_{selected_id}"
            ):

                if not confirm_delete:

                    st.warning(
                        "Please confirm before deleting."
                    )

                else:

                    remaining_records = user_expenses[
                        user_expenses["Expense ID"] != selected_id
                    ].copy()

                    try:

                        save_current_expenses(
                            remaining_records
                        )

                        st.success(
                            "Expense deleted successfully!"
                        )

                        st.rerun()

                    except OSError as error:

                        st.error(
                            f"Unable to delete expense: {error}"
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
    label="📥 Download Filtered Expense Data",
    data=display_expenses.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="expense_data.csv",
    mime="text/csv"
)


st.header("📂 Your Current Expense Records")

user_expenses = load_current_expenses()

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
        "Use the form above to add your first expense."
    )

st.divider()

st.caption(
    "IML Mini Project | Personal Expense & Spending Predictor"
)
