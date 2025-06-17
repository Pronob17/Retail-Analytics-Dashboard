import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tests.errorlog import log_error


class BasicGraphsClass:
    def __init__(self, basic_analytics_df, selection):
        # initializing a variable called data
        self.selection = selection
        self.data = None
        # get today's time
        self.today = pd.Timestamp.today()

        # Optional placeholder empty figure
        self.empty_fig = go.Figure()
        self.empty_fig.update_layout(
            title="No Data Available",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "No chart to display",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )

        # condition to check for Date column
        if "Date" not in basic_analytics_df.columns:
            print("Missing 'Date' column. Graphs may not render correctly.")
            basic_analytics_df["Date"] = pd.NaT

        # verifying again that the data is in datetime
        basic_analytics_df["Date"] = pd.to_datetime(basic_analytics_df["Date"], errors='coerce')

        # Filter for current year
        yearly_data = basic_analytics_df[basic_analytics_df["Date"].dt.year == self.today.year]

        # Filter for current quarter
        quarterly_data = yearly_data[
            yearly_data["Date"].dt.quarter == self.today.quarter
            ]

        # Filter for current month
        monthly_data = quarterly_data[quarterly_data["Date"].dt.month == self.today.month]

        # Filter for current week (by weekday) — this line is incorrect in your version
        # Fix: Use `.dt.weekday` instead of `.dt.month` for comparison with `self.today.weekday()`
        start_of_week = self.today - pd.Timedelta(days=self.today.weekday())
        end_of_week = start_of_week + pd.Timedelta(days=6)

        weekly_data = basic_analytics_df[
            (basic_analytics_df["Date"] >= start_of_week) &
            (basic_analytics_df["Date"] <= end_of_week)
            ]

        # Filter for current day
        daily_data = monthly_data[monthly_data["Date"].dt.day == self.today.day]

        # Filtering based on selection
        if self.selection == 'Today':
            self.data = daily_data
        elif self.selection == 'This Week':
            self.data = weekly_data
        elif self.selection == 'This Month':
            self.data = monthly_data
        elif self.selection == 'This Quarter':
            self.data = quarterly_data
        elif self.selection == 'This Year':
            self.data = yearly_data

    def trend_analysis_graph_func(self):
        """
        Creates a trend analysis graph object based on the selection.
        :return: fig
        """
        try:
            # trend analysis
            totals = self.data.groupby(self.data['Date'].dt.to_period('M'))['FinalAmount'].sum().reset_index()

            totals['Date'] = totals['Date'].dt.to_timestamp()

            fig = px.line(
                totals,
                x='Date',
                y='FinalAmount',
                title=f"{self.selection}'s Monthly Sales",
                labels={'FinalAmount': 'Total Sales'},
                markers=True
            )
        except Exception as e:
            print(e)
            log_error(str(e), source="trend_analysis_graph_func in basicgraph.py")
            fig = self.empty_fig

        return fig

    def top_bestselling_products_graph_func(self):
        """
        Creates top 10 bestselling products graph as a bar chart based on selection.
        :return: fig
        """
        try:
            # top 10 bestselling products
            if self.data is None or "ProductName" not in self.data.columns:
                print("Missing 'ProductName' column for bar graph.")

            top_products = self.data["ProductName"].value_counts().sort_values(ascending=False).head(20)
            df_top = top_products.reset_index()
            df_top.columns = ["ProductName", "SalesCount"]

            fig = px.bar(
                df_top,
                x="ProductName",
                y="SalesCount",
                color="ProductName",  # Categorical coloring
                title=f"{self.selection}'s Top 20 Bestselling Products",
                color_discrete_sequence=px.colors.qualitative.Plotly  # Use a vibrant palette
            )
        except Exception as e:
            print(e)
            log_error(str(e), source="top_bestselling_products_graph_func in basicgraph.py")
            fig = self.empty_fig


        return fig

    def profit_margin_by_category_graph_func(self):
        """
        Creates a profit_margin_by_category_graph object
        :return: fig
        """
        try:
            category_col = "Category"
            revenue_col = "FinalAmount"
            cost_price_col = "CostPrice"
            quantity_col = "Quantity"

            required_cols = [category_col, revenue_col, cost_price_col, quantity_col]
            missing = [col for col in required_cols if col not in self.data.columns]
            if missing:
                print(f"Missing columns for profit margin chart: {', '.join(missing)}")
                return self.empty_fig

            self.data["Cost"] = self.data[cost_price_col] * self.data[quantity_col]

            if "FinalAmount" in self.data.columns:
                self.data["Profit"] = self.data[revenue_col] - self.data["Cost"]
            elif "SellingPrice" in self.data.columns and "Quantity" in self.data.columns:
                self.data["Profit"] = (self.data["SellingPrice"] * self.data["Quantity"]) - self.data["Cost"]
            else:
                print("No valid price columns to calculate profit.")
                return self.empty_fig

            if self.data.empty:
                print("Data is empty.")
                return self.empty_fig

            grouped = self.data.groupby(category_col).agg(
                Total_Profit=("Profit", "sum"),
                Total_Revenue=(revenue_col, "sum")
            ).reset_index()

            if grouped.empty:
                print("Grouped data is empty.")
                return self.empty_fig

            grouped["Profit_Margin (%)"] = round((grouped["Total_Profit"] / grouped["Total_Revenue"]) * 100, 2)
            grouped = grouped.sort_values("Profit_Margin (%)", ascending=False)

            title = f"{getattr(self, 'selection', 'Business')}'s Profit Margin By Category"

            fig = px.bar(
                grouped,
                x=category_col,
                y="Profit_Margin (%)",
                color="Profit_Margin (%)",
                title=title,
                color_continuous_scale="Viridis"
            )
        except Exception as e:
            print(e)
            log_error(str(e), source="profit_margin_by_category_graph_func in basicgraph.py")
            fig = self.empty_fig

        return fig

    def inventory_aging_table_func(self, basic_analytics_df):
        """
        The top 10 inventory aging table is returned in sorted order
        :return: inventory_aging_df
        """
        try:
            inventory_df = basic_analytics_df.copy()
            if "ProductName" not in inventory_df.columns or "Date" not in inventory_df.columns:
                print("Missing 'ProductName' or 'Date' column for inventory aging.")

            # Step 1: Get latest transaction date per ProductName
            last_transaction = inventory_df.groupby('ProductName')['Date'].max().reset_index()
            last_transaction.columns = ['ProductName', 'LastTransactionDate']

            # Step 2: Calculate inventory age
            last_transaction['InventoryAge (days)'] = (self.today - last_transaction['LastTransactionDate']).dt.days

            # Step 3: Sort by age
            last_transaction = last_transaction.sort_values(by='InventoryAge (days)', ascending=False)
            inventory_aging_df = last_transaction[['ProductName', 'LastTransactionDate', 'InventoryAge (days)']]
        except Exception as e:
            print(e)
            log_error(str(e), source="profit_margin_by_category_graph_func in basicgraph.py")
            inventory_aging_df = pd.DataFrame(columns=['ProductName', 'LastTransactionDate', 'InventoryAge (days)'])

        return inventory_aging_df
