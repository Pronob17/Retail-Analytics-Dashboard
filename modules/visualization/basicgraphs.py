import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tests.errorlog import log_error

class BasicGraphsClass:
    def __init__(self, basic_analytics_df, selection):
        self.selection = selection
        self.data = None
        self.today = pd.Timestamp.now().normalize()

        # Placeholder empty fig
        fig, ax = plt.subplots()
        ax.set_title("No Data Available")
        ax.text(0.5, 0.5, "No chart to display", fontsize=16, ha='center')
        ax.axis('off')
        self.empty_fig = fig

        # --- Data validation ---
        if "Date" not in basic_analytics_df.columns:
            print("Missing 'Date' column.")
            basic_analytics_df["Date"] = pd.NaT

        basic_analytics_df["Date"] = pd.to_datetime(basic_analytics_df["Date"], errors='coerce')
        basic_analytics_df = basic_analytics_df[basic_analytics_df["Date"].notna()]

        df = basic_analytics_df.copy()
        df["Quarter"] = ((df["Date"].dt.month - 1) // 3) + 1

        start_of_week = self.today - pd.Timedelta(days=self.today.weekday())
        end_of_week = start_of_week + pd.Timedelta(days=6)

        daily_data = df[df["Date"].dt.date == self.today.date()]
        weekly_data = df[(df["Date"] >= start_of_week) & (df["Date"] <= end_of_week)]
        monthly_data = df[(df["Date"].dt.month == self.today.month) & (df["Date"].dt.year == self.today.year)]
        quarterly_data = df[(df["Quarter"] == ((self.today.month - 1) // 3 + 1)) & (df["Date"].dt.year == self.today.year)]
        yearly_data = df[df["Date"].dt.year == self.today.year]

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

        print(f"[{self.selection}] → Filtered rows: {len(self.data)}")

    def trend_analysis_graph_func(self):
        try:
            if self.data.empty:
                print("No data for trend analysis.")
                return self.empty_fig

            totals = self.data.groupby(self.data['Date'].dt.to_period('M'))['FinalAmount'].sum().reset_index()
            totals['Date'] = totals['Date'].dt.to_timestamp()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=totals, x='Date', y='FinalAmount', marker='o', ax=ax)
            ax.set_title(f"{self.selection}'s Monthly Sales")
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Sales')
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
        except Exception as e:
            print(e)
            log_error(str(e), source="trend_analysis_graph_func in basicgraph.py")
            fig = self.empty_fig

        return fig

    def top_bestselling_products_graph_func(self):
        try:
            if self.data.empty or "ProductName" not in self.data.columns:
                print("Missing 'ProductName' or data is empty.")
                return self.empty_fig

            top_products = self.data["ProductName"].value_counts().sort_values(ascending=False).head(20)
            df_top = top_products.reset_index()
            df_top.columns = ["ProductName", "SalesCount"]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_top, x="SalesCount", y="ProductName", palette="viridis", ax=ax)
            ax.set_title(f"{self.selection}'s Top 20 Bestselling Products")
            ax.set_xlabel('Sales Count')
            ax.set_ylabel('Product Name')
            ax.grid(True)
            plt.tight_layout()
        except Exception as e:
            print(e)
            log_error(str(e), source="top_bestselling_products_graph_func in basicgraph.py")
            fig = self.empty_fig

        return fig

    def profit_margin_by_category_graph_func(self):
        try:
            if self.data.empty:
                print("Data is empty.")
                return self.empty_fig

            required = ["Category", "FinalAmount", "CostPrice", "Quantity"]
            missing = [col for col in required if col not in self.data.columns]
            if missing:
                print(f"Missing columns: {missing}")
                return self.empty_fig

            self.data["Cost"] = self.data["CostPrice"] * self.data["Quantity"]
            self.data["Profit"] = self.data["FinalAmount"] - self.data["Cost"]

            grouped = self.data.groupby("Category").agg(
                Total_Profit=("Profit", "sum"),
                Total_Revenue=("FinalAmount", "sum")
            ).reset_index()

            if grouped.empty:
                print("Grouped data is empty.")
                return self.empty_fig

            grouped["Profit_Margin (%)"] = grouped.apply(
                lambda row: round((row["Total_Profit"] / row["Total_Revenue"]) * 100, 2)
                if row["Total_Revenue"] != 0 else 0,
                axis=1
            )

            grouped = grouped.sort_values("Profit_Margin (%)", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=grouped, x="Profit_Margin (%)", y="Category", palette="magma", ax=ax)
            ax.set_title(f"{self.selection}'s Profit Margin By Category")
            ax.set_xlabel('Profit Margin (%)')
            ax.set_ylabel('Category')
            ax.grid(True)
            plt.tight_layout()
        except Exception as e:
            print(e)
            log_error(str(e), source="profit_margin_by_category_graph_func in basicgraph.py")
            fig = self.empty_fig

        return fig

    def inventory_aging_table_func(self, basic_analytics_df):
        try:
            inventory_df = basic_analytics_df.copy()
            if "ProductName" not in inventory_df.columns or "Date" not in inventory_df.columns:
                print("Missing 'ProductName' or 'Date' column for inventory aging.")
                return pd.DataFrame(columns=['ProductName', 'LastTransactionDate', 'InventoryAge (days)'])

            inventory_df["Date"] = pd.to_datetime(inventory_df["Date"], errors='coerce')
            inventory_df = inventory_df[inventory_df["Date"].notna()]

            last_transaction = inventory_df.groupby('ProductName')['Date'].max().reset_index()
            last_transaction.columns = ['ProductName', 'LastTransactionDate']
            last_transaction['InventoryAge (days)'] = (self.today - last_transaction['LastTransactionDate']).dt.days
            last_transaction = last_transaction.sort_values(by='InventoryAge (days)', ascending=False)

            inventory_aging_df = last_transaction[['ProductName', 'LastTransactionDate', 'InventoryAge (days)']]
        except Exception as e:
            print(e)
            log_error(str(e), source="inventory_aging_table_func in basicgraph.py")
            inventory_aging_df = pd.DataFrame(columns=['ProductName', 'LastTransactionDate', 'InventoryAge (days)'])

        return inventory_aging_df
