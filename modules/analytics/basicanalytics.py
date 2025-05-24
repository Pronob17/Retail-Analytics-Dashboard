import pandas as pd

from tests.errorlog import log_error



class BasicAnalyticsClass:
    def __init__(self):
        pass

    def kpi_calculation_func(self, main_df, start_date, end_date):
        """
        Calculates the KPI of the dataframe.
        :return: total_sales, gross_profit_margin, total_customers, customer_frequency, average_order_value
        """
        # convert to pandas datetime
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)

        # initialize total_sales
        total_sales = 0

        # create filter of date range
        mask = ((main_df["Date"] >= start_date_pd) & (main_df["Date"] <= end_date_pd))

        # create the filtered dataframe
        filtered_df = main_df[mask]

        # total_sales
        # 'FinalAmount', 'SellingPrice', 'Discount', 'Quantity'
        try:
            if 'FinalAmount' in filtered_df.columns:
                total_sales = round(filtered_df['FinalAmount'].sum(), 2)
                print(f"Total Sales: {total_sales}")
            elif 'SellingPrice' in filtered_df.columns and 'Discount' in filtered_df.columns:
                # Calculate total sales based on SellingPrice and Discount
                total_sales = round((filtered_df['Quantity'] * filtered_df['SellingPrice']) - (
                            filtered_df['Quantity'] * filtered_df['SellingPrice'] * filtered_df[
                        'Discount'] / 100).sum(), 2)
        except Exception as e:
            print(e)
            log_error(str(e), source="kpi_calculation_func in basicanalytics.py")
            total_sales = "N/A"

        # gross profit margin
        # 'FinalAmount', 'CostPrice', 'Quantity'
        try:
            gross_profit = filtered_df['FinalAmount'].sum() - (
                        filtered_df['Quantity'] * filtered_df['CostPrice']).sum()
            gross_profit_margin = round(gross_profit / filtered_df['FinalAmount'].sum() * 100, 2)
        except Exception as e:
            print(e)
            log_error(str(e), source="kpi_calculation_func in basicanalytics.py")
            gross_profit_margin = "N/A"

        # total customers
        # 'CustomerID'
        try:
            total_customers = round(filtered_df['CustomerID'].nunique(), 2)
        except Exception as e:
            print(e)
            log_error(str(e), source="some_function in basicanalytics.py")
            total_customers = "N/A"

        # customer frequency
        # 'TransactionID', 'CustomerID'
        # customer_frequency = Total Orders / Unique Customers
        try:
            customer_frequency = round(filtered_df['TransactionID'].nunique() / filtered_df['CustomerID'].nunique(), 2)
        except Exception as e:
            print(e)
            log_error(str(e), source="kpi_calculation_func in basicanalytics.py")
            customer_frequency = "N/A"

        # average order value
        # 'FinalAmount', 'TransactionID'
        try:
            average_order_value = round(filtered_df['FinalAmount'].sum() / filtered_df['TransactionID'].nunique(), 2)
        except Exception as e:
            print(e)
            log_error(str(e), source="kpi_calculation_func in basicanalytics.py")
            average_order_value = "N/A"

        return total_sales, gross_profit_margin, total_customers, customer_frequency, average_order_value