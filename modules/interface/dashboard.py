import streamlit as st

# wide layout display
st.set_page_config(layout="wide")

import pandas as pd

from tests.errorlog import log_error




class DashboardClass:
    def __init__(self):
        pass

    def title_func(self):
        """
        Creates title display.
        :return: None
        """

        with st.container():
            # create title display
            path = "./assets/"
            lt, ct, rt = st.columns([1,3,1])
            lt.image(path+"retail.png")
            ct.title("RETAIL ANALYTICS DASHBOARD")
            rt.image(path+"analytics.png")
            # divider
            st.divider()

        # create the necessary column
        # Minimum required columns with unambiguous descriptions
        with st.container():
            dashboard_key_descriptions = {
                "CustomerID": "Unique ID given to each customer to track their purchases.",
                "CustomerName": "Full name of the customer (optional, used for display).",
                "Date": "The date the purchase was made (format: YYYY-MM-DD).",
                "FinalAmount": "Total amount paid after all discounts and taxes.",
                "SellingPrice": "Price of one unit before any discount.",
                "CostPrice": "Cost of one unit to the business (used to calculate profit).",
                "ProductName": "Name or description of the product sold.",
                "Quantity": "Number of units sold in the transaction.",
                "Category": "Product type or category (e.g., Electronics, Clothing).",
                "TransactionID": "Unique ID for the sale, used to identify each transaction.",
                "DiscountPercent": "Percentage of discount applied before tax."
            }

            # Create DataFrame
            df_sidebar_keys = pd.DataFrame.from_dict(
                dashboard_key_descriptions, orient="index", columns=["Description"]
            )

            # Streamlit sidebar display
            with st.sidebar.expander("Required Columns for Dashboard to Work"):
                st.dataframe(df_sidebar_keys)

    def upload_func(self):
        """
        Creates upload and demo selectbox.
        :return: selection, upload_file
        """
        with st.container():
            options = ['Demo Data', 'Upload CSV file']
            selection = st.sidebar.selectbox(label="SELECT THE DATA", options=options)

            # if selection is upload data
            if selection == 'Upload CSV file':
                upload_file = st.sidebar.file_uploader("Select file")
                # check if uploaded file exist
                if upload_file:
                    return selection, upload_file
                # if it doesnt stop the streamlit display
                else:
                    st.sidebar.warning("No file uploaded.")
                    st.stop()
            # if selection is not upload data
            else:
                return selection, None

    def sidebar_func(self, validity):
        """
        Shows the matched columns.
        :param validity:
        :return: None
        """
        with st.container():
            with st.sidebar.expander("EXPAND TO SEE MATCHED COLUMNS"):
                st.dataframe(validity)

    def kpi_date_range_func(self, start_date_default, end_date_default):
        """
        Get date ranges from user.
        :return: start_date, end_date
        """
        # kpi title
        with st.container():
            st.markdown("## **KEY PERFORMANCE INDICATORS**")

            # create the date range input
            lt, cn, rt = st.columns([2,1,3])
            date_range = lt.date_input(
                "SELECT DATE RANGE",
                value=(start_date_default, end_date_default),
                min_value=start_date_default,
                max_value=end_date_default
            )
            # unpack the tuple
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                rt.warning("Select a start date and an end date.")
                st.stop()

        return start_date, end_date

    def show_kpi_func(self, total_sales, gross_profit_margin, total_customers, customer_frequency, average_order_value, avg_days_between_purchases):
        """
        Shows the kpis on the dashboard
        :param total_sales:
        :param gross_profit_margin:
        :param total_customers:
        :param customer_frequency:
        :param average_order_value:
        :param avg_days_between_purchases
        :return: None
        """

        with st.container():
            col11, col12, col13 = st.columns(3)
            col11.metric("TOTAL SALES", total_sales)
            col12.metric("GROSS PROFIT MARGIN (%)", gross_profit_margin)
            col13.metric("AVERAGE TIME BETWEEN PURCHASES", avg_days_between_purchases)

            col21, col22, col23 = st.columns(3)
            col21.metric("TOTAL CUSTOMERS", total_customers)
            col22.metric("CUSTOMER FREQUENCY", customer_frequency)
            col23.metric("AVERAGE ORDER VALUE", average_order_value)

            for val in [total_sales, gross_profit_margin, avg_days_between_purchases,
                        total_customers, customer_frequency, average_order_value]:
                print(f"KPI Value: {val} | Type: {type(val)}")

            # create an info expander
            with st.expander("EXPAND FOR INFORMATION ON **KEY PERFORMANCE INDICATORS**"):
                inf1, inf2, inf3 = st.columns(3)
                inf1.markdown("**TOTAL SALES** – Sum of all sales revenue.")
                inf2.markdown("**GROSS PROFIT MARGIN (%)** – Percentage of revenue remaining after deducting COGS.")
                inf3.markdown("**AVERAGE TIME BETWEEN PURCHASES** – Average number of days between two purchases by the same customer.")

                inf4, inf5, inf6 = st.columns(3)
                inf4.markdown("**TOTAL CUSTOMERS** – Count of unique customers.")
                inf5.markdown("**CUSTOMER FREQUENCY** – Average number of orders per customer.")
                inf6.markdown("**AVERAGE ORDER VALUE** – Average sales revenue per order.")

            # create divider
            st.divider()
    def graph_time_granularity_func(self):
        """
        Selects time granularity.
        :return: selection
        """
        with st.container():
            st.markdown("## **VISUAL INSIGHTS**")
            selection = st.selectbox(label="SELECT FILTER OPTION", options=['Today', 'This Week', 'This Month', 'This Quarter', 'This Year'])
        return selection

    def show_graphs_func(self, trend_analysis_graph, top_bestselling_products_graph, profit_margin_by_category_graph):
        """
        Graph showing the analysis using Matplotlib/Seaborn.
        :param trend_analysis_graph, top_bestselling_products_graph, profit_margin_by_category_graph
        :return: None
        """
        with st.container():
            lt, cn, rt = st.tabs(['TREND ANALYSIS', 'TOP 20 BESTSELLING PRODUCTS', 'PROFIT MARGIN BY CATEGORY'])

            with lt:
                st.pyplot(trend_analysis_graph)

            with cn:
                st.pyplot(top_bestselling_products_graph)

            with rt:
                st.pyplot(profit_margin_by_category_graph)

            with st.expander("EXPAND FOR INFORMATION ON **GRAPHS**"):
                d1, d2, d3 = st.columns(3)
                d1.markdown("**TREND ANALYSIS** – Displays how total sales changes over time.")
                d2.markdown(
                    "**TOP 20 BESTSELLING PRODUCTS** – Shows the highest-selling products based on sales volume or revenue.")
                d3.markdown("**PROFIT MARGIN BY CATEGORY** – Compares profit margins across different product categories.")
            # divider
            st.divider()


    def show_inventory_func(self, inventory_aging_df):
        """
        Shows the top 10 inventory aging table.
        :param inventory_aging_df:
        :return: None
        """
        with st.container():
            st.markdown("## **INVENTORY AGING TABLE**")
            st.dataframe(inventory_aging_df)

            # description expander
            with st.expander("EXPAND FOR INFORMATION ON **INVENTORY AGING TABLE**"):
                st.markdown("**INVENTORY AGING TABLE** – Shows how long products have been in stock, helping identify slow-moving and old inventory.")

            # create divider
            st.divider()

    def show_ml_model_func(self, sales_forecast_dict, customer_segmentation_dict, customer_lifetime_value_dict):
        """
        Shows Sales Forecasting, Customer Segmentation and Customer Lifetime Value.
        :return: None
        """
        with st.container():
            # create main heading
            st.markdown("## **MACHINE LEARNING INSIGHTS**")

            # create multiple tabs
            ml1, ml2, ml3 = st.tabs(["Sales Forecasting", "Customer Segmentation", "Customer Lifetime Value"])

            # ----- SALES FORECASTING TAB -----
            ml1.success(
                f"Next day's ({sales_forecast_dict['Next Day']}) Final Amount Forecast: **{sales_forecast_dict['Next Day Predictions']:.2f}**")
            ml1.pyplot(sales_forecast_dict['Line Chart Figure'], use_container_width=True)
            ml1.info(f"Model Reliability Percentage: **{sales_forecast_dict['Reliability Percentage']}**")

            # additional verification details
            with ml1.expander("Technical Details of Model's Reliability"):
                st.dataframe(sales_forecast_dict['Sales Forecast Dataframe'])
                st.markdown(
                    f"Train R2 Score: **{sales_forecast_dict['Train R2 Score']}** | Test R2 Score: **{sales_forecast_dict['Test R2 Score']}**")

            # ----- CUSTOMER SEGMENTATION TAB -----
            ml2.success(f"Total number of clusters: **{customer_segmentation_dict['Best K']}**")
            ml2.dataframe(customer_segmentation_dict['Cluster Summary'])
            ml2.info(f"Model Reliability Percentage: **{customer_segmentation_dict['Reliability Percentage']}**")

            with ml2.expander("Technical Details of Model's Reliability"):
                st.pyplot(customer_segmentation_dict['Elbow Plot Figure'], use_container_width=True)
                st.pyplot(customer_segmentation_dict['Scatter Plot Figure'], use_container_width=True)
                st.dataframe(customer_segmentation_dict['Segmented RFM Dataframe'])

            # ----- CUSTOMER LIFETIME VALUE TAB -----
            ml3.success("Top 10 Customers based on Predicted Lifetime values:")
            ml3.dataframe(customer_lifetime_value_dict['Sample Results'])
            ml3.info(f"Model Reliability: **{customer_lifetime_value_dict['Reliability']}**")

            with ml3.expander("Technical Details of Model's Reliability"):
                st.pyplot(customer_lifetime_value_dict['fig_hist'], use_container_width=True)
                st.markdown(
                    f"Train R2 Score: **{customer_lifetime_value_dict['r2_train']}** | Test R2 Score: **{customer_lifetime_value_dict['r2_test']}**")
                st.markdown(f"Mean Absolute Error: **{customer_lifetime_value_dict['mae']}**")
                st.markdown(f"Mean Squared Error: **{customer_lifetime_value_dict['mse']}**")

            # ----- ML MODELS DESCRIPTION -----
            with st.expander("EXPAND FOR INFORMATION ON **MACHINE LEARNING INSIGHTS**"):
                des1, des2, des3 = st.columns(3)
                st.markdown(
                    "**NOTE - Model Reliability Percentage** shows how much you can trust the model’s predictions on both past and new data.")
                des1.markdown("**Sales Forecasting**: Predicts future sales based on past trends and influencing factors.")
                des2.markdown(
                    "**Customer Segmentation**: Groups customers into clusters based on similar behaviors or attributes.")
                des3.markdown(
                    "**Customer Lifetime Value**: Estimates the total revenue a business can expect from a customer over their lifetime.")
            # create divider
            st.divider()

    def pdf_download_func(self, summary_result):
        """
        This dashboard function gets the summary result from the main module and when the download button is clicked, it sends the summary_result.
        :param summary_result:
        :return: None
        """
        with st.container():
            # create a divider
            st.sidebar.divider()

            # create a button
            st.sidebar.download_button(
                label="📄 DOWNLOAD SUMMARY",
                data=summary_result,
                file_name="Retail_Summary_Result.pdf",
                mime="application/pdf"
            )
            st.sidebar.warning("NOTE: The summary reflects only the filters applied in the dashboard.")

            # create a divider
            st.sidebar.divider()