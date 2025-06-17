import streamlit as st
# wide layout display
st.set_page_config(layout="wide")

import pandas as pd
import plotly.graph_objects as go

from tests.errorlog import log_error





class DashboardClass:
    def __init__(self):
        pass

    def title_func(self):
        """
        Creates title display.
        :return: None
        """
        # create title display
        path = "./assets/"
        lt, ct, rt = st.columns([1,3,1])
        lt.image(path+"retail.png")
        ct.title("RETAIL ANALYTICS DASHBOARD")
        rt.image(path+"analytics.png")

        # create a divider
        st.divider()

    def upload_func(self):
        """
        Creates upload and demo selectbox.
        :return: selection, upload_file
        """
        options = ['Demo Data', 'Upload Data']
        selection = st.sidebar.selectbox(label="SELECT THE DATA", options=options)

        # if selection is upload data
        if selection == 'Upload Data':
            upload_file = st.sidebar.file_uploader("Select file")
            # check if uploaded file exist
            if upload_file:
                st.sidebar.success("File Uploaded Successfully!!")
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
        with st.sidebar.expander("EXPAND TO SEE MATCHED COLUMNS"):
            st.dataframe(validity)

    def kpi_date_range_func(self, start_date_default, end_date_default):
        """
        Get date ranges from user.
        :return: start_date, end_date
        """
        # kpi title
        st.markdown("## **KEY PERFORMANCE INDICATOR**")

        # create the date range input
        lt, cn, rt = st.columns([1,1,3])
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

    def show_kpi_func(self, total_sales, gross_profit_margin, total_customers, customer_frequency, average_order_value):
        """
        Shows the kpis on the dashboard
        :param total_sales:
        :param gross_profit_margin:
        :param total_customers:
        :param customer_frequency:
        :param average_order_value:
        :return: None
        """


        col1, col2, col3, col4, col5 = st.columns([2,1,1,1,1])
        col1.metric("TOTAL SALES", total_sales)
        col2.metric("GROSS PROFIT MARGIN (%)", gross_profit_margin)
        col3.metric("TOTAL CUSTOMERS", total_customers)
        col4.metric("CUSTOMER FREQUENCY", customer_frequency)
        col5.metric("AVERAGE ORDER VALUE", average_order_value)

        # create an info expander
        with st.expander("EXPAND FOR INFORMATION ON **KEY PERFORMANCE INDICATORS**"):
            inf1, inf2, inf3, inf4, inf5 = st.columns([2, 1, 1, 1, 1])
            inf1.markdown("**TOTAL SALES** – Sum of all sales revenue.")
            inf2.markdown("**GROSS PROFIT MARGIN (%)** – Percentage of revenue remaining after deducting COGS.")
            inf3.markdown("**TOTAL CUSTOMERS** – Count of unique customers.")
            inf4.markdown("**CUSTOMER FREQUENCY** – Average number of orders per customer.")
            inf5.markdown("**AVERAGE ORDER VALUE** – Average sales revenue per order.")

        # create divider
        st.divider()
    def graph_time_granularity_func(self):
        """
        Selects time granularity.
        :return: selection
        """
        st.markdown("## **VISUAL INSIGHTS**")
        selection = st.selectbox(label="SELECT FILTER OPTION", options=['Today', 'This Week', 'This Month', 'This Quarter', 'This Year'])
        return selection

    def show_graphs_func(self, trend_analysis_graph, top_bestselling_products_graph, profit_margin_by_category_graph):
        """
        Graph showing the analysis.
        :param trend_analysis_graph, top_bestselling_products_graph, profit_margin_by_category_graph
        :return: None
        """
        lt, cn, rt = st.tabs(['TREND ANALYSIS', 'TOP 20 BESTSELLING PRODUCTS', 'PROFIT MARGIN BY CATEGORY'])
        lt.plotly_chart(trend_analysis_graph, key="trend_analysis")
        cn.plotly_chart(top_bestselling_products_graph, key="top_products")
        rt.plotly_chart(profit_margin_by_category_graph, key="profit_margin")

        with st.expander("EXPAND FOR INFORMATION ON **GRAPHS**"):
            d1, d2, d3 = st.columns(3)
            d1.markdown("**TREND ANALYSIS** – Displays how total sales changes over time.")
            d2.markdown("**TOP 20 BESTSELLING PRODUCTS** – Shows the highest-selling products based on sales volume or revenue.")
            d3.markdown("**PROFIT MARGIN BY CATEGORY** – Compares profit margins across different product categories.")

        # create divider
        st.divider()

    def show_inventory_func(self, inventory_aging_df):
        """
        Shows the top 10 inventory aging table.
        :param inventory_aging_df:
        :return: None
        """
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
        # create main heading
        st.markdown("## **MACHINE LEARNING INSIGHTS**")

        # create multiple tabs
        ml1, ml2, ml3 = st.tabs(["Sales Forecasting", "Customer Segmentation", "Customer Lifetime Value"])

        # sales forecasting tab
        ml1.success(f"Next day's ({sales_forecast_dict['Next Day']}) Final Amount Forecast: **{sales_forecast_dict['Next Day Predictions']:.2f}**")
        ml1.plotly_chart(sales_forecast_dict['Line Chart Figure'])
        ml1.info(f"Model Reliability Percentage: **{sales_forecast_dict['Reliability Percentage']}%**")

        # additional verification details
        with ml1.expander("Technical Details of Model's Reliability"):
            st.dataframe(sales_forecast_dict['Sales Forecast Dataframe'])
            st.markdown(f"Train R2 Score: **{sales_forecast_dict['Train R2 Score']}** | Test R2 Score: **{sales_forecast_dict['Test R2 Score']}**")

        #  customer segmentation tab
        ml2.success(f"Total number of clusters: **{customer_segmentation_dict['Best K']}**")
        ml2.dataframe(customer_segmentation_dict['Cluster Summary'])
        ml2.info(f"Model Reliability Percentage: **{customer_segmentation_dict['Reliability Percentage']}**")

        with ml2.expander("Technical Details of Model's Reliability"):
            st.plotly_chart(customer_segmentation_dict['Elbow Plot Figure'])
            st.plotly_chart(customer_segmentation_dict['Scatter Plot Figure'])
            st.dataframe(customer_segmentation_dict['Segmented RFM Dataframe'])

        # customer lifetime value tab
        ml3.success("Top 10 Customers based on Predicted Lifetime values:")
        ml3.dataframe(customer_lifetime_value_dict['Sample Results'])
        ml3.info(f"Model Reliability: **{customer_lifetime_value_dict['Reliability']}**")


        with ml3.expander("Technical Details of Model's Reliability"):
            st.plotly_chart(customer_lifetime_value_dict['fig_hist'])
            st.markdown(
                f"Train R2 Score: **{customer_lifetime_value_dict['r2_train']}** | Test R2 Score: **{customer_lifetime_value_dict['r2_test']}**")
            st.markdown(f"Mean Absolute Error: **{customer_lifetime_value_dict['mae']}**")
            st.markdown(f"Mean Squared Error: **{customer_lifetime_value_dict['mse']}**")

        # ml models description
        with st.expander("EXPAND FOR INFORMATION ON **MACHINE LEARNING INSIGHTS**"):
            des1, des2, des3 = st.columns(3)
            st.markdown("**NOTE - Model Reliability Percentage** shows how much you can trust the model’s predictions on both past and new data.")
            des1.markdown("**Sales Forecasting**: Predicts future sales based on past trends and influencing factors.")
            des2.markdown("**Customer Segmentation**: Groups customers into clusters based on similar behaviors or attributes.")
            des3.markdown("**Customer Lifetime Value**: Estimates the total revenue a business can expect from a customer over their lifetime.")

        st.divider()

    def show_download_func(self):
        pass