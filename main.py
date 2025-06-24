from datetime import date
import streamlit as st

from modules.analytics.basicanalytics import BasicAnalyticsClass
from modules.interface.dashboard import DashboardClass
from modules.machinelearning.machinelearningoperations import cache_func
from modules.preprocessor.cleaner import datetime_cleaner_func, complete_cleaner_func
from modules.preprocessor.loader import LoaderClass
from modules.visualization.basicgraphs import BasicGraphsClass
from modules.interface.pdf import pdf_generator_func


def main():
    """
    Main function of the app.
    :return: multiple objects
    """
    main_df = None

    # create dashboard instance
    dashboard = DashboardClass()

    # call title display
    dashboard.title_func()

    # call the loader module to get demo dataframe
    load = LoaderClass()

    # call upload display and load file
    selection, uploaded_file = dashboard.upload_func()

    if selection == 'Upload CSV file' and uploaded_file:
        uploaded_df, base_name = load.loader_func(uploaded_file)
        main_df, validity_df = load.standardize_func(uploaded_df)

        # send the result to show column details
        dashboard.sidebar_func(validity_df)

    elif selection == 'Demo Data' and uploaded_file is None:
        main_df, base_name = load.demo_func()

    # CLEANING PROCESS
    # call the datetime cleaner function
    with st.spinner('Cleanly formatting the dates...'):
        cleaned_date_main_df, datetime_list = datetime_cleaner_func(main_df)

    # call the complete cleaner function
    with st.spinner('Processing data for KPI analytics...'):
        basic_analytics_df = complete_cleaner_func(cleaned_date_main_df, datetime_list)

    # KEY PERFORMANCE INDICATORS

    # initialize the date range
    start_date_default = date(1980, 1, 1)
    end_date_default = date(2030, 1, 1)
    start_date, end_date = dashboard.kpi_date_range_func(start_date_default, end_date_default)

    #initialising the basic analytics class
    kpi = BasicAnalyticsClass()

    # call the basic analytics function
    total_sales, gross_profit_margin, total_customers, customer_frequency, average_order_value, avg_days_between_purchases = kpi.kpi_calculation_func(
        basic_analytics_df,
        start_date,
        end_date
    )

    # call the kpi display
    dashboard.show_kpi_func(
        total_sales,
        gross_profit_margin,
        total_customers,
        customer_frequency,
        average_order_value,
        avg_days_between_purchases
    )


    # GRAPHS

    # get the time granularity
    selection_granular = dashboard.graph_time_granularity_func()

    # initialise the basic graph class
    graph = BasicGraphsClass(basic_analytics_df, selection_granular)

    # call the basic graph operation
    trend_analysis_graph = graph.trend_analysis_graph_func()
    top_bestselling_products_graph = graph.top_bestselling_products_graph_func()
    profit_margin_by_category_graph = graph.profit_margin_by_category_graph_func()

    # send all the graphs to the dashboard
    dashboard.show_graphs_func(trend_analysis_graph, top_bestselling_products_graph, profit_margin_by_category_graph)


    # INVENTORY AGING TABLE
    # get the inventory_aging_df table
    inventory_aging_df = graph.inventory_aging_table_func(basic_analytics_df)
    # send to the dashboard
    dashboard.show_inventory_func(inventory_aging_df)


    # MACHINE LEARNING

    # call cache function to get the ml dictionaries
    with st.spinner("Running Machine Learning Operations..."):
        sales_forecast_dict, customer_segmentation_dict, customer_lifetime_value_dict = cache_func(cleaned_date_main_df)

    # Finally send the data to dashboard
    dashboard.show_ml_model_func(sales_forecast_dict, customer_segmentation_dict, customer_lifetime_value_dict)


    # --- PDF GENERATOR SETUP ---

    # 1. KPIs: convert to list of tuples with labels
    kpi_tuple = [
        ("DATE RANGE", f"Starting Date: {start_date} and Ending Date: {end_date}"),
        ("Total Sales", total_sales),
        ("Gross Profit Margin", gross_profit_margin),
        ("Total Customers", total_customers),
        ("Customer Frequency", customer_frequency),
        ("Average Order Value", average_order_value),
        ("Avg Days Between Purchases", avg_days_between_purchases)
    ]

    # 2. Graphs: make sure they are in a list or tuple
    graph_tuple = (
        trend_analysis_graph,
        top_bestselling_products_graph,
        profit_margin_by_category_graph
    )

    # 3. Inventory DataFrame
    inv_tuple = inventory_aging_df  # it's already a DataFrame

    # 4. ML Results: Combine all ML model dictionaries into a single one (optional: keep them separate)
    ml_tuple = {
        "Sales Forecasting": sales_forecast_dict,
        "Customer Segmentation": customer_segmentation_dict,
        "Customer Lifetime Value": customer_lifetime_value_dict
    }

    # 5. Generate PDF Summary
    with st.spinner("Now Generating Summary Report..."):
        summary_result = pdf_generator_func(kpi_tuple, graph_tuple, inv_tuple, ml_tuple, base_name)

    # 6. Download Trigger via Dashboard
    dashboard.pdf_download_func(summary_result, base_name)


if __name__ == "__main__":
    main()