from datetime import date

from modules.analytics.basicanalytics import BasicAnalyticsClass
from modules.interface.dashboard import DashboardClass
from modules.machinelearning.machinelearningoperations import MachineLearningClass
from modules.preprocessor.cleaner import CleanerClass
from modules.preprocessor.loader import LoaderClass
from modules.visualization.basicgraphs import BasicGraphsClass



def main():
    data_path = "./data/"
    main_df = None

    # create dashboard instance
    dashboard = DashboardClass()

    # call title display
    dashboard.title_func()

    # call the loader module to get demo dataframe
    load = LoaderClass()

    # call upload display and load file
    selection, uploaded_file = dashboard.upload_func()

    if selection=='Upload Data' and uploaded_file:
        uploaded_df = load.loader_func(uploaded_file)
        main_df, validity = load.standardize_func(uploaded_df, data_path)
        # send the result to show column details
        dashboard.sidebar_func(validity)
    elif selection=='Demo Data' and uploaded_file is None:
        main_df = load.demo_func()

    # create clean instance
    clean = CleanerClass()

    # pass the dataframe clean datetime of the dataframe (cleaned_date_main_df will be used in ml too)
    cleaned_date_main_df = clean.datetime_func(main_df)

    # impute the dataframe
    basic_analytics_df = clean.imputer_func(cleaned_date_main_df)

    # KEY PERFORMANCE INDICATORS

    # initialize the date range
    start_date_default = date(1980, 1, 1)
    end_date_default = date(2030, 1, 1)
    start_date, end_date = dashboard.kpi_date_range_func(start_date_default, end_date_default)

    # call the basic analytics operation
    kpi = BasicAnalyticsClass()
    total_sales, gross_profit_margin, total_customers, customer_frequency, average_order_value = kpi.kpi_calculation_func(
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
        average_order_value
    )

    # GRAPHS

    # get the time granularity
    selection = dashboard.graph_time_granularity_func()

    # call the basic graph operation
    graph = BasicGraphsClass(basic_analytics_df, selection)
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
    # create machine learning instance
    machine = MachineLearningClass(cleaned_date_main_df)

    # call sales forecasting function
    sales_forecast_dict = machine.ml_sales_forecasting_func()
    # send sales forecast to dashboard
    dashboard.show_ml_model_func(sales_forecast_dict)


    # call customer segmentation function


if __name__ == "__main__":
    main()