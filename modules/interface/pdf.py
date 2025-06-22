import os
from fpdf import FPDF
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# use ggplot style
plt.style.use('ggplot')

def pdf_generator_func(kpi_tuple, graph_tuple, inv_tuple, ml_tuple):
    pdf = FPDF()
    pdf.set_margins(left=10, top=20, right=10)
    pdf.add_page()
    image_paths = []

    def add_horizontal_line(pdf_obj, thickness=0.5):
        pdf_obj.set_line_width(thickness)
        pdf_obj.line(10, pdf_obj.get_y(), 200, pdf_obj.get_y())
        pdf_obj.ln(3)

    def add_dataframe_section(pdf_obj, df, title, max_rows=10):
        if df is None or df.empty:
            return
        pdf_obj.ln(5)
        pdf_obj.set_font("Arial", "B", 12)
        pdf_obj.cell(0, 10, title, ln=True)

        df = df.head(max_rows).fillna("").astype(str)
        col_width = pdf_obj.w / (len(df.columns) + 1)

        pdf_obj.set_font("Arial", "B", 10)
        for col in df.columns:
            pdf_obj.cell(col_width, 8, str(col), border=1)
        pdf_obj.ln()

        pdf_obj.set_font("Arial", "", 10)
        for _, row in df.iterrows():
            for item in row:
                pdf_obj.cell(col_width, 8, str(item), border=1)
            pdf_obj.ln()
        pdf_obj.ln(3)

    def save_fig(fig_or_df, path, plot_type="auto", title="", xlabel="", ylabel=""):
        if hasattr(fig_or_df, "savefig"):  # matplotlib figure
            fig_or_df.savefig(path, format="png")
            plt.close(fig_or_df)

        elif isinstance(fig_or_df, pd.DataFrame):  # Pandas DataFrame
            fig, ax = plt.subplots(figsize=(10, 6))
            kind = plot_type if plot_type != "auto" else "line"
            fig_or_df.plot(kind=kind, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            fig.savefig(path, format="png")
            plt.close(fig)

        else:
            raise ValueError("Unsupported figure type.")
        image_paths.append(path)

    # Title
    add_horizontal_line(pdf, thickness=0.9)
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(255, 165, 0)
    pdf.cell(0, 10, "RETAIL ANALYTICS REPORT", ln=True, align='C')
    add_horizontal_line(pdf, thickness=0.9)
    pdf.set_text_color(64, 64, 64)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Downloaded on: {dt.datetime.now().date()}", ln=True, align='R')
    pdf.ln(10)

    # KPIs
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)
    add_horizontal_line(pdf)
    pdf.cell(0, 10, "Key Performance Indicators (KPIs)", ln=True, align='C')
    add_horizontal_line(pdf)
    pdf.ln(5)
    pdf.set_text_color(64, 64, 64)
    pdf.set_font("Arial", "", 12)
    for i, (label, value) in enumerate(kpi_tuple):
        pdf.cell(0, 8, f"{label}: {value}", ln=True)
        if i == 0:
            pdf.ln(5)

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 10, "How to interpret the above result?", ln=True)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 8, (
        "This section presents six key performance indicators (KPIs) based on the selected date range. Each metric is designed to summarize a specific aspect of business activity and can be interpreted in relation to the reported values:\n\n"
        "- Total Sales: The total monetary value of all completed transactions during the period. Higher values indicate stronger revenue generation.\n"
        "- Gross Profit Margin: The percentage of revenue retained after subtracting the cost of goods sold. Higher margins reflect better pricing power or cost efficiency.\n"
        "- Total Customers: The count of unique individuals who made purchases. A higher count suggests broader market reach or improved customer acquisition.\n"
        "- Customer Frequency: The average number of purchases made per customer. Higher frequency reflects stronger customer retention and repeat engagement.\n"
        "- Average Order Value: The mean value of individual transactions. Larger values may indicate upselling success, premium offerings, or bundled purchases.\n"
        "- Average Days Between Purchases: The typical number of days between repeat purchases by the same customer. Shorter intervals indicate faster buying cycles and more frequent engagement."
    ))

    # Graphs
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)
    add_horizontal_line(pdf)
    pdf.cell(0, 10, "Visual Insights", ln=True, align='C')
    add_horizontal_line(pdf)
    pdf.ln(5)
    pdf.set_text_color(64, 64, 64)
    for idx, fig in enumerate(graph_tuple):
        img_path = f"temp_chart_{idx}.png"
        save_fig(fig, img_path)
        pdf.image(img_path, x=10, w=180)
        pdf.ln(5)

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 10, "How to interpret the above result?", ln=True)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 8, (
        "This section visualizes key patterns and trends derived from the selected data range. These charts provide context for evaluating performance dynamics, product demand, and profitability distribution:\n\n"
        "- Trend Analysis: Illustrates the progression of total sales over time, helping to identify growth trends, seasonal cycles, or declining phases.\n"
        "- Top 20 Bestselling Products: Displays the products with the highest sales volumes during the selected period. High-ranking items indicate strong demand or successful promotion.\n"
        "- Profit Margin by Category: Compares profit margins across product categories by evaluating revenue relative to direct costs. Categories with higher margins contribute more effectively to overall profitability."
    ))

    # Inventory Aging
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)
    add_horizontal_line(pdf)
    pdf.cell(0, 10, "Inventory Aging Table", ln=True, align='C')
    add_horizontal_line(pdf)
    pdf.set_text_color(64, 64, 64)
    if isinstance(inv_tuple, pd.DataFrame):
        add_dataframe_section(pdf, inv_tuple, "Inventory Aging (Top 10 Rows)")

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 10, "How to interpret the above result?", ln=True)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 8, (
        "This table provides an overview of inventory aging, showing how long different products have remained in stock without being sold. The information supports stock management and operational efficiency:\n\n"
        "- Highlights items with extended holding periods, which may require discounting, clearance, or promotional action.\n"
        "- Aids in stock rotation planning to ensure older inventory is prioritized before reaching obsolescence.\n"
        "- Contributes to better space utilization and cash flow management by reducing capital locked in non-moving stock."
    ))

    # Sales Forecasting
    sf = ml_tuple.get("Sales Forecasting", {})
    if sf:
        pdf.add_page()
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Sales Forecasting Summary", ln=True, align='C')
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(64, 64, 64)
        pdf.cell(0, 8, f"Model Reliability: {sf.get('Reliability Percentage', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Next Day: {sf.get('Next Day', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Next Day Prediction: {round(sf.get('Next Day Predictions', 0), 2)}", ln=True)
        fig = sf.get("Line Chart Figure")
        if fig:
            img_path = "sales_forecast.png"
            save_fig(fig, img_path)
            pdf.image(img_path, x=10, w=180)

        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 10, "How to interpret the above result?", ln=True)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 8, (
            "This section provides a forward-looking estimate of future sales based on historical transaction patterns. It applies time-series modeling to uncover underlying trends and seasonality in past sales performance:\n\n"
            "- Projects upcoming sales by analyzing historical growth, declines, and cyclical behavior.\n"
            "- Supports inventory, workforce, and marketing planning by identifying periods of expected demand fluctuation.\n"
            "- Minimizes the likelihood of stockouts or overstock by aligning operational decisions with predicted sales volume.\n\n"
            "The forecasted figures offer a data-driven outlook derived from past business behavior, allowing for more informed and proactive decision-making."
        ))

    # Customer Segmentation
    seg = ml_tuple.get("Customer Segmentation", {})
    if seg:
        pdf.add_page()
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Customer Segmentation Summary", ln=True, align='C')
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(64, 64, 64)
        pdf.cell(0, 8, f"Model Reliability: {seg.get('Reliability Percentage', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Optimal Segments (Best K): {seg.get('Best K', 'N/A')}", ln=True)
        add_dataframe_section(pdf, seg.get("Cluster Summary", pd.DataFrame()), "Cluster Summary")
        if seg.get("Elbow Plot Figure"):
            elbow_path = "elbow_chart.png"
            save_fig(seg["Elbow Plot Figure"], elbow_path)
            pdf.image(elbow_path, x=10, w=180)
        if seg.get("Scatter Plot Figure"):
            cluster_path = "cluster_chart.png"
            save_fig(seg["Scatter Plot Figure"], cluster_path)
            pdf.image(cluster_path, x=10, w=180)

        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 10, "How to interpret the above result?", ln=True)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 8, (
            "This section categorizes customers into distinct segments based on purchasing behavior. The segmentation is designed to uncover behavioral patterns and support targeted strategies:\n\n"
            "- Grouping is performed using key variables: Recency (how recently a customer made a purchase), Frequency (how often purchases occur), and Monetary value (total spending).\n"
            "- Clustering techniques identify customers with similar RFM profiles and assign them to the same segment.\n"
            "- Segments may represent, for example, high-frequency and high-value buyers, recent one-time purchasers, or long-inactive customers.\n"
            "- Each segment reflects a behavioral identity that can guide differentiated marketing, engagement, and retention approaches.\n\n"
            "The summary table shows average recency, frequency, and monetary values for each cluster. These values can be interpreted directly to understand the characteristics and potential of each customer segment."
        ))

    # CLV
    clv = ml_tuple.get("Customer Lifetime Value", {})
    if clv:
        pdf.add_page()
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Customer Lifetime Value (CLV)", ln=True, align='C')
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(64, 64, 64)
        pdf.cell(0, 8, f"Model Reliability: {clv.get('Reliability', 'N/A')}", ln=True)
        add_dataframe_section(pdf, clv.get("Sample Results", pd.DataFrame()), "Top 10 Predicted CLV Customers")
        pdf.cell(0, 8, f"Train R2: {clv.get('r2_train', 'N/A')} | Test R2: {clv.get('r2_test', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"MAE: {clv.get('mae', 'N/A')} | MSE: {clv.get('mse', 'N/A')}", ln=True)
        if clv.get("fig_hist"):
            clv_path = "clv_chart.png"
            save_fig(clv["fig_hist"], clv_path)
            pdf.image(clv_path, x=10, w=180)

        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 10, "How to interpret the above result?", ln=True)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 8, (
            "This section estimates the Customer Lifetime Value (CLV), representing the total expected revenue from a customer throughout the duration of their relationship with the business:\n\n"
            "- CLV is derived from historical purchasing patterns, including purchase frequency, average spend, and projected retention period.\n"
            "- Higher CLV values indicate customers with greater long-term financial contribution.\n"
            "- This insight enables strategic allocation of resources toward acquiring, nurturing, and retaining the most valuable customers.\n\n"
            "The accompanying summary table displays predicted CLV per customer, providing a quantitative basis for prioritizing marketing, loyalty, and service initiatives."
        ))

    # Final cleanup
    pdf.ln(10)
    add_horizontal_line(pdf, thickness=0.9)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return pdf_bytes
