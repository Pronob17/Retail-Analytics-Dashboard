import os
from fpdf import FPDF
import pandas as pd
import datetime as dt

def pdf_generator_func(kpi_tuple, graph_tuple, inv_tuple, ml_tuple):
    """
    Generates a comprehensive Retail Analytics PDF report using KPIs, graphs, inventory data,
    and ML insights including Customer Lifetime Value (CLV), Customer Segmentation, and Sales Forecasting.
    """
    pdf = FPDF()
    pdf.set_margins(left=10, top=20, right=10)
    pdf.add_page()
    image_paths = []

    # --- Horizontal Line Helper ---
    def add_horizontal_line(pdf_obj, thickness=0.5):
        pdf_obj.set_line_width(thickness)
        pdf_obj.line(10, pdf_obj.get_y(), 200, pdf_obj.get_y())
        pdf_obj.ln(3)

    # --- Helper: Draw DataFrame as Table ---
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

    # --- Title ---
    add_horizontal_line(pdf, thickness=0.9)
    pdf.set_font("Arial", "B", 24)
    # orange
    pdf.set_text_color(255, 165, 0)
    pdf.cell(0, 10, "RETAIL ANALYTICS REPORT", ln=True, align='C')
    add_horizontal_line(pdf, thickness=0.9)
    pdf.set_text_color(64, 64, 64)  # Dark gray
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"{dt.datetime.now()}", ln=True, align='R')
    pdf.ln(10)

    # --- KPIs Section ---
    pdf.set_font("Arial", "B", 16)
    # subtopic
    # blue
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Key Performance Indicators (KPIs)", ln=True, align='C')
    # paragraph gap
    pdf.ln(5)
    pdf.set_text_color(64, 64, 64)  # Dark gray
    pdf.set_font("Arial", "", 12)
    for i, (label, value) in enumerate(kpi_tuple):
        pdf.cell(0, 8, f"{label}: {value}", ln=True)
        if i == 0:
            pdf.ln(5)

    # KPI description section
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 10, "What does the above mean?", ln=True)
    pdf.set_text_color(130, 130, 130)  # Light gray
    pdf.multi_cell(0, 8,
                   "This section presents 6 key performance indicators (KPIs):\n"
                   "- Total Sales (in amount): The overall revenue generated in the selected period.\n"
                   "- Gross Profit Margin (in percentages): Percentage of sales retained after deducting the cost of goods sold.\n"
                   "- Total Customers: Unique number of customers who made purchases.\n"
                   "- Customer Frequency: How often, on average, customers return to make another purchase.\n"
                   "- Average Order Value (in amount): The mean amount spent per order.\n"
                   "- Average Days Between Purchases: Helps understand buying intervals, indicating customer loyalty or drop-off.",
                   0, "L")

    # --- Graphs Section ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    # subtopic
    # blue
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Visual Insights", ln=True, align='C')
    # paragraph gap
    pdf.ln(5)
    pdf.set_text_color(64, 64, 64)  # Dark gray

    for idx, fig in enumerate(graph_tuple):
        img_path = f"temp_chart_{idx}.png"
        fig.write_image(img_path)
        image_paths.append(img_path)
        pdf.image(img_path, x=10, w=180)
        pdf.ln(5)

    # Graph description section
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 10, "What does the above mean?", ln=True)
    pdf.set_text_color(130, 130, 130)  # Light gray
    pdf.multi_cell(0, 8,
                   "This section visualizes important patterns from your data:\n"
                   "- Trend Analysis: Tracks revenue performance across time to highlight seasonality or growth trends.\n"
                   "- Top 20 Bestselling Products: Lists the highest-selling products by quantity or revenue.\n"
                   "- Profit Margin by Category: Shows which product categories are contributing most to profit.",
                   0, "L")

    # --- Inventory Aging Section ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    # subtopic
    # blue
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Inventory Aging Table", ln=True, align='C')
    # paragraph gap
    pdf.ln(5)
    pdf.set_text_color(64, 64, 64)  # Dark gray
    if isinstance(inv_tuple, pd.DataFrame):
        add_dataframe_section(pdf, inv_tuple, "Inventory Aging (Top 10 Rows)")

    # Inventory aging table description
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 10, "What does the above mean?", ln=True)
    pdf.set_text_color(130, 130, 130)  # Light gray
    pdf.multi_cell(0, 8,
                   "This table breaks down your inventory based on how long items have remained unsold:\n"
                   "- It helps you identify aging stock that may need clearance or discounting.\n"
                   "- Useful for optimizing stock rotation, warehouse space, and cash flow planning.",
                   0, "L")

    # -------------------------------
    # ---- SALES FORECASTING -------
    # -------------------------------
    pdf.add_page()
    sf = ml_tuple.get("Sales Forecasting", {})
    if sf:
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        # blue
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Sales Forecasting Summary", ln=True, align='C')
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        pdf.set_text_color(64, 64, 64)  # Dark gray

        reliability = sf.get("Reliability Percentage", 'N/A')
        reliability_str = f"{reliability:.2f}%" if isinstance(reliability, (int, float)) else str(reliability)
        if isinstance(reliability, (int, float)):
            label = "High" if reliability >= 75 else "Moderate" if reliability >= 50 else "Low"
            reliability_str = f"{reliability:.2f}% ({label} reliability)"

        pdf.cell(0, 8, f"Model Reliability: {reliability_str}", ln=True)
        pdf.cell(0, 8, f"Next Day: {sf.get('Next Day', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Next Day Prediction: {round(sf.get('Next Day Predictions', 0), 2)}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        add_horizontal_line(pdf, thickness=0.1)
        pdf.cell(0, 10, "TECHNICAL DETAILS OF MODEL RELIABILITY", ln=True)

        fig = sf.get("Line Chart Figure")
        if fig:
            img_path = "sales_forecast_fig.png"
            fig.write_image(img_path)
            pdf.image(img_path, x=10, w=180)
            image_paths.append(img_path)

        # sales forecasting description
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 10, "What does the above mean?", ln=True)
        pdf.set_text_color(130, 130, 130)  # Light gray
        pdf.multi_cell(0, 8,
                       "This section displays future sales predictions using machine learning:\n"
                       "- Sales Forecasting uses historical sales data to project future revenue.\n"
                       "- Helps in planning for demand, optimizing inventory levels, and avoiding stockouts or overstock.\n"
                       "- Useful for aligning marketing campaigns and staffing based on expected sales volume.",
                       0, "L")

    # -------------------------------
    # ---- CUSTOMER SEGMENTATION ---
    # -------------------------------
    pdf.add_page()
    seg = ml_tuple.get("Customer Segmentation", {})
    if seg:
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        # blue
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Customer Segmentation Summary", ln=True, align='C')
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        pdf.set_text_color(64, 64, 64)  # Dark gray

        reliability = seg.get("Reliability Percentage", 'N/A')
        reliability_str = f"{reliability:.2f}%" if isinstance(reliability, (int, float)) else str(reliability)
        if isinstance(reliability, (int, float)):
            label = "High" if reliability >= 75 else "Moderate" if reliability >= 50 else "Low"
            reliability_str = f"{reliability:.2f}% ({label} reliability)"

        pdf.cell(0, 8, f"Model Reliability: {reliability_str}", ln=True)
        pdf.cell(0, 8, f"Optimal Number of Segments (Best K): {seg.get('Best K', 'N/A')}", ln=True)

        add_dataframe_section(pdf, seg.get("Cluster Summary", pd.DataFrame()), "Cluster Summary")

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        add_horizontal_line(pdf, thickness=0.1)
        pdf.cell(0, 10, "TECHNICAL DETAILS OF MODEL RELIABILITY", ln=True)

        fig_elbow = seg.get("Elbow Plot Figure")
        if fig_elbow:
            elbow_path = "elbow_chart.png"
            fig_elbow.write_image(elbow_path)
            pdf.image(elbow_path, x=10, w=180)
            image_paths.append(elbow_path)

        fig_cluster = seg.get("Scatter Plot Figure")
        if fig_cluster:
            cluster_path = "cluster_chart.png"
            fig_cluster.write_image(cluster_path)
            pdf.image(cluster_path, x=10, w=180)
            image_paths.append(cluster_path)

        # customer segmentation description
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 10, "What does the above mean?", ln=True)
        pdf.set_text_color(130, 130, 130)  # Light gray
        pdf.multi_cell(0, 8,
                       "This section shows how customers are grouped using clustering algorithms:\n"
                       "- Customer Segmentation uses K-Means to group customers with similar purchase behaviors.\n"
                       "- Segments may include high-value repeat buyers, occasional shoppers, or one-time customers.\n"
                       "- Helps tailor marketing strategies, offers, and communications for each segment.",
                       0, "L")

    # -------------------------------
    # ---- CUSTOMER LIFETIME VALUE -
    # -------------------------------
    pdf.add_page()
    clv = ml_tuple.get("Customer Lifetime Value", {})
    if clv:
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        # blue
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Customer Lifetime Value (CLV) Summary", ln=True, align='C')
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        pdf.set_text_color(64, 64, 64)  # Dark gray

        reliability = clv.get("Reliability", 'N/A')
        pdf.cell(0, 8, f"Model Reliability: {reliability}", ln=True)

        add_dataframe_section(pdf, clv.get("Sample Results", pd.DataFrame()), "Top 10 Predicted CLV Customers:")

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        add_horizontal_line(pdf, thickness=0.1)
        pdf.cell(0, 10, "TECHNICAL DETAILS OF MODEL RELIABILITY", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Train R2: {clv.get('r2_train', 'N/A')} | Test R2: {clv.get('r2_test', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"MAE: {clv.get('mae', 'N/A')} | MSE: {clv.get('mse', 'N/A')}", ln=True)

        fig = clv.get("fig_hist")
        if fig:
            clv_path = "clv_chart.png"
            fig.write_image(clv_path)
            pdf.image(clv_path, x=10, w=180)
            image_paths.append(clv_path)

        # customer lifetime value description
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 10, "What does the above mean?", ln=True)
        pdf.set_text_color(130, 130, 130)  # Light gray
        pdf.multi_cell(0, 8,
                       "This section estimates how valuable each customer is over the long term:\n"
                       "- Customer Lifetime Value (CLV) predicts the total revenue a customer is expected to bring over their relationship.\n"
                       "- Helps identify high-value customers worth retaining and nurturing.\n"
                       "- Useful for prioritizing loyalty campaigns and customer service investments.",
                       0, "L")

    # final end line
    add_horizontal_line(pdf, thickness=0.9)
    # --- Export PDF as bytes ---
    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    # --- Cleanup temp images ---
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return pdf_bytes
