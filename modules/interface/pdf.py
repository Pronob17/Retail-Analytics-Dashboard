import os
from fpdf import FPDF
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

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
    pdf.cell(0, 10, f"{dt.datetime.now()}", ln=True, align='R')
    pdf.ln(10)

    # KPIs
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Key Performance Indicators (KPIs)", ln=True, align='C')
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
    pdf.cell(0, 10, "What does the above mean?", ln=True)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 8, (
        "This section presents 6 key performance indicators (KPIs):\n"
        "- Total Sales: Overall revenue.\n"
        "- Gross Profit Margin: Profitability percentage.\n"
        "- Total Customers: Count of unique customers.\n"
        "- Customer Frequency: Repeat purchase rate.\n"
        "- Average Order Value: Mean purchase size.\n"
        "- Avg Days Between Purchases: Time gap in repeat buying."))

    # Graphs
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Visual Insights", ln=True, align='C')
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
    pdf.cell(0, 10, "What does the above mean?", ln=True)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 8, (
        "This section visualizes patterns in your data:\n"
        "- Trend Analysis: Sales performance over time.\n"
        "- Top 20 Bestselling Products: Highest-selling items.\n"
        "- Profit Margin by Category: Most profitable categories."))

    # Inventory Aging
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Inventory Aging Table", ln=True, align='C')
    pdf.set_text_color(64, 64, 64)
    if isinstance(inv_tuple, pd.DataFrame):
        add_dataframe_section(pdf, inv_tuple, "Inventory Aging (Top 10 Rows)")

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 10, "What does the above mean?", ln=True)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 8, (
        "This table highlights aging inventory:\n"
        "- Detects old stock needing clearance.\n"
        "- Aids in stock rotation planning.\n"
        "- Optimizes space and cash flow."))

    # Sales Forecasting
    sf = ml_tuple.get("Sales Forecasting", {})
    if sf:
        pdf.add_page()
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Sales Forecasting Summary", ln=True, align='C')
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
        pdf.cell(0, 10, "What does the above mean?", ln=True)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 8, (
            "This section predicts future sales:\n"
            "- Forecasts based on historical trends.\n"
            "- Guides inventory and campaign planning.\n"
            "- Prevents stockouts and overstocking."))

    # Customer Segmentation
    seg = ml_tuple.get("Customer Segmentation", {})
    if seg:
        pdf.add_page()
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Customer Segmentation Summary", ln=True, align='C')
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
        pdf.cell(0, 10, "What does the above mean?", ln=True)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 8, (
            "This section groups customers by behavior:\n"
            "- Uses clustering (e.g., K-Means).\n"
            "- Identifies loyal vs. one-time buyers.\n"
            "- Enables targeted marketing and offers."))

    # CLV
    clv = ml_tuple.get("Customer Lifetime Value", {})
    if clv:
        pdf.add_page()
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 10, "Customer Lifetime Value (CLV)", ln=True, align='C')
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
        pdf.cell(0, 10, "What does the above mean?", ln=True)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 8, (
            "This section estimates customer value:\n"
            "- CLV = Predicted total revenue per customer.\n"
            "- Prioritizes high-value customers.\n"
            "- Helps budget acquisition and retention."))

    # Final cleanup
    add_horizontal_line(pdf, thickness=0.9)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return pdf_bytes
