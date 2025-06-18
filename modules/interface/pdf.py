import os
from fpdf import FPDF
import pandas as pd

def pdf_generator_func(kpi_tuple, graph_tuple, inv_tuple, ml_tuple):
    """
    Generates a comprehensive Retail Analytics PDF report using KPIs, graphs, inventory data,
    and ML insights including Customer Lifetime Value (CLV), Customer Segmentation, and Sales Forecasting.
    """
    pdf = FPDF()
    pdf.add_page()
    image_paths = []

    # --- Title ---
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "Retail Analytics Report", ln=True, align='C')
    pdf.ln(10)

    # --- Horizontal Line Helper ---
    def add_horizontal_line(pdf_obj, thickness=0.5):
        pdf_obj.set_line_width(thickness)
        pdf_obj.line(10, pdf_obj.get_y(), 200, pdf_obj.get_y())
        pdf_obj.ln(3)

    # --- Helper: Draw DataFrame as Table ---
    def add_dataframe_section(pdf_obj, df, title, max_rows=5):
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

    # --- KPIs Section ---
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Key Performance Indicators (KPIs):", ln=True)
    pdf.set_font("Arial", "", 12)
    for label, value in kpi_tuple:
        pdf.cell(0, 8, f"{label}: {value}", ln=True)
    pdf.ln(8)

    # --- Graphs Section ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Visual Insights:", ln=True)
    pdf.ln(4)

    for idx, fig in enumerate(graph_tuple):
        img_path = f"temp_chart_{idx}.png"
        fig.write_image(img_path)
        image_paths.append(img_path)
        pdf.image(img_path, x=10, w=180)
        pdf.ln(5)

    # --- Inventory Aging Section ---
    pdf.add_page()
    if isinstance(inv_tuple, pd.DataFrame):
        add_dataframe_section(pdf, inv_tuple, "Inventory Aging (Top 5 Rows):")

    # -------------------------------
    # ---- SALES FORECASTING -------
    # -------------------------------
    pdf.add_page()
    sf = ml_tuple.get("Sales Forecasting", {})
    if sf:
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Sales Forecasting Summary:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.ln(3)

        reliability = sf.get("Reliability Percentage", 'N/A')
        reliability_str = f"{reliability:.2f}%" if isinstance(reliability, (int, float)) else str(reliability)
        if isinstance(reliability, (int, float)):
            label = "High" if reliability >= 75 else "Moderate" if reliability >= 50 else "Low"
            reliability_str = f"{reliability:.2f}% ({label} reliability)"

        pdf.cell(0, 8, f"Model Reliability: {reliability_str}", ln=True)
        pdf.cell(0, 8, f"Next Day: {sf.get('Next Day', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Next Day Prediction: {round(sf.get('Next Day Predictions', 0), 2)}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "TECHNICAL DETAILS OF MODEL RELIABILITY", ln=True)

        fig = sf.get("Line Chart Figure")
        if fig:
            img_path = "sales_forecast_fig.png"
            fig.write_image(img_path)
            pdf.image(img_path, x=10, w=180)
            image_paths.append(img_path)

    # -------------------------------
    # ---- CUSTOMER SEGMENTATION ---
    # -------------------------------
    pdf.add_page()
    seg = ml_tuple.get("Customer Segmentation", {})
    if seg:
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Customer Segmentation Summary:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.ln(3)

        reliability = seg.get("Reliability Percentage", 'N/A')
        reliability_str = f"{reliability:.2f}%" if isinstance(reliability, (int, float)) else str(reliability)
        if isinstance(reliability, (int, float)):
            label = "High" if reliability >= 75 else "Moderate" if reliability >= 50 else "Low"
            reliability_str = f"{reliability:.2f}% ({label} reliability)"

        pdf.cell(0, 8, f"Model Reliability: {reliability_str}", ln=True)
        pdf.cell(0, 8, f"Optimal Number of Segments (Best K): {seg.get('Best K', 'N/A')}", ln=True)

        add_dataframe_section(pdf, seg.get("Cluster Summary", pd.DataFrame()), "Cluster Summary")

        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
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

    # -------------------------------
    # ---- CUSTOMER LIFETIME VALUE -
    # -------------------------------
    pdf.add_page()
    clv = ml_tuple.get("Customer Lifetime Value", {})
    if clv:
        add_horizontal_line(pdf)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Customer Lifetime Value (CLV) Summary:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.ln(3)

        reliability = clv.get("Reliability", 'N/A')
        pdf.cell(0, 8, f"Model Reliability: {reliability}", ln=True)

        add_dataframe_section(pdf, clv.get("Sample Results", pd.DataFrame()), "Top 5 Predicted CLV Customers:")

        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
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

    # --- Export PDF as bytes ---
    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    # --- Cleanup temp images ---
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return pdf_bytes
