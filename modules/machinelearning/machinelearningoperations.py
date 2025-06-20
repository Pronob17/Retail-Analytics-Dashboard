from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from tests.errorlog import log_error



# function for storing the caches
@st.cache_data
def cache_func(cleaned_date_main_df):
    """
    Function to cache the stored files
    :param cleaned_date_main_df:
    :return:sales_forecast_dict, customer_segment_dict, clv_dict
    """
    # create machine learning instance
    machine = MachineLearningClass(cleaned_date_main_df)

    # calling the respective functions
    sales_forecast_dict = machine.ml_sales_forecasting_func()
    customer_segment_dict = machine.ml_customer_segmentation_func()
    customer_lifetime_value_dict = machine.ml_customer_lifetime_value_func()

    return sales_forecast_dict, customer_segment_dict, customer_lifetime_value_dict


class MachineLearningClass:
    def __init__(self, cleaned_date_main_df):
        self.ml_df = cleaned_date_main_df.copy()

    def ml_sales_forecasting_func(self):
        """
        Imputes and scales the cleaned_date_main_df dataframe using SimpleImputer for the numerical columns and scales it too.
        Calculates the sales forecasting for test data and most importantly, the next day's Final Amount.
        :return: sales_forecast_dict
        """
        ml_linear_regression_df = self.ml_df.copy()

        # only take the numerical column
        num_col = ml_linear_regression_df.select_dtypes(include=['number']).columns.tolist()

        # create new df with only numerical column for ml model
        ml_final_df = ml_linear_regression_df[num_col].copy()

        # split the data for target and feature
        try:
            X = ml_final_df.drop('FinalAmount', axis=1)
            y = ml_final_df['FinalAmount']

            # train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

            # pipeline
            pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])

            # fit
            pipeline.fit(X_train, y_train)

            # scores
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)

            # prediction
            y_pred = pipeline.predict(X_test)

            # prediction dataframe
            sales_forecast_df = pd.DataFrame({
                'Actual Final Amount': y_test,
                'Predicted Final Amount': y_pred
            })

            # attach predictions to test set
            test_df = self.ml_df.loc[y_test.index].copy()
            test_df['Predicted Final Amount'] = y_pred
            test_df['Actual Final Amount'] = y_test
            test_df = test_df.sort_values(by='Date')
            test_df.set_index('Date', inplace=True)

            # monthly resample
            monthly_df = test_df.resample('M').sum(numeric_only=True).reset_index()

            # seaborn lineplot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=monthly_df, x='Date', y='Actual Final Amount', marker='o',
                         label='Actual Final Amount', color='red', ax=ax)
            sns.lineplot(data=monthly_df, x='Date', y='Predicted Final Amount', marker='o',
                         label='Predicted Final Amount', color='blue', ax=ax)

            ax.set_title('Actual vs Predicted Final Amount (MONTHLY)', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Final Amount', fontsize=12)
            ax.legend(title='Legend')
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # next day forecast
            last_day = ml_linear_regression_df['Date'].max()
            next_day = (last_day + pd.Timedelta(days=1)).date()

            # average features for prediction
            lifetime_avg_features = X.mean().values.reshape(1, -1)
            next_day_prediction = pipeline.predict(lifetime_avg_features)[0]

        except Exception as e:
            print(e)
            log_error(str(e), source="ml_sales_forecasting_func in machinelearningoperations.py")
            sales_forecast_df = pd.DataFrame(columns=['Actual Final Amount', 'Predicted Final Amount'])
            fig, ax = plt.subplots()
            train_score = 0
            test_score = 0
            next_day = 0
            next_day_prediction = 0

        sales_forecast_dict = {
            'Train R2 Score': round(train_score, 4),
            'Test R2 Score': round(test_score, 4),
            'Sales Forecast Dataframe': sales_forecast_df,
            'Reliability Percentage': round(((train_score + test_score) / 2), 2) * 100,
            'Next Day': next_day,
            'Next Day Predictions': next_day_prediction,
            'Line Chart Figure': fig  # You can save this or render it in Streamlit using st.pyplot(fig)
        }

        return sales_forecast_dict

    def ml_customer_segmentation_func(self, k_range=range(1, 11), plot=True):
        """
        Performs RFM-based customer segmentation using KMeans.
        Uses pipeline with scaling, finds optimal K using elbow (Seaborn), performs PCA and returns all results.
        :param k_range: Range of K values to test (e.g., range(1, 11))
        :param plot: Whether to generate elbow plot
        :return: Dictionary of results
        """
        best_k = 0
        fig_elbow = None
        fig_cluster = None
        rfm = pd.DataFrame()
        summary_df = pd.DataFrame()
        reliability_percentage = 0

        try:
            # Load and clean
            ml_customer_segmentation_df = self.ml_df.copy()
            ml_customer_segmentation_df['Date'] = pd.to_datetime(ml_customer_segmentation_df['Date'])

            if 'FinalAmount' not in ml_customer_segmentation_df.columns:
                ml_customer_segmentation_df['FinalAmount'] = (
                        ml_customer_segmentation_df['Quantity'] * ml_customer_segmentation_df['UnitPrice']
                )

            snapshot_date = ml_customer_segmentation_df['Date'].max() + pd.Timedelta(days=1)

            # RFM Computation
            rfm = ml_customer_segmentation_df.groupby('CustomerID').agg({
                'Date': lambda x: (snapshot_date - x.max()).days,
                'TransactionID': 'nunique',
                'FinalAmount': 'sum'
            }).reset_index()
            rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

            X = rfm[['Recency', 'Frequency', 'Monetary']]

            # Elbow method
            wcss = []
            for k in k_range:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('kmeans', KMeans(n_clusters=k, random_state=42))
                ])
                pipeline.fit(X)
                inertia = pipeline.named_steps['kmeans'].inertia_
                wcss.append(inertia)

            kl = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
            best_k = kl.elbow or 3

            # Elbow Plot using Seaborn
            if plot:
                fig_elbow, ax1 = plt.subplots(figsize=(8, 5))
                sns.lineplot(x=list(k_range), y=wcss, marker='o', ax=ax1, color='blue', label='WCSS')
                ax1.axvline(x=best_k, color='red', linestyle='--', label=f'Elbow at k={best_k}')
                ax1.set_title('Elbow Method - Optimal K')
                ax1.set_xlabel('Number of Clusters (K)')
                ax1.set_ylabel('WCSS (Inertia)')
                ax1.legend()
                ax1.grid(True)
                plt.tight_layout()

            # Final clustering
            final_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('kmeans', KMeans(n_clusters=best_k, random_state=42))
            ])
            rfm['Segment'] = final_pipeline.fit_predict(X)

            # PCA
            scaled_X = final_pipeline.named_steps['scaler'].transform(X)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_X)
            rfm['PCA1'] = pca_result[:, 0]
            rfm['PCA2'] = pca_result[:, 1]

            # Cluster Scatterplot using Seaborn
            fig_cluster, ax2 = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Segment', palette='tab10', ax=ax2, s=60)
            ax2.set_title(f'Customer Segmentation with K={best_k}')
            ax2.set_xlabel('PCA Component 1')
            ax2.set_ylabel('PCA Component 2')
            ax2.legend(title='Segment')
            ax2.grid(True)
            plt.tight_layout()

            # Segment Summary
            summary_df = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
            summary_df['Number of Customers'] = rfm['Segment'].value_counts().sort_index()
            summary_df = summary_df.reset_index()

            # Reliability Score
            dbi_score = davies_bouldin_score(scaled_X, rfm['Segment'])
            reliability_percentage = round(max(0.0, 1 - (dbi_score / 2)) * 100, 2)

            if dbi_score < 0.5:
                reliability_label = "Excellent"
            elif dbi_score < 0.8:
                reliability_label = "Good"
            elif dbi_score < 1.2:
                reliability_label = "Moderate"
            elif dbi_score < 1.6:
                reliability_label = "Weak"
            else:
                reliability_label = "Poor"

            reliability_score_combined = f"{reliability_percentage}% ({reliability_label} reliability)"

        except Exception as e:
            print(e)
            log_error(str(e), source="customer_segmentation_func (Seaborn Version)")
            fig_elbow, ax1 = plt.subplots()
            ax1.set_title("Error: Elbow plot not available")
            fig_cluster, ax2 = plt.subplots()
            ax2.set_title("Error: Cluster plot not available")
            reliability_score_combined = "N/A"

        customer_segment_dict = {
            'Best K': best_k,
            'Segmented RFM Dataframe': rfm,
            'Cluster Summary': summary_df,
            'Elbow Plot Figure': fig_elbow,
            'Scatter Plot Figure': fig_cluster,
            'Reliability Percentage': reliability_score_combined
        }

        return customer_segment_dict

    def ml_customer_lifetime_value_func(self):
        """
        Calculates Customer Lifetime Value (CLV) using a regression model.
        Returns a dictionary with predictions, reliability, and visualization.
        """
        clv_dict = {
            'Sample Results': pd.DataFrame(),
            'fig_hist': None,
            'Reliability': 'N/A',
            'r2_train': "N/A",
            'r2_test': "N/A",
            'mae': "N/A",
            'mse': "N/A",
        }

        try:
            df = self.ml_df.copy()
            required_cols = ['CustomerID', 'TransactionID', 'Date', 'Quantity', 'UnitPrice', 'DiscountPercent']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])

            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
            df['DiscountPercent'] = pd.to_numeric(df['DiscountPercent'], errors='coerce')
            df = df.dropna(subset=['Quantity', 'UnitPrice', 'DiscountPercent'])

            if df.empty:
                raise ValueError("Cleaned dataframe is empty.")

            df['FinalAmount'] = df['Quantity'] * df['UnitPrice'] * (1 - df['DiscountPercent'] / 100)

            customer_df = df.groupby('CustomerID').agg({
                'TransactionID': 'nunique',
                'FinalAmount': 'sum',
                'DiscountPercent': 'mean',
                'Quantity': 'mean',
                'Date': [lambda x: (x.max() - x.min()).days, 'count']
            }).reset_index()

            customer_df.columns = [
                'CustomerID', 'Frequency', 'Monetary', 'AvgDiscount',
                'AvgQuantity', 'CustomerAge', 'TotalTransactions'
            ]

            customer_df['CustomerAge'] = customer_df['CustomerAge'].fillna(0)
            customer_df['AvgDiscount'] = customer_df['AvgDiscount'].fillna(0)
            customer_df['AvgQuantity'] = customer_df['AvgQuantity'].fillna(0)

            if customer_df.shape[0] < 3:
                raise ValueError("Insufficient unique customers for CLV modeling.")

            X = customer_df[['CustomerAge', 'Frequency', 'AvgDiscount', 'AvgQuantity']]
            y = customer_df['Monetary']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            if len(X_train) < 2 or len(X_test) < 1:
                raise ValueError("Insufficient training or testing data.")

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)

            reliability_label = "Good" if r2_test > 0.6 else "Low"
            clv_dict['Reliability'] = f"{round(r2_test * 100, 2)}% ({reliability_label} reliability)"
            clv_dict['r2_train'] = round(r2_train, 4)
            clv_dict['r2_test'] = round(r2_test, 4)
            clv_dict['mae'] = round(mae, 2)
            clv_dict['mse'] = round(mse, 2)

            customer_df['CLV_Predicted'] = model.predict(X)

            # ---------- Overlay Histogram using Seaborn ----------
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.histplot(customer_df['Monetary'], kde=False, color='blue', label='Actual CLV', bins=30, alpha=0.6,
                         ax=ax)
            sns.histplot(customer_df['CLV_Predicted'], kde=False, color='teal', label='Predicted CLV', bins=30,
                         alpha=0.6, ax=ax)

            ax.set_title("Actual vs Predicted CLV Distribution", fontsize=14)
            ax.set_xlabel("CLV Value", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            clv_dict['fig_hist'] = fig

            clv_dict['Sample Results'] = customer_df[['CustomerID', 'CLV_Predicted']].sort_values(
                by='CLV_Predicted', ascending=False).head(10)

        except Exception as e:
            import traceback
            traceback.print_exc()
            log_error(str(e), source="ml_customer_lifetime_value_func")

        return clv_dict











