from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
        :return: sales_forecast_dict,
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

            # create pipeline for preprocessing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

            # create the pipeline object
            pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])

            # fit the model
            pipeline.fit(X_train, y_train)

            # scores
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)

            # predicting the test set
            y_pred = pipeline.predict(X_test)

            # create dataframe of predictions
            sales_forecast_df = pd.DataFrame({
                'Actual Final Amount':y_test,
                'Predicted Final Amount': y_pred
            })

            # Add predictions back to the test set DataFrame (***IMPORTANT FOR GETTING THE DATAFRAME***)
            test_df = self.ml_df.loc[y_test.index].copy()
            test_df['Predicted Final Amount'] = y_pred
            test_df['Actual Final Amount'] = y_test  # ensure actual is aligned (redundant but explicit)
            # sort values
            test_df = test_df.sort_values(by='Date')

            # resample by month
            test_df.set_index('Date', inplace=True)
            monthly_df = test_df.resample('M').sum(numeric_only=True)
            monthly_df = monthly_df.reset_index()

            # Create line chart comparing actual and predicted
            fig = go.Figure()

            # Actual values (red)
            fig.add_trace(go.Scatter(
                x=monthly_df['Date'],
                y=monthly_df['Actual Final Amount'],
                mode='lines+markers',
                name='Actual Final Amount',
                line=dict(color='red')
            ))

            # Predicted values (blue)
            fig.add_trace(go.Scatter(
                x=monthly_df['Date'],
                y=monthly_df['Predicted Final Amount'],
                mode='lines+markers',
                name='Predicted Final Amount',
                line=dict(color='blue')
            ))

            fig.update_layout(
                title='Actual vs Predicted Final Amount (MONTHLY)',
                xaxis_title='Date',
                yaxis_title='Final Amount',
                template='plotly_white',
                legend_title='Legend'
            )

            # next day forecast
            last_day = ml_linear_regression_df['Date'].max()
            next_day = last_day + pd.Timedelta(days=1)
            next_day = next_day.date()
            # Calculate lifetime average of features (mean of all numerical columns except target)
            lifetime_avg_features = X.mean().values.reshape(1, -1)  # reshape for prediction (1 sample, n features)

            # Predict next day FinalAmount using lifetime averages
            next_day_prediction = pipeline.predict(lifetime_avg_features)[0]

        except Exception as e:
            print(e)
            log_error(str(e), source="ml_sales_forecasting_func in machinelearningoperations.py")
            sales_forecast_df = pd.DataFrame(columns=['Actual Final Amount', 'Predicted Final Amount'])
            fig = go.Figure()
            # exception handling
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
            'Line Chart Figure': fig
        }

        return sales_forecast_dict

    def ml_customer_segmentation_func(self, k_range=range(1, 11), plot=True):
        """
        Performs RFM-based customer segmentation using KMeans.
        Uses pipeline with scaling, finds optimal K using elbow (Plotly), performs PCA and returns all results.
        :param k_range: Range of K values to test (e.g., range(1, 11))
        :param plot: Whether to generate elbow plot in Plotly
        :return: Dictionary of results
        """
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from kneed import KneeLocator
        import plotly.graph_objects as go
        import plotly.express as px

        best_k = 0
        fig_cluster = None
        fig_elbow = None
        rfm = pd.DataFrame()
        summary_df = pd.DataFrame()
        reliability_percentage = 0

        try:
            # ✅ Use the ML dataframe
            ml_customer_segmentation_df = self.ml_df.copy()

            ml_customer_segmentation_df['Date'] = pd.to_datetime(ml_customer_segmentation_df['Date'])

            # Create FinalAmount if not present
            if 'FinalAmount' not in ml_customer_segmentation_df.columns:
                ml_customer_segmentation_df['FinalAmount'] = (
                        ml_customer_segmentation_df['Quantity'] * ml_customer_segmentation_df['UnitPrice']
                )

            snapshot_date = ml_customer_segmentation_df['Date'].max() + pd.Timedelta(days=1)

            # RFM computation
            rfm = ml_customer_segmentation_df.groupby('CustomerID').agg({
                'Date': lambda x: (snapshot_date - x.max()).days,
                'TransactionID': 'nunique',
                'FinalAmount': 'sum'
            }).reset_index()
            rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

            X = rfm[['Recency', 'Frequency', 'Monetary']]

            # Elbow Method
            wcss = []
            for k in k_range:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('kmeans', KMeans(n_clusters=k, random_state=42))
                ])
                pipeline.fit(X)
                inertia = pipeline.named_steps['kmeans'].inertia_
                wcss.append(inertia)

            # Find best k
            kl = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
            best_k = kl.elbow or 3

            # Plotly Elbow Plot
            if plot:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(k_range),
                    y=wcss,
                    mode='lines+markers',
                    name='WCSS',
                    line=dict(color='blue')
                ))
                fig_elbow.add_vline(
                    x=best_k,
                    line=dict(color='red', dash='dash'),
                    annotation_text=f"Elbow at k={best_k}",
                    annotation_position="top right"
                )
                fig_elbow.update_layout(
                    title='Elbow Method - Optimal K',
                    xaxis_title='Number of Clusters (K)',
                    yaxis_title='WCSS (Inertia)',
                    template='plotly_white'
                )

            # Final Model
            final_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('kmeans', KMeans(n_clusters=best_k, random_state=42))
            ])
            rfm['Segment'] = final_pipeline.fit_predict(X)

            # PCA for 2D visualization
            scaled_X = final_pipeline.named_steps['scaler'].transform(X)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_X)
            rfm['PCA1'] = pca_result[:, 0]
            rfm['PCA2'] = pca_result[:, 1]

            fig_cluster = px.scatter(
                rfm, x='PCA1', y='PCA2', color='Segment',
                hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
                title=f'Customer Segmentation with K={best_k}',
                template='plotly_white'
            )

            summary_df = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
            summary_df['Number of Customers'] = rfm['Segment'].value_counts().sort_index()
            summary_df = summary_df.reset_index()

            # Calculate Davies-Bouldin Index
            dbi_score = davies_bouldin_score(scaled_X, rfm['Segment'])


            # Convert to reliability %
            reliability_percentage = round(max(0.0, 1 - (dbi_score / 2)) * 100, 2)

            # Determine quality label
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

            # Combine into a single string
            reliability_score_combined = f"{reliability_percentage}% ({reliability_label} reliability)"

        except Exception as e:
            print(e)
            log_error(str(e), source="customer_segmentation_func (Plotly Version)")
            # Create empty fallback figures
            reliability_score_combined = "N/A"
            fig_elbow = go.Figure()
            fig_elbow.update_layout(title="Error: Elbow plot not available")
            fig_cluster = go.Figure()
            fig_cluster.update_layout(title="Error: Cluster plot not available")



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

        :return:
        """
        pass

