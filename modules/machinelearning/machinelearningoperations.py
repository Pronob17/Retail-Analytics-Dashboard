from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from tests.errorlog import log_error




class MachineLearningClass:
    def __init__(self, cleaned_date_main_df):
        self.ml_df = cleaned_date_main_df

    def ml_sales_forecasting_func(self):
        """
        Imputes and scales the cleaned_date_main_df dataframe using SimpleImputer for the numerical columns and scales it too.
        :param
        :return: ml_cleaned_df
        """
        # initializing train and test score
        train_score = 0
        test_score = 0
        next_day = 0
        next_day_prediction = 0

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

        sales_forecast_dict = {
            'Train R2 Score': round(train_score, 4),
            'Test R2 Score': round(test_score, 4),
            'Sales Forecast Dataframe': sales_forecast_df,
            'Reliability Percentage': round(((train_score + test_score)/2)) * 100,
            'Next Day': next_day,
            'Next Day Predictions': next_day_prediction,
            'Line Chart Figure': fig
        }

        return sales_forecast_dict

