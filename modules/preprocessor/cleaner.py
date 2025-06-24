import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from tests.errorlog import log_error


@st.cache_data()
def datetime_cleaner_func(main_df):
    # create clean instance
    clean_date = CleanerClass()

    # pass the dataframe clean datetime of the dataframe (cleaned_date_main_df will be used in ml too)
    cleaned_date_main_df, datetime_list = clean_date.datetime_func(main_df)

    return cleaned_date_main_df, datetime_list


@st.cache_data()
def complete_cleaner_func(cleaned_date_main_df, datetime_list):
    # create clean instance
    complete_clean = CleanerClass()

    # imputing the values for basic analytics
    basic_analytics_df = complete_clean.imputer_func(cleaned_date_main_df, datetime_list)

    return basic_analytics_df


class CleanerClass:
    def __init__(self):
        # GET datetime columns
        self.num_col = []
        self.cat_col = []

    def datetime_func(self, main_df):
        """
        Convert the dates to proper datetime format.
        :param main_df:
        :return: cleaned_date_df
        """
        datetime_list = []
        # create a copy of the original dataframe
        cleaned_date_main_df = main_df.copy()

        # take each column
        for col in cleaned_date_main_df.columns:
            # Drop nulls and convert to string
            non_null_data = cleaned_date_main_df[col].dropna().astype(str)

            # Sample min(20, available rows)
            sample_size = min(20, len(non_null_data))
            sample_rows = non_null_data.sample(sample_size)

            count = 0

            if pd.api.types.is_object_dtype(cleaned_date_main_df[col]):
                for item in sample_rows:
                    try:
                        pd.to_datetime(item)
                        count += 1
                    except Exception:
                        continue

                if count / sample_size >= 0.8:
                    datetime_list.append(col)

        # convert datetime_list to datetime
        for dt in datetime_list:
            cleaned_date_main_df[dt] = pd.to_datetime(cleaned_date_main_df[dt], errors='coerce', infer_datetime_format=True)

        # SORT OUT THE DATATYPES FOR NUMERICAL AND CATEGORICAL
            # get id column
            id_col = [col for col in cleaned_date_main_df.columns if "ID" in col]

            # filter out the id column
            except_id_col = [col for col in cleaned_date_main_df.columns if 'ID' not in col]

            # filter out the date column
            num_cat_col = [col for col in except_id_col if col not in datetime_list]

            # convert the convertable numerical column
            for col in num_cat_col:
                converted = pd.to_numeric(cleaned_date_main_df[col], errors='coerce')

                # numerical ration
                numeric_ratio = converted.notna().sum() / len(cleaned_date_main_df)

                # check 50% threshold and then convert to numeric if greater than that
                if numeric_ratio > 0.5:
                    cleaned_date_main_df[col] = converted
                    self.num_col.append(col)
                else:
                    self.cat_col.append(col)

        # Ensure key columns have correct types
        cleaned_date_main_df["TransactionID"] = cleaned_date_main_df["TransactionID"].astype(str)
        cleaned_date_main_df["CustomerID"] = cleaned_date_main_df["CustomerID"].astype(str)

        return cleaned_date_main_df, datetime_list

    def imputer_func(self, cleaned_date_main_df, datetime_list):
        """
        Imputes median and mode for numerical and categorical columns
        :param cleaned_date_main_df:
        :param datetime_list:
        :return: basic_analytics_df
        """
        # create the copy of the dataframe that have cleaned datetime columns
        cleaned_df = cleaned_date_main_df.copy()

        # get id column
        id_col = [col for col in cleaned_df.columns if "ID" in col]

        # filter out the id column
        except_id_col = [col for col in cleaned_df.columns if 'ID' not in col]

        # filter out the date column
        num_cat_col = [col for col in except_id_col if col not in datetime_list]

        # convert the convertable numerical column
        num_col = []
        cat_col = []
        for col in num_cat_col:
            converted = pd.to_numeric(cleaned_df[col], errors='coerce')

            # numerical ration
            numeric_ratio = converted.notna().sum() / len(cleaned_df)

            # check 50% threshold and then convert to numeric if greater than that
            if numeric_ratio > 0.5:
                cleaned_df[col] = converted
                num_col.append(col)
            else:
                cat_col.append(col)

        # STARTING THE IMPUTATION
        # numerical transformation
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        # categorical transformation
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        # full column transformer
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, num_col),
            ('cat', categorical_transformer, cat_col)
        ])

        # fit and transform the data
        processed_array = preprocessor.fit_transform(cleaned_df)

        # convert to dataframe
        processed_df = pd.DataFrame(processed_array, columns=num_col + cat_col)

        # add the id columns
        processed_final_df = pd.concat([cleaned_df[id_col].reset_index(drop=True), processed_df], axis=1)
        basic_analytics_df = pd.concat([processed_final_df, cleaned_df[datetime_list].reset_index(drop=True)], axis=1)

        # Ensure key columns have correct types
        cleaned_df["TransactionID"] = cleaned_df["TransactionID"].astype(str)
        cleaned_df["CustomerID"] = cleaned_df["CustomerID"].astype(str)

        return basic_analytics_df


