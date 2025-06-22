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
    print(cleaned_date_main_df.columns)


    return cleaned_date_main_df, datetime_list

@st.cache_data()
def complete_cleaner_func(cleaned_date_main_df, datetime_list):
    # create clean instance
    complete_clean = CleanerClass()

    # imputing the values for basic analytics
    basic_analytics_df = complete_clean.imputer_func(cleaned_date_main_df, datetime_list)
    print(basic_analytics_df.columns)

    return basic_analytics_df


class CleanerClass:
    def __init__(self):
        # GET datetime columns
        pass

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

        return cleaned_date_main_df, datetime_list

    def imputer_func(self, cleaned_date_main_df, datetime_list):
        """
        Imputes median and mode for numerical and categorical columns
        :param cleaned_date_main_df:
        :return: imputed cleaned_date_main_df
        """
        # create the copy of the dataframe that have cleaned datetime columns
        cleaned_df = cleaned_date_main_df.copy()

        # get id column
        id_col = [col for col in cleaned_df.columns if "ID" in col]

        # filter out the id column
        num_cat_col = [col for col in cleaned_df.columns if 'ID' not in col]

        # get numerical columns
        num_col = cleaned_df[num_cat_col].select_dtypes(include=['number']).columns.tolist()

        # categorical column
        cat_col = cleaned_df[num_cat_col].select_dtypes(include=['object', 'category']).columns.tolist()

        # get numeric column that have small unique values so will be converted to categorical
        numeric_but_cat = [col for col in num_col if cleaned_df[col].nunique() < 10]
        
        # extend the list
        for col in numeric_but_cat:
            cleaned_df[col] = cleaned_df[col].astype('object')
            cat_col.append(col)
            num_col.remove(col)

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
        processed_array = preprocessor.fit_transform(cleaned_date_main_df)

        # convert to dataframe
        processed_df = pd.DataFrame(processed_array, columns=num_col + cat_col)

        # add the id columns
        processed_final_df = pd.concat([cleaned_df[id_col].reset_index(drop=True), processed_df], axis=1)
        basic_analytics_df = pd.concat([processed_final_df, cleaned_df[datetime_list].reset_index(drop=True)], axis=1)
        print("Datetime columns being restored:", datetime_list)

        return basic_analytics_df


